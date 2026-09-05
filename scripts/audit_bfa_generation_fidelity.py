"""Compare generated BFA windows with their real training anchors and distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = np.asarray(list("ABCDEFGHIJKLMNOPQRST"))
MOD = np.asarray([512, 512, 128, 128], dtype=np.int32)


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64) + 1e-12
    right = right.astype(np.float64) + 1e-12
    left /= left.sum()
    right /= right.sum()
    middle = 0.5 * (left + right)
    return float(
        0.5 * np.sum(left * np.log2(left / middle))
        + 0.5 * np.sum(right * np.log2(right / middle))
    )


def signed_delta(values: np.ndarray, modulus: int) -> np.ndarray:
    delta = np.diff(values.astype(np.int32), axis=1)
    return (delta + modulus // 2) % modulus - modulus // 2


def histogram(values: np.ndarray, modulus: int) -> np.ndarray:
    return np.histogram(values, bins=64, range=(-modulus / 2, modulus / 2))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real_npz", type=Path)
    parser.add_argument("generated_npz", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()

    with np.load(args.real_npz, allow_pickle=False) as data:
        real_x = data["x"]
        real_y = data["y"].astype(np.int64)
    with np.load(args.generated_npz, allow_pickle=False) as data:
        generated_x = data["x"]
        generated_y = data["y"].astype(np.int64)
        if "allocation_real_index" not in data.files:
            raise SystemExit("Generated NPZ lacks allocation_real_index")
        allocation = data["allocation_real_index"].astype(np.int64)

    if generated_x.shape[1:] != (10, 234, 4):
        raise SystemExit(f"Unexpected generated shape: {generated_x.shape}")
    if not (len(generated_x) == len(generated_y) == len(allocation)):
        raise SystemExit("Generated arrays have different lengths")
    if np.any((allocation < 0) | (allocation >= len(real_x))):
        raise SystemExit("allocation_real_index is out of range")
    paired_real = real_x[allocation]
    if np.any(real_y[allocation] != generated_y):
        raise SystemExit("Generated labels do not match allocated real labels")

    report: dict[str, object] = {
        "real_npz": str(args.real_npz),
        "generated_npz": str(args.generated_npz),
        "samples": len(generated_x),
        "shape": list(generated_x.shape),
        "dtype": str(generated_x.dtype),
        "finite": bool(np.isfinite(generated_x).all()),
        "anchor_exact_match_rate": float(
            np.mean(np.all(generated_x[:, 0] == paired_real[:, 0], axis=(1, 2)))
        ),
        "paired_exact_window_rate": float(
            np.mean(np.all(generated_x == paired_real, axis=(1, 2, 3)))
        ),
        "class_counts": {
            name: int(count)
            for name, count in zip(
                CLASS_NAMES, np.bincount(generated_y, minlength=len(CLASS_NAMES))
            )
        },
    }

    channel_reports = []
    real_temporal = np.empty((len(generated_x), 4), dtype=np.float64)
    generated_temporal = np.empty_like(real_temporal)
    for channel, modulus in enumerate(MOD):
        real_delta = signed_delta(paired_real[..., channel], int(modulus))
        generated_delta = signed_delta(generated_x[..., channel], int(modulus))
        real_abs = np.abs(real_delta)
        generated_abs = np.abs(generated_delta)
        real_temporal[:, channel] = real_abs.mean(axis=(1, 2))
        generated_temporal[:, channel] = generated_abs.mean(axis=(1, 2))

        paired_error = signed_delta(
            np.stack((real_delta, generated_delta), axis=1), int(modulus)
        )[:, 0]
        class_js = []
        for label in range(len(CLASS_NAMES)):
            rows = generated_y == label
            if np.any(rows):
                class_js.append(
                    js_divergence(
                        histogram(real_delta[rows], int(modulus)),
                        histogram(generated_delta[rows], int(modulus)),
                    )
                )
        real_mean = float(real_abs.mean())
        generated_mean = float(generated_abs.mean())
        channel_reports.append(
            {
                "channel": channel,
                "modulus": int(modulus),
                "real_abs_delta_mean": real_mean,
                "generated_abs_delta_mean": generated_mean,
                "mean_ratio": generated_mean / max(real_mean, 1e-12),
                "real_abs_delta_std": float(real_abs.std()),
                "generated_abs_delta_std": float(generated_abs.std()),
                "real_abs_delta_p95": float(np.percentile(real_abs, 95)),
                "generated_abs_delta_p95": float(np.percentile(generated_abs, 95)),
                "delta_js_divergence": js_divergence(
                    histogram(real_delta, int(modulus)),
                    histogram(generated_delta, int(modulus)),
                ),
                "mean_class_conditional_delta_js": float(np.mean(class_js)),
                "paired_delta_mae_normalized": float(
                    np.mean(np.abs(paired_error)) / max(float(real_delta.std()), 1e-12)
                ),
            }
        )
        del real_delta, generated_delta, real_abs, generated_abs, paired_error

    temporal_z = np.empty_like(generated_temporal)
    for label in range(len(CLASS_NAMES)):
        rows = generated_y == label
        if not np.any(rows):
            continue
        mean = real_temporal[rows].mean(axis=0)
        std = np.maximum(real_temporal[rows].std(axis=0), 1e-3)
        temporal_z[rows] = np.abs(generated_temporal[rows] - mean) / std

    report["channels"] = channel_reports
    report["mean_temporal_z"] = float(temporal_z.mean())
    report["temporal_z_p95"] = float(np.percentile(temporal_z.mean(axis=1), 95))
    report["mean_delta_js_divergence"] = float(
        np.mean([entry["delta_js_divergence"] for entry in channel_reports])
    )
    report["mean_class_conditional_delta_js"] = float(
        np.mean(
            [entry["mean_class_conditional_delta_js"] for entry in channel_reports]
        )
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
