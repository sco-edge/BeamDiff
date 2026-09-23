"""Plot a BeamDiff ratio sweep evaluated with BeamSense."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_LABELS = {
    "accuracy": "Accuracy (%)",
    "macro_f1": "Macro-F1 (%)",
    "macro_recall": "Macro Recall (%)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("beamdiff_csv", type=Path)
    parser.add_argument("output_prefix", type=Path, help="output path without extension")
    parser.add_argument("--metric", choices=tuple(METRIC_LABELS), default="accuracy")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--title",
        default=None,
        help="optional figure title; omit for a compact paper figure",
    )
    return parser.parse_args()


def load(path: Path, metric: str) -> tuple[list[float], list[float], list[float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty summary CSV: {path}")
    x = [float(row["ratio_percent"]) for row in rows]
    mean = [float(row[f"{metric}_mean_percent"]) for row in rows]
    sd = [float(row[f"{metric}_sd_percent"]) for row in rows]
    order = sorted(range(len(x)), key=x.__getitem__)
    return (
        [x[index] for index in order],
        [mean[index] for index in order],
        [sd[index] for index in order],
    )


def main() -> None:
    args = parse_args()
    if args.dpi < 72:
        raise SystemExit("--dpi must be at least 72")

    import matplotlib.pyplot as plt

    beam_x, beam_mean, beam_sd = load(args.beamdiff_csv, args.metric)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    axis.errorbar(
        beam_x,
        beam_mean,
        yerr=beam_sd,
        color="#1f5a99",
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        capsize=3,
        label="BeamSense with BeamDiff augmentation",
    )

    all_low = [mean - sd for mean, sd in zip(beam_mean, beam_sd)]
    all_high = [mean + sd for mean, sd in zip(beam_mean, beam_sd)]
    lower = max(0.0, min(all_low) - 1.5)
    upper = min(100.0, max(all_high) + 1.5)
    if upper - lower < 5:
        middle = (upper + lower) / 2
        lower, upper = max(0, middle - 2.5), min(100, middle + 2.5)

    axis.set_xlabel("Augmented Data (%)")
    axis.set_ylabel(METRIC_LABELS[args.metric])
    axis.set_xticks(beam_x, [f"+{value:g}%" for value in beam_x])
    axis.set_xlim(min(beam_x) - 3, max(beam_x) + 3)
    axis.set_ylim(lower, upper)
    axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    if args.title:
        axis.set_title(args.title)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    for ratio, mean in zip(beam_x, beam_mean):
        axis.annotate(
            f"{mean:.2f}",
            (ratio, mean),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = args.output_prefix.with_suffix(".png")
    pdf = args.output_prefix.with_suffix(".pdf")
    figure.savefig(png, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
