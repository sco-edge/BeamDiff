"""Summarize a BeamDiff augmentation-ratio sweep across model seeds.

Expected directory layout::

    ROOT/ratio0_seed42/random_window_metrics.json
    ROOT/ratio0.25_seed42/random_window_metrics.json
    ROOT/ratio0.5_seed42/random_window_metrics.json
    ...

The script writes CSV, JSON, and a paper-ready LaTeX table.  All gains are
paired against the real-only result from the same model seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = ("accuracy", "macro_f1", "macro_recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="ratio sweep result directory")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
        help="augmentation ratios to summarize",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(42, 111, 2026, 3407, 7777),
        help="classifier initialization seeds",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="output path without extension; default: ROOT/beamdiff_ratio_summary",
    )
    parser.add_argument(
        "--baseline-template",
        help=(
            "optional real-only metrics path containing {seed}; when supplied, "
            "ratio 0 is loaded from this template instead of ROOT/ratio0_seed{seed}"
        ),
    )
    return parser.parse_args()


def ratio_name(ratio: float) -> str:
    return f"{ratio:g}"


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_result(
    root: Path, ratio: float, seed: int, baseline_template: str | None = None
) -> dict:
    if ratio == 0.0 and baseline_template:
        path = Path(baseline_template.format(seed=seed))
        candidates = [path]
    else:
        names = list(dict.fromkeys((ratio_name(ratio), str(float(ratio)))))
        candidates = [
            root / f"ratio{name}_seed{seed}" / "random_window_metrics.json"
            for name in names
        ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        expected = " or ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Missing result: {expected}")
    result = json.loads(path.read_text())
    for metric in METRICS:
        if metric not in result or not math.isfinite(float(result[metric])):
            raise ValueError(f"Invalid {metric} in {path}")
    if int(result.get("seed", seed)) != seed:
        raise ValueError(f"Model seed mismatch in {path}")
    return result


def summarize(
    root: Path,
    ratios: list[float],
    seeds: list[int],
    baseline_template: str | None = None,
) -> tuple[list[dict], dict]:
    runs: dict[float, dict[int, dict]] = {
        ratio: {
            seed: load_result(root, ratio, seed, baseline_template)
            for seed in seeds
        }
        for ratio in ratios
    }
    if 0.0 not in runs:
        raise ValueError("Ratios must include 0 for paired baseline comparisons")

    baseline = runs[0.0]
    rows = []
    for ratio in ratios:
        results = runs[ratio]
        row: dict[str, object] = {
            "ratio": ratio,
            "ratio_percent": ratio * 100,
            "seeds": len(seeds),
            "synthetic_samples": round(
                statistics.mean(float(results[s].get("synthetic_train_samples", 0)) for s in seeds)
            ),
        }
        for metric in METRICS:
            values = [float(results[s][metric]) * 100 for s in seeds]
            gains = [
                (float(results[s][metric]) - float(baseline[s][metric])) * 100
                for s in seeds
            ]
            row[f"{metric}_mean_percent"] = statistics.mean(values)
            row[f"{metric}_sd_percent"] = sample_sd(values)
            row[f"{metric}_gain_mean_pp"] = statistics.mean(gains)
            row[f"{metric}_gain_sd_pp"] = sample_sd(gains)
        accuracy_gains = [
            float(results[s]["accuracy"]) - float(baseline[s]["accuracy"])
            for s in seeds
        ]
        row["accuracy_seeds_improved"] = sum(gain > 0 for gain in accuracy_gains)
        rows.append(row)
    return rows, runs


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Downstream sensing performance at different BeamDiff augmentation ratios.}",
        r"\label{tab:beamdiff_ratio}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Ratio & Synthetic & Accuracy & $\Delta$Acc. & Macro-F1 & Improved \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['ratio_percent']:.0f}\\% & "
            f"{row['synthetic_samples']:,} & "
            f"{row['accuracy_mean_percent']:.2f} $\\pm$ {row['accuracy_sd_percent']:.2f} & "
            f"{row['accuracy_gain_mean_pp']:+.2f} pp & "
            f"{row['macro_f1_mean_percent']:.2f} $\\pm$ {row['macro_f1_sd_percent']:.2f} & "
            f"{row['accuracy_seeds_improved']}/{row['seeds']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'Ratio':>7}  {'Synthetic':>9}  {'Accuracy (mean±SD)':>22}  "
        f"{'Gain':>10}  {'Macro-F1 (mean±SD)':>22}  {'Improved':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['ratio_percent']:6.0f}%  "
            f"{row['synthetic_samples']:9,d}  "
            f"{row['accuracy_mean_percent']:8.2f}±{row['accuracy_sd_percent']:<8.2f}  "
            f"{row['accuracy_gain_mean_pp']:+8.2f}pp  "
            f"{row['macro_f1_mean_percent']:8.2f}±{row['macro_f1_sd_percent']:<8.2f}  "
            f"{row['accuracy_seeds_improved']}/{row['seeds']:>1}"
        )


def main() -> None:
    args = parse_args()
    ratios = list(dict.fromkeys(args.ratios))
    seeds = list(dict.fromkeys(args.seeds))
    if any(ratio < 0 for ratio in ratios):
        raise SystemExit("Ratios must be nonnegative")
    rows, runs = summarize(args.root, ratios, seeds, args.baseline_template)
    prefix = args.output_prefix or args.root / "beamdiff_ratio_summary"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(prefix.with_suffix(".csv"), rows)
    write_latex(prefix.with_suffix(".tex"), rows)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "root": str(args.root.resolve()),
                "ratios": ratios,
                "seeds": seeds,
                "baseline_template": args.baseline_template,
                "summary": rows,
            },
            indent=2,
        )
    )
    print_table(rows)
    print(f"\nSaved: {prefix.with_suffix('.csv')}")
    print(f"Saved: {prefix.with_suffix('.tex')}")
    print(f"Saved: {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
