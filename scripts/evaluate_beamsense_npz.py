"""Evaluate a trained BeamSense Keras model on a labeled BFA NPZ file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


CLASS_NAMES = np.asarray(list("ABCDEFGHIJKLMNOPQRST"))
ANGLE_SCALE = np.asarray([511.0, 511.0, 127.0, 127.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="BeamSense .keras checkpoint")
    parser.add_argument("npz", type=Path, help="labeled BFA NPZ containing x and y")
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--normalize", choices=("none", "angle-range"), default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import tensorflow as tf

    args.output.mkdir(parents=True, exist_ok=True)
    data = np.load(args.npz, mmap_mode="r", allow_pickle=False)
    if "x" not in data.files or "y" not in data.files:
        raise SystemExit("NPZ must contain x and y")
    x = data["x"]
    y = np.asarray(data["y"], dtype=np.int64)
    if x.shape[1:] != (10, 234, 4):
        raise SystemExit(f"Expected x shape (N,10,234,4), got {x.shape}")
    if len(x) != len(y):
        raise SystemExit(f"x/y length mismatch: {len(x)} != {len(y)}")
    if np.any((y < 0) | (y >= 20)):
        raise SystemExit("Labels must be zero-based integers in [0, 19]")

    class EvaluationSequence(tf.keras.utils.Sequence):
        def __len__(self):
            return int(np.ceil(len(x) / args.batch_size))

        def __getitem__(self, batch):
            start = batch * args.batch_size
            stop = min(start + args.batch_size, len(x))
            features = x[start:stop].astype(np.float32)
            if args.normalize == "angle-range":
                features /= ANGLE_SCALE
            return features

    model = tf.keras.models.load_model(args.model)
    probabilities = model.predict(EvaluationSequence(), verbose=1)
    prediction = np.argmax(probabilities, axis=1)
    true_counts = np.bincount(y, minlength=20)
    prediction_counts = np.bincount(prediction, minlength=20)
    metrics = {
        "model": str(args.model),
        "data": str(args.npz),
        "samples": len(y),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "macro_recall": float(
            recall_score(y, prediction, average="macro", zero_division=0)
        ),
        "normalization": args.normalize,
        "true_class_counts": {
            name: int(count) for name, count in zip(CLASS_NAMES, true_counts)
        },
        "prediction_class_counts": {
            name: int(count) for name, count in zip(CLASS_NAMES, prediction_counts)
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "predictions.npz",
        truth=y,
        prediction=prediction,
        probabilities=probabilities,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
