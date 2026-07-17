"""Train the BeamSense 2-D CNN on a HAR-1 BFA NPZ dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np


CLASS_NAMES = np.asarray(list("ABCDEFGHIJKLMNOPQRST"))
ANGLE_SCALE = np.asarray([511.0, 511.0, 127.0, 127.0], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--test-participant", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalize",
        choices=("none", "angle-range"),
        default="none",
        help="Use none for the closest match to the public BeamSense generator.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    return parser.parse_args()


def build_model():
    from tensorflow.keras import layers, models

    # Same layer sequence and input shape as BeamSense/CNN_station.py.
    return models.Sequential(
        [
            layers.Input(shape=(10, 234, 4)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(pool_size=(2, 1)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(pool_size=(2, 1)),
            layers.Flatten(),
            layers.Dense(20, activation="softmax"),
        ]
    )


def split_indexes(source, participant, test_participant, validation_fraction):
    """Hold out a tail from every training capture for validation."""
    test = np.flatnonzero(participant == test_participant)
    train_candidates = np.flatnonzero(participant != test_participant)
    train, validation = [], []
    for name in np.unique(source[train_candidates]):
        indexes = train_candidates[source[train_candidates] == name]
        cut = max(1, int(np.floor(len(indexes) * validation_fraction)))
        if cut >= len(indexes):
            train.extend(indexes)
        else:
            train.extend(indexes[:-cut])
            validation.extend(indexes[-cut:])
    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(validation, dtype=np.int64),
        test.astype(np.int64),
    )


def balanced_limit(indexes, labels, limit, rng):
    if limit is None or len(indexes) <= limit:
        return indexes
    chosen = []
    per_class = max(1, limit // len(CLASS_NAMES))
    for label in range(len(CLASS_NAMES)):
        candidates = indexes[labels[indexes] == label]
        if len(candidates):
            chosen.extend(rng.choice(candidates, min(per_class, len(candidates)), replace=False))
    return np.asarray(chosen, dtype=np.int64)


def class_weights(labels):
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    weights = np.divide(
        len(labels),
        len(CLASS_NAMES) * counts,
        out=np.zeros_like(counts),
        where=counts > 0,
    )
    return {index: float(weight) for index, weight in enumerate(weights)}


def main():
    args = parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise SystemExit("--validation-fraction must be between 0 and 0.5")

    random.seed(args.seed)
    np.random.seed(args.seed)
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

    tf.random.set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    data = np.load(args.npz, allow_pickle=False)
    x, y = data["x"], data["y"].astype(np.int64)
    participant, source = data["participant"], data["source"]
    if x.shape[1:] != (10, 234, 4):
        raise SystemExit(f"Expected x shape [N,10,234,4], got {x.shape}")

    train_idx, val_idx, test_idx = split_indexes(
        source, participant, args.test_participant, args.validation_fraction
    )
    rng = np.random.default_rng(args.seed)
    train_idx = balanced_limit(train_idx, y, args.max_train_samples, rng)

    class NpzSequence(tf.keras.utils.Sequence):
        def __init__(self, indexes, shuffle):
            super().__init__()
            self.indexes = np.asarray(indexes).copy()
            self.shuffle = shuffle
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.indexes) / args.batch_size))

        def __getitem__(self, batch):
            indexes = self.indexes[batch * args.batch_size : (batch + 1) * args.batch_size]
            features = x[indexes].astype(np.float32)
            if args.normalize == "angle-range":
                features /= ANGLE_SCALE
            return features, y[indexes]

        def on_epoch_end(self):
            if self.shuffle:
                rng.shuffle(self.indexes)

    train_seq = NpzSequence(train_idx, True)
    val_seq = NpzSequence(val_idx, False)
    test_seq = NpzSequence(test_idx, False)
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    checkpoint = args.output / f"p{args.test_participant}_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint, save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=6, factor=0.5, min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.05, patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger(args.output / f"p{args.test_participant}_history.csv"),
    ]
    print(
        f"fold=P{args.test_participant} train={len(train_idx)} "
        f"validation={len(val_idx)} test={len(test_idx)} normalize={args.normalize}"
    )
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights(y[train_idx]),
        verbose=1,
    )

    probabilities = model.predict(test_seq, verbose=1)
    prediction = np.argmax(probabilities, axis=1)
    truth = y[test_idx]
    metrics = {
        "test_participant": args.test_participant,
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "test_samples": len(test_idx),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, prediction, average="macro", zero_division=0)),
        "normalization": args.normalize,
        "seed": args.seed,
    }
    (args.output / f"p{args.test_participant}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    matrix = confusion_matrix(truth, prediction, labels=np.arange(20), normalize="true")
    np.savetxt(args.output / f"p{args.test_participant}_confusion.csv", matrix, delimiter=",")
    np.savez_compressed(
        args.output / f"p{args.test_participant}_predictions.npz",
        truth=truth,
        prediction=prediction,
        probabilities=probabilities,
        source=source[test_idx],
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
