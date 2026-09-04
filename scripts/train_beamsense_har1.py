"""Train the BeamSense 2-D CNN on a HAR-1 BFA NPZ dataset."""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument(
        "--split", choices=("participant", "random-window"), default="participant"
    )
    parser.add_argument("--test-participant", type=int, choices=(1, 2, 3))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        help="data split seed; defaults to --seed for backward compatibility",
    )
    parser.add_argument(
        "--split-indices-dir",
        type=Path,
        help=(
            "directory containing train_indices.npy, validation_indices.npy, and "
            "test_indices.npy; overrides generated random-window indices"
        ),
    )
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="balanced")
    parser.add_argument(
        "--normalize",
        choices=("none", "angle-range"),
        default="none",
        help="Use none for the closest match to the public BeamSense generator.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--augment-npz", type=Path)
    parser.add_argument("--augment-ratio", type=float, default=1.0)
    parser.add_argument(
        "--allow-partial-augmentation",
        action="store_true",
        help="Allow an augmentation file containing only a subset of real train keys.",
    )
    parser.add_argument(
        "--augment-seed",
        type=int,
        help="synthetic subset seed; defaults to --seed for backward compatibility",
    )
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


def random_window_split(size, rng):
    """Reproduce BeamSense create_csv_CNN.py's 70/15/15 window split."""
    values = rng.random(size)
    return (
        np.flatnonzero(values < 0.70),
        np.flatnonzero((values >= 0.70) & (values < 0.85)),
        np.flatnonzero(values >= 0.85),
    )


def load_fixed_split(directory, size):
    names = ("train", "validation", "test")
    indexes = []
    for name in names:
        path = directory / f"{name}_indices.npy"
        if not path.is_file():
            raise SystemExit(f"Missing fixed split file: {path}")
        values = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64).reshape(-1)
        if len(values) == 0:
            raise SystemExit(f"Fixed split is empty: {path}")
        if values.min() < 0 or values.max() >= size:
            raise SystemExit(f"Fixed split contains out-of-range indices: {path}")
        if len(np.unique(values)) != len(values):
            raise SystemExit(f"Fixed split contains duplicate indices: {path}")
        indexes.append(values)
    combined = np.concatenate(indexes)
    if len(np.unique(combined)) != len(combined):
        raise SystemExit("Fixed train/validation/test splits overlap")
    if len(combined) != size:
        raise SystemExit(
            f"Fixed splits cover {len(combined)} of {size} samples; expected full coverage"
        )
    return tuple(indexes)


def nested_augmentation_subset(indexes, ratio, seed):
    """A fixed seeded permutation prefix, so smaller ratios nest in larger."""
    indexes = np.asarray(indexes, dtype=np.int64)
    if not 0 <= ratio <= 1:
        raise ValueError("ratio must be in [0,1]")
    order = np.random.default_rng(seed).permutation(len(indexes))
    return indexes[order[: int(np.floor(len(indexes) * ratio))]]


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


def canonical_source(value):
    name = Path(str(value)).name
    for suffix in (".pcapng", ".pcap", ".mat", ".npy"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def sample_keys(data, labels):
    return [
        (canonical_source(source), int(start), int(label))
        for source, start, label in zip(data["source"], data["window_start"], labels)
    ]


def main():
    args = parse_args()
    if args.split == "participant" and args.test_participant is None:
        raise SystemExit("--test-participant is required for --split participant")
    if not 0 < args.validation_fraction < 0.5:
        raise SystemExit("--validation-fraction must be between 0 and 0.5")
    if not 0 <= args.augment_ratio <= 1:
        raise SystemExit("--augment-ratio must be between 0 and 1")

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

    split_seed = args.seed if args.split_seed is None else args.split_seed
    augment_seed = args.seed if args.augment_seed is None else args.augment_seed
    split_rng = np.random.default_rng(split_seed)
    rng = np.random.default_rng(args.seed)
    if args.split_indices_dir is not None:
        if args.split != "random-window":
            raise SystemExit("--split-indices-dir requires --split random-window")
        train_idx, val_idx, test_idx = load_fixed_split(args.split_indices_dir, len(y))
        fold_name = "random_window"
    elif args.split == "random-window":
        train_idx, val_idx, test_idx = random_window_split(len(y), split_rng)
        fold_name = "random_window"
    else:
        train_idx, val_idx, test_idx = split_indexes(
            source, participant, args.test_participant, args.validation_fraction
        )
        fold_name = f"p{args.test_participant}"
    train_idx = balanced_limit(train_idx, y, args.max_train_samples, rng)

    augment_x = augment_y = None
    augment_idx = np.empty(0, dtype=np.int64)
    if args.augment_npz is not None:
        augmented = np.load(args.augment_npz, allow_pickle=False)
        augment_x = augmented["x"]
        augment_y = augmented["y"].astype(np.int64)
        if augment_x.shape[1:] != (10, 234, 4):
            raise SystemExit(
                f"Expected augmented x shape [N,10,234,4], got {augment_x.shape}"
            )
        if "augmentation_eligible" in augmented.files:
            if (args.split != "random-window"
                    or int(augmented["train_split_seed"]) != split_seed):
                raise SystemExit("Direct BFA augmentation requires its original random-window split seed")
        generated_by_key = {
            key: index for index, key in enumerate(sample_keys(augmented, augment_y))
        }
        real_keys = sample_keys(data, y)
        matched = [generated_by_key.get(real_keys[index]) for index in train_idx]
        missing = sum(index is None for index in matched)
        if missing and not args.allow_partial_augmentation:
            raise SystemExit(f"Generated data is missing {missing} real training samples")
        augment_idx = np.asarray([index for index in matched if index is not None], dtype=np.int64)
        if not len(augment_idx):
            raise SystemExit("Generated data has no keys matching the real training fold")
        if "augmentation_eligible" in augmented.files:
            if not np.all(augmented["augmentation_eligible"][augment_idx]):
                raise SystemExit("Direct BFA augmentation contains ineligible training matches")
        augment_idx = nested_augmentation_subset(
            augment_idx, args.augment_ratio, augment_seed
        )

    class NpzSequence(tf.keras.utils.Sequence):
        def __init__(self, indexes, shuffle, generated_indexes=None):
            super().__init__()
            indexes = np.asarray(indexes, dtype=np.int64)
            generated_indexes = np.asarray(
                [] if generated_indexes is None else generated_indexes, dtype=np.int64
            )
            self.real_count = len(indexes)
            self.indexes = np.concatenate([indexes, generated_indexes])
            self.generated = np.concatenate(
                [np.zeros(len(indexes), dtype=bool), np.ones(len(generated_indexes), dtype=bool)]
            )
            self.shuffle = shuffle
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.indexes) / args.batch_size))

        def __getitem__(self, batch):
            selection = slice(batch * args.batch_size, (batch + 1) * args.batch_size)
            indexes = self.indexes[selection]
            generated = self.generated[selection]
            features = np.empty((len(indexes), 10, 234, 4), dtype=np.float32)
            labels = np.empty(len(indexes), dtype=np.int64)
            if np.any(~generated):
                features[~generated] = x[indexes[~generated]].astype(np.float32)
                labels[~generated] = y[indexes[~generated]]
            if np.any(generated):
                features[generated] = augment_x[indexes[generated]].astype(np.float32)
                labels[generated] = augment_y[indexes[generated]]
            if args.normalize == "angle-range":
                features /= ANGLE_SCALE
            return features, labels

        def on_epoch_end(self):
            if self.shuffle:
                order = rng.permutation(len(self.indexes))
                self.indexes = self.indexes[order]
                self.generated = self.generated[order]

    train_seq = NpzSequence(train_idx, True, augment_idx)
    val_seq = NpzSequence(val_idx, False)
    test_seq = NpzSequence(test_idx, False)
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    checkpoint = args.output / f"{fold_name}_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=6, factor=0.5, min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.05, patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger(str(args.output / f"{fold_name}_history.csv")),
    ]
    print(
        f"split={args.split} fold={fold_name} real_train={len(train_idx)} "
        f"synthetic_train={len(augment_idx)} train={len(train_idx) + len(augment_idx)} "
        f"validation={len(val_idx)} test={len(test_idx)} normalize={args.normalize} "
        f"model_seed={args.seed} split_seed={split_seed} augment_seed={augment_seed}"
    )
    fit_labels = np.concatenate(
        [y[train_idx], augment_y[augment_idx] if augment_y is not None else np.empty(0, int)]
    )
    fit_class_weight = class_weights(fit_labels) if args.class_weight == "balanced" else None
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=fit_class_weight,
        verbose=1,
    )

    probabilities = model.predict(test_seq, verbose=1)
    prediction = np.argmax(probabilities, axis=1)
    truth = y[test_idx]
    metrics = {
        "split": args.split,
        "test_participant": args.test_participant,
        "real_data": str(args.npz),
        "augmentation_data": str(args.augment_npz) if args.augment_npz else None,
        "augmentation_ratio": args.augment_ratio if args.augment_npz else 0.0,
        "real_train_samples": len(train_idx),
        "synthetic_train_samples": len(augment_idx),
        "train_samples": len(train_idx) + len(augment_idx),
        "validation_samples": len(val_idx),
        "test_samples": len(test_idx),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, prediction, average="macro", zero_division=0)),
        "normalization": args.normalize,
        "class_weight": args.class_weight,
        "seed": args.seed,
        "split_seed": split_seed,
        "split_indices_dir": (
            str(args.split_indices_dir) if args.split_indices_dir else None
        ),
        "augmentation_seed": augment_seed if args.augment_npz else None,
        "augmentation_selection": (
            "seeded-permutation-prefix-v1" if args.augment_npz else None
        ),
        "partial_augmentation_allowed": (
            args.allow_partial_augmentation if args.augment_npz else None
        ),
        "eligible_synthetic_samples": (
            len(matched) - missing if args.augment_npz else 0
        ),
        "augmentation_indices_sha256": (
            hashlib.sha256(augment_idx.astype("<i8").tobytes()).hexdigest()
            if args.augment_npz else None
        ),
    }
    (args.output / f"{fold_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    matrix = confusion_matrix(truth, prediction, labels=np.arange(20), normalize="true")
    np.savetxt(args.output / f"{fold_name}_confusion.csv", matrix, delimiter=",")
    np.savez_compressed(
        args.output / f"{fold_name}_predictions.npz",
        truth=truth,
        prediction=prediction,
        probabilities=probabilities,
        source=source[test_idx],
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
