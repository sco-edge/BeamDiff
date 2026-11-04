import numpy as np
import pytest

from scripts.train_beamsense_har1 import load_fixed_split, random_window_split


def save_split(root, train=(0, 1), validation=(2,), test=(3,)):
    for name, values in zip(('train', 'validation', 'test'), (train, validation, test)):
        np.save(root / f'{name}_indices.npy', values)


def test_fixed_split_covers_dataset_without_overlap(tmp_path):
    save_split(tmp_path)
    groups = load_fixed_split(tmp_path, 4)
    np.testing.assert_array_equal(np.sort(np.concatenate(groups)), np.arange(4))


@pytest.mark.parametrize('train,validation,test', [
    ((0, 0), (2,), (3,)), ((0, 1), (1,), (3,)),
    ((-1, 1), (2,), (3,)), ((0, 4), (2,), (3,)), ((), (2,), (3,)),
])
def test_invalid_fixed_splits_fail(tmp_path, train, validation, test):
    save_split(tmp_path, train, validation, test)
    with pytest.raises(SystemExit):
        load_fixed_split(tmp_path, 4)


def test_missing_split_is_reported(tmp_path):
    with pytest.raises(SystemExit, match='Missing fixed split'):
        load_fixed_split(tmp_path, 4)


def test_random_window_split_is_repeatable_and_exhaustive():
    first = random_window_split(1000, np.random.default_rng(111))
    second = random_window_split(1000, np.random.default_rng(111))
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(np.sort(np.concatenate(first)), np.arange(1000))
