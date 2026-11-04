import numpy as np
import pytest

from scripts.train_beamsense_har1 import layered_augmentation_subset, nested_augmentation_subset


def test_first_layer_uses_each_anchor_before_reusing_one():
    groups = [[0, 3], [1, 4], [2, 5]]
    first, pool = layered_augmentation_subset(groups, 1.0, 111, 3)
    full, _ = layered_augmentation_subset(groups, 2.0, 111, 3)
    np.testing.assert_array_equal(np.sort(first), [0, 1, 2])
    np.testing.assert_array_equal(first, full[:3])
    np.testing.assert_array_equal(full, pool)


def test_request_larger_than_pool_is_rejected():
    with pytest.raises(SystemExit, match='only 2 are available'):
        layered_augmentation_subset([[0], [1]], 2, 42, 2)


@pytest.mark.parametrize('ratio', [-1, float('nan')])
def test_invalid_nested_ratio(ratio):
    with pytest.raises(ValueError):
        nested_augmentation_subset(np.arange(3), ratio, 42)


def test_empty_layered_pool_at_zero_ratio():
    selected, pool = layered_augmentation_subset([], 0, 42, 0)
    assert selected.size == pool.size == 0
