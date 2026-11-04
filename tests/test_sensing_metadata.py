import numpy as np
import pytest

from scripts.train_beamsense_har1 import canonical_source, class_weights, sample_keys


@pytest.mark.parametrize('suffix', ['.pcapng', '.pcap', '.mat', '.npy'])
def test_capture_identifiers_are_canonical_across_formats(suffix):
    assert canonical_source('/data/A_1_M1_P1' + suffix) == 'A_1_M1_P1'


def test_sample_key_preserves_start_and_activity():
    data = {'source': ['A_1_M1_P1.pcapng', 'A_1_M1_P1.npy'], 'window_start': [0, 10]}
    assert sample_keys(data, [0, 0]) == [('A_1_M1_P1', 0, 0), ('A_1_M1_P1', 10, 0)]


def test_missing_classes_have_finite_zero_weights():
    weights = class_weights(np.asarray([0, 0, 1]))
    assert len(weights) == 20
    assert weights[1] == 2 * weights[0]
    assert weights[19] == 0
    assert all(np.isfinite(value) for value in weights.values())
