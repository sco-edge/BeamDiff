import numpy as np
import pytest

from tfdiff.bfi_pcapng import records_to_feature, save_records_npz


def test_variable_report_lengths_are_padded():
    records = [{'bfi_report': bytes([1, 2])}, {'bfi_report': bytes([3])}]
    actual = records_to_feature(records, pad_value=7)
    np.testing.assert_array_equal(actual, [[1, 2], [3, 7]])
    assert actual.dtype == np.float32


def test_bit_features_have_expected_order():
    actual = records_to_feature([{'bfi_report': bytes([128, 1])}], mode='bits')
    np.testing.assert_array_equal(actual, [[1, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 1]])


@pytest.mark.parametrize('mode', ['invalid', ''])
def test_invalid_feature_mode(mode):
    with pytest.raises(ValueError, match='mode'):
        records_to_feature([{'bfi_report': b'1'}], mode=mode)


def test_empty_records_fail_clearly():
    with pytest.raises(ValueError, match='No VHT'):
        records_to_feature([])


def test_saved_features_keep_packet_metadata(tmp_path):
    records = [{'bfi_report': b'\x01\x02', 'mimo_control': 144, 'packet_index': 3}]
    path = save_records_npz(records, tmp_path / 'nested' / 'features.npz', label=4)
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['feature'], [[1, 2]])
        np.testing.assert_array_equal(data['cond'], [4])
        np.testing.assert_array_equal(data['packet_index'], [3])
