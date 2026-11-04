import numpy as np
import pytest

from tfdiff.bfa_har import decode_quantized_bfa, parse_har1_name, parse_vht_mimo_control


@pytest.mark.parametrize('name', ['U_1_M1_P1.pcapng', 'A_1_M1.pcapng', 'A_1_M1_P1.mat'])
def test_invalid_capture_name(name):
    with pytest.raises(ValueError, match='Unexpected HAR-1 filename'):
        parse_har1_name(name)


def test_case_insensitive_capture_metadata():
    metadata = parse_har1_name('t_2_m1_p3.PCAPNG')
    assert (metadata.activity, metadata.label, metadata.participant) == ('T', 19, 3)


@pytest.mark.parametrize('nc,nr', [(2, 3), (1, 2)])
def test_unsupported_antenna_layout(nc, nr):
    with pytest.raises(ValueError, match='Nc=1, Nr=3'):
        decode_quantized_bfa(bytes(32), nc=nc, nr=nr, subcarriers=1)


def test_truncated_angle_report():
    with pytest.raises(ValueError, match='too short'):
        decode_quantized_bfa(bytes(4), subcarriers=1)


def test_zero_and_maximum_angles():
    report = bytes([99]) + bytes(4) + bytes([255] * 4)
    result = decode_quantized_bfa(report, subcarriers=2)
    np.testing.assert_array_equal(result, [[0, 0, 0, 0], [511, 511, 127, 127]])
    assert result.dtype == np.uint16


def test_vht_mimo_control_layout():
    assert parse_vht_mimo_control((2 << 6) | (2 << 3)) == (1, 3, 2)
