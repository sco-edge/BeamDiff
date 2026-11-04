import numpy as np
import pytest

from tfdiff.bfa_har import packet_windows


@pytest.mark.parametrize('window,stride', [(0, 1), (1, 0), (-1, 2), (2, -1)])
def test_nonpositive_window_configuration(window, stride):
    with pytest.raises(ValueError, match='positive'):
        packet_windows(np.zeros((10, 2)), window_size=window, stride=stride)


@pytest.mark.parametrize('count', [0, 3, 9])
def test_short_capture_has_no_padded_windows(count):
    windows, starts = packet_windows(np.zeros((count, 234, 4), dtype=np.uint16))
    assert windows.shape == (0, 10, 234, 4)
    assert windows.dtype == np.uint16
    assert starts.size == 0


def test_overlapping_windows_preserve_exact_packet_order():
    packets = np.arange(14, dtype=np.uint16)[:, None]
    windows, starts = packet_windows(packets, window_size=10, stride=2)
    np.testing.assert_array_equal(starts, [0, 2, 4])
    for window, start in zip(windows, starts):
        np.testing.assert_array_equal(window, packets[start:start + 10])
