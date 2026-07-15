import numpy as np

from tfdiff.bfa_har import decode_quantized_bfa, packet_windows, parse_har1_name


def _pack_angles(rows):
    packed = []
    for row in rows:
        bits = []
        for value, width in zip(row, (9, 9, 7, 7)):
            bits.extend((value >> bit) & 1 for bit in range(width))
        packed.extend(np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="little"))
    return bytes([42]) + bytes(packed) + bytes([99, 100])


def test_decode_quantized_bfa_skips_snr_and_exclusive_report():
    expected = np.asarray([[0, 511, 1, 127], [123, 321, 45, 67]], dtype=np.uint16)
    actual = decode_quantized_bfa(_pack_angles(expected), subcarriers=2)
    np.testing.assert_array_equal(actual, expected)


def test_packet_windows_drops_incomplete_tail():
    packets = np.arange(12 * 2).reshape(12, 2)
    windows, starts = packet_windows(packets, window_size=5, stride=5)
    assert windows.shape == (2, 5, 2)
    np.testing.assert_array_equal(starts, [0, 5])


def test_parse_har1_name():
    metadata = parse_har1_name("G_1_M1_P3.pcapng")
    assert (metadata.label, metadata.repetition, metadata.monitor, metadata.participant) == (
        6,
        1,
        1,
        3,
    )
