import struct

import pytest

from tfdiff.bfi_pcapng import iter_pcapng_packets, _parse_radiotap, _parse_80211_action


@pytest.mark.parametrize('length', [0, 8, 100])
def test_invalid_pcapng_block_length(tmp_path, length):
    path = tmp_path / 'broken.pcapng'
    path.write_bytes(struct.pack('<III', 6, length, length))
    with pytest.raises(ValueError, match='Invalid pcapng block'):
        list(iter_pcapng_packets(path))


@pytest.mark.parametrize('packet', [b'', bytes(7), struct.pack('<BBH', 1, 0, 8) + bytes(4),
                                     struct.pack('<BBH', 0, 0, 99) + bytes(4)])
def test_invalid_radiotap_is_rejected(packet):
    assert _parse_radiotap(packet) is None
    assert _parse_80211_action(packet) is None


def test_empty_capture_has_no_packets(tmp_path):
    path = tmp_path / 'empty.pcapng'
    path.write_bytes(b'')
    assert list(iter_pcapng_packets(path)) == []
