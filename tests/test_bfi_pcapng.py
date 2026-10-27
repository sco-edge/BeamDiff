import struct

import numpy as np

from tfdiff.bfi_pcapng import extract_vht_bfi_records, records_to_feature


def _block(block_type, body):
    length = 12 + len(body)
    return struct.pack("<II", block_type, length) + body + struct.pack("<I", length)


def _minimal_pcapng(packet):
    section_body = struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1)
    interface_body = struct.pack("<HHI", 127, 0, 65535)
    epb_body = (
        struct.pack("<IIIII", 0, 0, 0, len(packet), len(packet))
        + packet
        + (b"\x00" * ((4 - len(packet) % 4) % 4))
    )
    return b"".join(
        (
            _block(0x0A0D0D0A, section_body),
            _block(0x00000001, interface_body),
            _block(0x00000006, epb_body),
        )
    )


def test_extract_vht_bfi_records_from_pcapng(tmp_path):
    radiotap = b"\x00\x00\x08\x00\x00\x00\x00\x00"
    dot11_header = (
        b"\xe0\x00\x00\x00"
        + b"\x01\x02\x03\x04\x05\x06"
        + b"\x07\x08\x09\x0a\x0b\x0c"
        + b"\x0d\x0e\x0f\x10\x11\x12"
        + b"\x00\x00"
    )
    payload = b"\x15\x00\x01\x02\x03" + bytes(range(12)) + b"\xaa\xbb\xcc\xdd"
    pcapng = tmp_path / "sample.pcapng"
    pcapng.write_bytes(_minimal_pcapng(radiotap + dot11_header + payload))

    records = extract_vht_bfi_records(pcapng)
    feature = records_to_feature(records)

    assert len(records) == 1
    assert records[0]["mimo_control"] == 0x030201
    assert feature.shape == (1, 12)
    assert np.array_equal(feature[0], np.arange(12, dtype=np.float32))
