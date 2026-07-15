"""Extract BFI-like VHT action payloads from 802.11 pcapng captures."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


PCAPNG_SECTION_HEADER = 0x0A0D0D0A
PCAPNG_ENHANCED_PACKET = 0x00000006
VHT_ACTION_CATEGORY = 0x15
VHT_COMPRESSED_BEAMFORMING_ACTION = 0x00


def _align32(length):
    return (length + 3) & ~3


def iter_pcapng_packets(filename):
    """Yield captured packet bytes from a pcapng file."""
    data = Path(filename).read_bytes()
    offset = 0
    endian = "<"

    while offset + 12 <= len(data):
        block_type_le = struct.unpack_from("<I", data, offset)[0]
        if block_type_le == PCAPNG_SECTION_HEADER:
            bom = data[offset + 8 : offset + 12]
            if bom == b"\x4d\x3c\x2b\x1a":
                endian = "<"
            elif bom == b"\x1a\x2b\x3c\x4d":
                endian = ">"

        block_type, block_len = struct.unpack_from(f"{endian}II", data, offset)
        if block_len < 12 or offset + block_len > len(data):
            raise ValueError(f"Invalid pcapng block at offset {offset}.")

        if block_type == PCAPNG_ENHANCED_PACKET:
            if block_len < 32:
                raise ValueError(f"Invalid enhanced packet block at offset {offset}.")
            captured_len = struct.unpack_from(f"{endian}I", data, offset + 20)[0]
            packet_start = offset + 28
            packet_end = packet_start + captured_len
            if packet_end <= offset + block_len - 4:
                yield data[packet_start:packet_end]

        offset += block_len


def _parse_radiotap(packet):
    if len(packet) < 8:
        return None
    version, _, radiotap_len = struct.unpack_from("<BBH", packet, 0)
    if version != 0 or radiotap_len > len(packet):
        return None
    return radiotap_len


def _parse_80211_action(packet):
    radiotap_len = _parse_radiotap(packet)
    if radiotap_len is None or len(packet) < radiotap_len + 26:
        return None

    frame_control = struct.unpack_from("<H", packet, radiotap_len)[0]
    frame_type = (frame_control >> 2) & 0x3
    subtype = (frame_control >> 4) & 0xF
    if frame_type != 0 or subtype not in (13, 14):
        return None

    header_len = 24
    payload_offset = radiotap_len + header_len
    payload = packet[payload_offset:]
    if len(payload) < 2:
        return None

    return {
        "radiotap_len": radiotap_len,
        "subtype": subtype,
        "addr1": packet[radiotap_len + 4 : radiotap_len + 10],
        "addr2": packet[radiotap_len + 10 : radiotap_len + 16],
        "addr3": packet[radiotap_len + 16 : radiotap_len + 22],
        "payload": payload,
    }


def _format_mac(raw):
    return ":".join(f"{byte:02x}" for byte in raw)


def extract_vht_bfi_records(filename, *, strip_fcs=True):
    """Extract VHT compressed beamforming action records from a pcapng file."""
    records = []
    for packet_index, packet in enumerate(iter_pcapng_packets(filename)):
        action = _parse_80211_action(packet)
        if not action:
            continue

        payload = action["payload"]
        category = payload[0]
        action_code = payload[1]
        if category != VHT_ACTION_CATEGORY or action_code != VHT_COMPRESSED_BEAMFORMING_ACTION:
            continue
        if len(payload) < 5:
            continue

        report = payload[5:]
        if strip_fcs and len(report) > 4:
            report = report[:-4]

        mimo_control = int.from_bytes(payload[2:5], byteorder="little")
        records.append(
            {
                "packet_index": packet_index,
                "subtype": action["subtype"],
                "addr1": _format_mac(action["addr1"]),
                "addr2": _format_mac(action["addr2"]),
                "addr3": _format_mac(action["addr3"]),
                "mimo_control": mimo_control,
                "payload": payload,
                "bfi_report": report,
            }
        )

    return records


def records_to_feature(records, *, mode="bytes", pad_value=0):
    """Convert extracted BFI records to a rectangular time-major feature array."""
    if mode not in ("bytes", "bits"):
        raise ValueError("mode must be 'bytes' or 'bits'.")
    if not records:
        raise ValueError("No VHT compressed beamforming BFI records found.")

    rows = []
    max_len = max(len(record["bfi_report"]) for record in records)
    for record in records:
        row = np.frombuffer(record["bfi_report"], dtype=np.uint8).astype(np.float32)
        if len(row) < max_len:
            row = np.pad(row, (0, max_len - len(row)), constant_values=pad_value)
        rows.append(row)

    feature = np.stack(rows, axis=0)
    if mode == "bits":
        feature = np.unpackbits(feature.astype(np.uint8), axis=1).astype(np.float32)
    return feature


def save_records_npz(records, output_file, *, label=0, mode="bytes"):
    """Save extracted BFI features in Widar-style NPZ keys."""
    feature = records_to_feature(records, mode=mode)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        feature=feature,
        cond=np.array([label]),
        mimo_control=np.array([record["mimo_control"] for record in records], dtype=np.uint32),
        packet_index=np.array([record["packet_index"] for record in records], dtype=np.int64),
    )
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract VHT compressed beamforming BFI records from pcapng."
    )
    parser.add_argument("input", help="Input .pcapng file.")
    parser.add_argument("output", help="Output .npz file.")
    parser.add_argument("--label", default=0, type=int, help="Condition/class label.")
    parser.add_argument("--mode", default="bytes", choices=("bytes", "bits"))
    parser.add_argument("--keep-fcs", action="store_true", help="Keep final 4 FCS bytes.")
    args = parser.parse_args(argv)

    records = extract_vht_bfi_records(args.input, strip_fcs=not args.keep_fcs)
    output = save_records_npz(records, args.output, label=args.label, mode=args.mode)
    feature = records_to_feature(records, mode=args.mode)
    print(f"Extracted {len(records)} BFI records -> {output}")
    print(f"feature shape: {feature.shape}")


if __name__ == "__main__":
    main()
