"""BeamSense-style BFA extraction for the CSI-BFI HAR captures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bfi_pcapng import extract_vht_bfi_records


ANGLE_BITS = (9, 9, 7, 7)
ANGLE_NAMES = ("phi_11", "phi_21", "psi_21", "psi_31")
HAR1_NAME = re.compile(
    r"^(?P<activity>[A-T])_(?P<repetition>\d+)_M(?P<monitor>\d+)_P(?P<participant>\d+)\.pcapng$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Har1Metadata:
    activity: str
    label: int
    repetition: int
    monitor: int
    participant: int


def parse_vht_mimo_control(value: int) -> tuple[int, int, int]:
    """Return (Nc, Nr, bandwidth_code) from a VHT MIMO control field."""
    return (value & 0x7) + 1, ((value >> 3) & 0x7) + 1, (value >> 6) & 0x3


def parse_har1_name(path: str | Path) -> Har1Metadata:
    match = HAR1_NAME.match(Path(path).name)
    if not match:
        raise ValueError(f"Unexpected HAR-1 filename: {Path(path).name}")
    activity = match.group("activity").upper()
    return Har1Metadata(
        activity=activity,
        label=ord(activity) - ord("A"),
        repetition=int(match.group("repetition")),
        monitor=int(match.group("monitor")),
        participant=int(match.group("participant")),
    )


def decode_quantized_bfa(
    report: bytes,
    *,
    nc: int = 1,
    nr: int = 3,
    subcarriers: int = 234,
) -> np.ndarray:
    """Decode the four quantized BeamSense angles into [subcarrier, angle].

    This follows pcap_to_bfa.m for the HAR-1 80 MHz, Nc=1, Nr=3 setup.
    The compressed report begins with one SNR byte per spatial stream.
    """
    if (nc, nr) != (1, 3):
        raise ValueError(f"BeamSense HAR-1 expects Nc=1, Nr=3; got Nc={nc}, Nr={nr}")

    bits_per_subcarrier = sum(ANGLE_BITS)
    bytes_per_subcarrier = (bits_per_subcarrier + 7) // 8
    angle_bytes = subcarriers * bytes_per_subcarrier
    start = nc
    end = start + angle_bytes
    if len(report) < end:
        raise ValueError(
            f"BFI report is too short: need at least {end} bytes, got {len(report)}"
        )

    raw = np.frombuffer(report[start:end], dtype=np.uint8).reshape(subcarriers, -1)
    bits = np.unpackbits(raw, axis=1, bitorder="little")
    result = np.empty((subcarriers, len(ANGLE_BITS)), dtype=np.uint16)
    offset = 0
    for index, width in enumerate(ANGLE_BITS):
        weights = (1 << np.arange(width, dtype=np.uint16)).astype(np.uint16)
        result[:, index] = bits[:, offset : offset + width] @ weights
        offset += width
    return result


def pcap_to_bfa_packets(path: str | Path, *, subcarriers: int = 234) -> np.ndarray:
    """Return all valid BFA packets as [packet, subcarrier, 4]."""
    records = extract_vht_bfi_records(path)
    packets = []
    for record in records:
        nc, nr, bandwidth = parse_vht_mimo_control(record["mimo_control"])
        if bandwidth != 2:
            raise ValueError(f"Expected 80 MHz bandwidth code 2, got {bandwidth}")
        packets.append(
            decode_quantized_bfa(
                record["bfi_report"], nc=nc, nr=nr, subcarriers=subcarriers
            )
        )
    if not packets:
        raise ValueError(f"No decodable BFA packets in {path}")
    return np.stack(packets)


def packet_windows(
    packets: np.ndarray, *, window_size: int = 10, stride: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Create complete packet-count windows without zero-only tail samples."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    starts = np.arange(0, len(packets) - window_size + 1, stride, dtype=np.int32)
    if not len(starts):
        return np.empty((0, window_size, *packets.shape[1:]), dtype=packets.dtype), starts
    return np.stack([packets[start : start + window_size] for start in starts]), starts
