"""Convert HAR-1 BFI PCAPNG files to BeamSense-style BFA tensors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfdiff.bfa_har import ANGLE_NAMES, packet_windows, parse_har1_name, pcap_to_bfa_packets


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="HAR-1/BFI/M1 directory or one PCAPNG")
    parser.add_argument("output", type=Path, help="Output compressed NPZ")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args(argv)

    files = sorted(args.input.glob("*.pcapng")) if args.input.is_dir() else [args.input]
    if not files:
        raise SystemExit(f"No PCAPNG files found in {args.input}")

    arrays = {key: [] for key in (
        "x", "y", "participant", "monitor", "repetition", "source", "window_start"
    )}
    failures = []
    for path in files:
        try:
            metadata = parse_har1_name(path)
            packets = pcap_to_bfa_packets(path)
            windows, starts = packet_windows(
                packets, window_size=args.window_size, stride=args.stride
            )
            if not len(windows):
                raise ValueError(f"only {len(packets)} packets; no complete window")
        except Exception as error:  # continue to report every bad capture
            failures.append((path.name, str(error)))
            print(f"FAIL {path.name}: {error}")
            continue

        count = len(windows)
        arrays["x"].append(windows)
        arrays["y"].append(np.full(count, metadata.label, dtype=np.int16))
        arrays["participant"].append(np.full(count, metadata.participant, dtype=np.int16))
        arrays["monitor"].append(np.full(count, metadata.monitor, dtype=np.int16))
        arrays["repetition"].append(np.full(count, metadata.repetition, dtype=np.int16))
        arrays["source"].append(np.full(count, path.name))
        arrays["window_start"].append(starts)
        print(f"OK   {path.name}: packets={len(packets)} windows={count}")

    if not arrays["x"]:
        raise SystemExit("No BFA samples were produced")

    output = {key: np.concatenate(value) for key, value in arrays.items()}
    output["activity_names"] = np.asarray(list("ABCDEFGHIJKLMNOPQRST"))
    output["angle_names"] = np.asarray(ANGLE_NAMES)
    output["window_size"] = np.asarray(args.window_size, dtype=np.int16)
    output["stride"] = np.asarray(args.stride, dtype=np.int16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)

    print(f"Saved: {args.output}")
    print(f"x={output['x'].shape} dtype={output['x'].dtype}, y={output['y'].shape}")
    print(f"failed_files={len(failures)}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
