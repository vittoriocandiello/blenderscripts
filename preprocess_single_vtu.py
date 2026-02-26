#!/usr/bin/env python3
"""
preprocess_single_vtu.py — Preprocess one VTU file for Blender rendering.

Writes the same frame/metadata format as preprocess.py, but for a single stream
and a single frame.

Usage:
    python preprocess_single_vtu.py \
        --vtu sphere_mesh_.vtu \
        --out preprocessed_single

    # Optional: write the same mesh to fish1 and fish2
    python preprocess_single_vtu.py \
        --vtu sphere_mesh_.vtu \
        --out preprocessed_single \
        --stream fish1 \
        --duplicate-to fish2
"""

import argparse
import os

from preprocess import process_series


def _validate_stream_name(name):
    if not name:
        raise ValueError("Stream name cannot be empty")
    if any(ch in name for ch in ("/", "\\")):
        raise ValueError(f"Invalid stream name '{name}': use a folder name, not a path")


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess a single .vtu file into Blender-ready NPZ frames."
    )
    ap.add_argument("--vtu", required=True, help="Path to input .vtu file")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--stream", default="fish1", help="Output stream folder name (default: fish1)")
    ap.add_argument(
        "--duplicate-to",
        default=None,
        help="Optional second stream name to write from the same VTU (e.g. fish2)",
    )
    ap.add_argument(
        "--timestep",
        type=float,
        default=0.0,
        help="Timestep value stored in the output frame (default: 0.0)",
    )
    args = ap.parse_args()

    vtu_path = os.path.abspath(args.vtu)
    if not os.path.isfile(vtu_path):
        raise FileNotFoundError(f"VTU not found: {args.vtu}")

    _validate_stream_name(args.stream)
    if args.duplicate_to is not None:
        _validate_stream_name(args.duplicate_to)
        if args.duplicate_to == args.stream:
            raise ValueError("--duplicate-to must be different from --stream")

    os.makedirs(args.out, exist_ok=True)
    series = [(float(args.timestep), vtu_path)]

    print(f"Input:  {vtu_path}")
    print(f"Output: {os.path.abspath(args.out)}")
    print(f"\n--- Stream {args.stream} ---")
    process_series(series, args.out, args.stream)

    if args.duplicate_to is not None:
        print(f"\n--- Stream {args.duplicate_to} (duplicate) ---")
        process_series(series, args.out, args.duplicate_to)

    print(f"\nDone -> {os.path.abspath(args.out)}")
    if args.stream == "fish1" and args.duplicate_to == "fish2":
        print(
            "Next: blender --background --python render_blender.py -- "
            f"--input {args.out} --output renders/"
        )
    elif args.stream == "fish1":
        print(
            "Next: blender --background --python render_blender.py -- "
            f"--input {args.out} --output renders/ --allow-missing-stream"
        )


if __name__ == "__main__":
    main()
