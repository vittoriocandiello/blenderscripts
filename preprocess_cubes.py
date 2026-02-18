#!/usr/bin/env python3
"""
preprocess_cubes.py — Cube-oriented wrapper around preprocess.py.

Writes the same preprocessed format but in:
  - cube1/frame_*.npz
  - cube2/frame_*.npz

Usage:
    python preprocess_cubes.py \
        --pvd1 cube_sim_a/scene.pvd \
        --pvd2 cube_sim_b/scene.pvd \
        --out preprocessed/
"""

import os
import argparse
from preprocess import parse_pvd, series_from_vtu_dir, process_series


def main():
    ap = argparse.ArgumentParser()
    g1 = ap.add_mutually_exclusive_group(required=True)
    g1.add_argument("--pvd1")
    g1.add_argument("--vtu-dir1")
    g2 = ap.add_mutually_exclusive_group(required=True)
    g2.add_argument("--pvd2")
    g2.add_argument("--vtu-dir2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s1 = parse_pvd(args.pvd1) if args.pvd1 else series_from_vtu_dir(args.vtu_dir1)
    s2 = parse_pvd(args.pvd2) if args.pvd2 else series_from_vtu_dir(args.vtu_dir2)

    print(f"Cube 1: {len(s1)} frames\nCube 2: {len(s2)} frames")
    if len(s1) != len(s2):
        n = min(len(s1), len(s2))
        s1, s2 = s1[:n], s2[:n]
        print(f"Truncating to {n}")

    os.makedirs(args.out, exist_ok=True)
    print("\n--- Cube 1 ---")
    process_series(s1, args.out, "cube1")
    print("\n--- Cube 2 ---")
    process_series(s2, args.out, "cube2")
    print(f"\nDone -> {os.path.abspath(args.out)}")
    print(
        "Next: blender --background --python render_blender_cubes.py -- "
        f"--input {args.out} --output renders/"
    )


if __name__ == "__main__":
    main()
