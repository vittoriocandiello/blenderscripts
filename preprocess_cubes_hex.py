#!/usr/bin/env python3
"""
preprocess_cubes_hex.py — Extract surface meshes from two hexahedral cube simulations.

Output layout:
  - <out>/cube1/frame_*.npz
  - <out>/cube2/frame_*.npz

Usage:
    python preprocess_cubes_hex.py \
        --pvd1 cube_sim_a/scene.pvd \
        --pvd2 cube_sim_b/scene.pvd \
        --out preprocessed_cubes/

Requirements: pip install numpy meshio lxml
"""

import os
import glob
import argparse
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import meshio


HEX_TYPES = {"hexahedron", "hexahedron20", "hexahedron27"}

# (face local indices, opposite face local indices)
HEX_FACES = [
    ((0, 1, 2, 3), (4, 5, 6, 7)),
    ((4, 5, 6, 7), (0, 1, 2, 3)),
    ((0, 1, 5, 4), (2, 3, 7, 6)),
    ((1, 2, 6, 5), (0, 3, 7, 4)),
    ((2, 3, 7, 6), (0, 1, 5, 4)),
    ((3, 0, 4, 7), (1, 2, 6, 5)),
]


def parse_pvd(pvd_path):
    if not os.path.isfile(pvd_path):
        raise FileNotFoundError(f"PVD not found: {pvd_path}")
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    collection = root.find("Collection")
    if collection is None:
        for el in root.iter():
            if el.tag.endswith("Collection"):
                collection = el
                break
    if collection is None:
        raise ValueError(f"No <Collection> in {pvd_path}")

    base = os.path.dirname(os.path.abspath(pvd_path))
    items = []
    for ds in list(collection):
        f = ds.attrib.get("file")
        ts = ds.attrib.get("timestep")
        if f is None:
            continue
        path = f if os.path.isabs(f) else os.path.join(base, f)
        items.append((float(ts) if ts else None, path))

    if not items:
        raise ValueError(f"No dataset entries in {pvd_path}")
    if all(t is not None for t, _ in items):
        items.sort(key=lambda x: x[0])
    else:
        items.sort(key=lambda x: x[1])
    return items


def series_from_vtu_dir(vtu_dir):
    if not os.path.isdir(vtu_dir):
        raise FileNotFoundError(f"VTU directory not found: {vtu_dir}")
    vtus = sorted(glob.glob(os.path.join(vtu_dir, "*.vtu")))
    if not vtus:
        raise ValueError(f"No VTU files found in: {vtu_dir}")
    return [(float(i), f) for i, f in enumerate(vtus)]


def atomic_savez(path, compressed=False, **arrays):
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".npz", dir=out_dir)
    os.close(fd)
    try:
        if compressed:
            np.savez_compressed(tmp_path, **arrays)
        else:
            np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def compact_mesh(vertices, triangles):
    used = np.unique(triangles.ravel())
    new_idx = np.full(len(vertices), -1, dtype=np.int64)
    new_idx[used] = np.arange(len(used))
    return vertices[used], new_idx[triangles]


def _orient_quad_face(vertices, elem, face_local, opp_local):
    face = [int(elem[face_local[0]]), int(elem[face_local[1]]), int(elem[face_local[2]]), int(elem[face_local[3]])]
    opp = [int(elem[opp_local[0]]), int(elem[opp_local[1]]), int(elem[opp_local[2]]), int(elem[opp_local[3]])]

    a = vertices[face[0]]
    b = vertices[face[1]]
    c = vertices[face[2]]
    opp_center = vertices[opp].mean(axis=0)
    n = np.cross(b - a, c - a)

    # Ensure normal points away from the opposite side of the hexahedron.
    if np.dot(n, opp_center - a) > 0:
        face = [face[0], face[3], face[2], face[1]]
    return tuple(face)


def extract_surface_hexes(vertices, elements, labels):
    if len(elements) == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int32)

    face_count = {}
    face_info = {}
    for ei, elem in enumerate(elements):
        for face_local, opp_local in HEX_FACES:
            q = _orient_quad_face(vertices, elem, face_local, opp_local)
            key = tuple(sorted(q))
            if key in face_count:
                face_count[key] += 1
            else:
                face_count[key] = 1
                face_info[key] = (ei, q)

    tris = []
    tri_labels = []
    for key, cnt in face_count.items():
        if cnt != 1:
            continue
        ei, q = face_info[key]
        tris.append([q[0], q[1], q[2]])
        tris.append([q[0], q[2], q[3]])
        tri_labels.append(int(labels[ei]))
        tri_labels.append(int(labels[ei]))

    return np.asarray(tris, dtype=np.int64), np.asarray(tri_labels, dtype=np.int32)


def gather_hex_elements(mesh, stream_name, vtu_path):
    color_data = None
    if mesh.cell_data:
        for key, val in mesh.cell_data.items():
            if key.lower() == "elementcolor" and isinstance(val, list):
                color_data = val
                break

    elements_parts = []
    labels_parts = []

    for bi, blk in enumerate(mesh.cells):
        if blk.type not in HEX_TYPES:
            continue
        elems = np.asarray(blk.data, dtype=np.int64)
        if elems.ndim != 2 or elems.shape[1] < 8:
            raise ValueError(f"[{stream_name}] Invalid {blk.type} block in {vtu_path}: {elems.shape}")
        elems = elems[:, :8]

        labels = np.zeros((len(elems),), dtype=np.int32)
        if color_data is not None and bi < len(color_data):
            raw = np.asarray(color_data[bi]).reshape(-1).astype(np.int32)
            if len(raw) == len(elems):
                labels = raw
            elif len(raw) == 1:
                labels = np.full((len(elems),), int(raw[0]), dtype=np.int32)
            elif len(raw) != 0:
                raise ValueError(
                    f"[{stream_name}] elementColor mismatch in {vtu_path}: "
                    f"{len(raw)} labels vs {len(elems)} {blk.type} cells"
                )

        elements_parts.append(elems)
        labels_parts.append(labels)

    if not elements_parts:
        raise ValueError(f"[{stream_name}] No hexahedral cells in {vtu_path}")

    elements = np.concatenate(elements_parts, axis=0)
    labels = np.concatenate(labels_parts, axis=0)
    return elements, labels


def process_series_hex(series, out_dir, stream_name):
    if not series:
        raise ValueError(f"[{stream_name}] No input frames to process")

    stream_dir = os.path.join(out_dir, stream_name)
    os.makedirs(stream_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(stream_dir, "frame_*.npz")):
        os.remove(stale)
    meta_path = os.path.join(stream_dir, "metadata.npz")
    if os.path.exists(meta_path):
        os.remove(meta_path)

    all_mins = []
    all_maxs = []
    labels_global = set()
    n_total = len(series)
    n_saved = 0

    for fi, (ts, vtu_path) in enumerate(series):
        print(f"  [{stream_name}] {fi+1}/{n_total}: {os.path.basename(vtu_path)}")
        if not os.path.isfile(vtu_path):
            raise FileNotFoundError(f"[{stream_name}] Missing VTU: {vtu_path}")

        mesh = meshio.read(vtu_path)
        vertices = np.asarray(mesh.points, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] < 3:
            raise ValueError(f"[{stream_name}] Invalid vertex array in {vtu_path}: {vertices.shape}")
        if vertices.shape[1] > 3:
            vertices = vertices[:, :3]

        hex_elements, labels_hex = gather_hex_elements(mesh, stream_name, vtu_path)

        # Keep same filtering convention as the original preprocess.
        keep = labels_hex != 10
        if not np.all(keep):
            hex_elements = hex_elements[keep]
            labels_hex = labels_hex[keep]
        if len(hex_elements) == 0:
            print(f"    [{stream_name}] skipping frame: all hexahedra filtered out")
            continue

        surf_tris, labels_tri = extract_surface_hexes(vertices, hex_elements, labels_hex)
        if len(surf_tris) == 0:
            print(f"    [{stream_name}] skipping frame: empty surface after extraction")
            continue

        labels_global.update(np.unique(labels_tri).tolist())
        verts_c, tris_c = compact_mesh(vertices, surf_tris)
        if len(verts_c) == 0 or len(tris_c) == 0:
            print(f"    [{stream_name}] skipping frame: compacted surface is empty")
            continue

        all_mins.append(verts_c.min(axis=0))
        all_maxs.append(verts_c.max(axis=0))
        atomic_savez(
            os.path.join(stream_dir, f"frame_{n_saved:05d}.npz"),
            compressed=True,
            vertices=verts_c.astype(np.float32),
            triangles=tris_c.astype(np.int32),
            labels=labels_tri.astype(np.int32),
            timestep=np.array([ts if ts is not None else float(fi)]),
        )
        n_saved += 1

    if n_saved == 0:
        raise RuntimeError(f"[{stream_name}] No valid frames were written")

    atomic_savez(
        meta_path,
        n_frames=np.array([n_saved]),
        bbox_min=np.min(all_mins, axis=0),
        bbox_max=np.max(all_maxs, axis=0),
        unique_labels=np.array(sorted(labels_global), dtype=np.int32),
    )

    print(f"  [{stream_name}] Saved frames: {n_saved}/{n_total}")
    print(f"  [{stream_name}] Labels: {sorted(labels_global)}")
    print(f"  [{stream_name}] Bbox: {np.min(all_mins, axis=0)} -> {np.max(all_maxs, axis=0)}")


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

    print(f"Cube 1 (hex): {len(s1)} frames\nCube 2 (hex): {len(s2)} frames")
    if len(s1) != len(s2):
        n = min(len(s1), len(s2))
        s1, s2 = s1[:n], s2[:n]
        print(f"Truncating to {n}")

    os.makedirs(args.out, exist_ok=True)
    print("\n--- Cube 1 ---")
    process_series_hex(s1, args.out, "cube1")
    print("\n--- Cube 2 ---")
    process_series_hex(s2, args.out, "cube2")
    print(f"\nDone -> {os.path.abspath(args.out)}")
    print(
        "Next: blender --background --python render_blender_cubes.py -- "
        f"--input {args.out} --output renders/"
    )


if __name__ == "__main__":
    main()
