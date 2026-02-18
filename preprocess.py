#!/usr/bin/env python3
"""
preprocess.py — Extract surface meshes from two fish simulations for Blender.

Usage:
    python preprocess.py \
        --pvd1 fish/ellipsoid.pvd \
        --pvd2 fishseq/seqGeometryActuation.pvd \
        --out preprocessed/

Requirements: pip install numpy meshio lxml
"""

import os, glob, argparse, tempfile
import xml.etree.ElementTree as ET
import numpy as np
import meshio


def parse_pvd(pvd_path):
    if not os.path.isfile(pvd_path):
        raise FileNotFoundError(f"PVD not found: {pvd_path}")
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    collection = root.find("Collection")
    if collection is None:
        for el in root.iter():
            if el.tag.endswith("Collection"):
                collection = el; break
    if collection is None:
        raise ValueError(f"No <Collection> in {pvd_path}")
    base = os.path.dirname(os.path.abspath(pvd_path))
    items = []
    for ds in list(collection):
        f = ds.attrib.get("file"); ts = ds.attrib.get("timestep")
        if f is None: continue
        path = f if os.path.isabs(f) else os.path.join(base, f)
        items.append((float(ts) if ts else None, path))
    if not items:
        raise ValueError(f"No dataset entries in {pvd_path}")
    if all(t is not None for t, _ in items):
        items.sort(key=lambda x: x[0])
    else:
        items.sort(key=lambda x: x[1])
    return items


def extract_surface_tets(vertices, elements):
    if len(elements) == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int64)
    # (face local indices, opposite local index)
    tet_faces = [
        ((0, 1, 2), 3),
        ((0, 1, 3), 2),
        ((0, 2, 3), 1),
        ((1, 2, 3), 0),
    ]
    face_count = {}
    face_info = {}
    for ei, elem in enumerate(elements):
        for fl, opp in tet_faces:
            verts = [int(elem[fl[0]]), int(elem[fl[1]]), int(elem[fl[2]])]
            opp_v = int(elem[opp])

            # Orient each tetra face so its normal points away from the opposite vertex.
            a, b, c = vertices[verts[0]], vertices[verts[1]], vertices[verts[2]]
            d = vertices[opp_v]
            n = np.cross(b - a, c - a)
            if np.dot(n, d - a) > 0:
                verts[1], verts[2] = verts[2], verts[1]

            key = tuple(sorted(verts))
            if key in face_count:
                face_count[key] += 1
            else:
                face_count[key] = 1
                face_info[key] = (ei, tuple(verts))
    tris, tri2elem = [], []
    for key, cnt in face_count.items():
        if cnt == 1:
            ei, v = face_info[key]
            tris.append(list(v)); tri2elem.append(ei)
    return np.array(tris, dtype=np.int64), np.array(tri2elem, dtype=np.int64)


def compact_mesh(vertices, triangles):
    used = np.unique(triangles.ravel())
    new_idx = np.full(len(vertices), -1, dtype=np.int64)
    new_idx[used] = np.arange(len(used))
    return vertices[used], new_idx[triangles]


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


def process_series(series, out_dir, fish_name):
    if not series:
        raise ValueError(f"[{fish_name}] No input frames to process")

    fish_dir = os.path.join(out_dir, fish_name)
    os.makedirs(fish_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(fish_dir, "frame_*.npz")):
        os.remove(stale)
    meta_path = os.path.join(fish_dir, "metadata.npz")
    if os.path.exists(meta_path):
        os.remove(meta_path)

    all_mins, all_maxs = [], []
    labels_global = set()
    n_total = len(series)
    n_saved = 0

    for fi, (ts, vtu_path) in enumerate(series):
        print(f"  [{fish_name}] {fi+1}/{n_total}: {os.path.basename(vtu_path)}")
        if not os.path.isfile(vtu_path):
            raise FileNotFoundError(f"[{fish_name}] Missing VTU: {vtu_path}")
        mesh = meshio.read(vtu_path)
        vertices = np.asarray(mesh.points, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] < 3:
            raise ValueError(f"[{fish_name}] Invalid vertex array in {vtu_path}: {vertices.shape}")
        if vertices.shape[1] > 3:
            vertices = vertices[:, :3]

        tet_elements = None; tet_block_idx = None
        for i, blk in enumerate(mesh.cells):
            if blk.type == "tetra":
                tet_elements = np.asarray(blk.data, dtype=np.int64)
                tet_block_idx = i; break
        if tet_elements is None:
            raise ValueError(f"No tetrahedra in {vtu_path}")
        if tet_elements.ndim != 2 or tet_elements.shape[1] != 4:
            raise ValueError(f"[{fish_name}] Invalid tetrahedra in {vtu_path}: {tet_elements.shape}")

        labels = np.zeros(len(tet_elements), dtype=np.int32)
        if mesh.cell_data:
            for key in mesh.cell_data:
                if key.lower() == "elementcolor":
                    arr = mesh.cell_data[key]
                    if isinstance(arr, list) and tet_block_idx < len(arr):
                        labels = np.asarray(arr[tet_block_idx]).reshape(-1).astype(np.int32)
                    break

        keep = labels != 10
        if not np.all(keep):
            tet_elements = tet_elements[keep]; labels = labels[keep]
        if len(tet_elements) == 0:
            print(f"    [{fish_name}] skipping frame: all tetrahedra filtered out")
            continue

        surf_tris, tri2elem = extract_surface_tets(vertices, tet_elements)
        if len(surf_tris) == 0:
            print(f"    [{fish_name}] skipping frame: empty surface after extraction")
            continue
        labels_tri = labels[tri2elem]
        labels_global.update(np.unique(labels_tri).tolist())

        verts_c, tris_c = compact_mesh(vertices, surf_tris)
        if len(verts_c) == 0 or len(tris_c) == 0:
            print(f"    [{fish_name}] skipping frame: compacted surface is empty")
            continue
        all_mins.append(verts_c.min(axis=0))
        all_maxs.append(verts_c.max(axis=0))

        atomic_savez(
            os.path.join(fish_dir, f"frame_{n_saved:05d}.npz"),
            compressed=True,
            vertices=verts_c.astype(np.float32),
            triangles=tris_c.astype(np.int32),
            labels=labels_tri.astype(np.int32),
            timestep=np.array([ts if ts is not None else float(fi)]),
        )
        n_saved += 1

    if n_saved == 0:
        raise RuntimeError(f"[{fish_name}] No valid frames were written")

    atomic_savez(
        meta_path,
        n_frames=np.array([n_saved]),
        bbox_min=np.min(all_mins, axis=0),
        bbox_max=np.max(all_maxs, axis=0),
        unique_labels=np.array(sorted(labels_global), dtype=np.int32),
    )

    print(f"  [{fish_name}] Saved frames: {n_saved}/{n_total}")
    print(f"  [{fish_name}] Labels: {sorted(labels_global)}")
    print(f"  [{fish_name}] Bbox: {np.min(all_mins,axis=0)} → {np.max(all_maxs,axis=0)}")


def series_from_vtu_dir(vtu_dir):
    if not os.path.isdir(vtu_dir):
        raise FileNotFoundError(f"VTU directory not found: {vtu_dir}")
    vtus = sorted(glob.glob(os.path.join(vtu_dir, "*.vtu")))
    if not vtus:
        raise ValueError(f"No VTU files found in: {vtu_dir}")
    return [(float(i), f) for i, f in enumerate(vtus)]


def main():
    ap = argparse.ArgumentParser()
    g1 = ap.add_mutually_exclusive_group(required=True)
    g1.add_argument("--pvd1"); g1.add_argument("--vtu-dir1")
    g2 = ap.add_mutually_exclusive_group(required=True)
    g2.add_argument("--pvd2"); g2.add_argument("--vtu-dir2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s1 = parse_pvd(args.pvd1) if args.pvd1 else series_from_vtu_dir(args.vtu_dir1)
    s2 = parse_pvd(args.pvd2) if args.pvd2 else series_from_vtu_dir(args.vtu_dir2)

    print(f"Fish 1: {len(s1)} frames\nFish 2: {len(s2)} frames")
    if len(s1) != len(s2):
        n = min(len(s1),len(s2)); s1,s2 = s1[:n],s2[:n]
        print(f"Truncating to {n}")

    os.makedirs(args.out, exist_ok=True)
    print("\n--- Fish 1 ---"); process_series(s1, args.out, "fish1")
    print("\n--- Fish 2 ---"); process_series(s2, args.out, "fish2")
    print(f"\nDone → {os.path.abspath(args.out)}")
    print(f"Next: blender --background --python render_blender.py -- --input {args.out} --output renders/")


if __name__ == "__main__":
    main()
