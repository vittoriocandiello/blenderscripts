#!/usr/bin/env python3
"""
render_blender.py — Headless Blender renderer for two-fish FEM comparison.
Black background, distinct color themes, orthographic top-down view.

Usage:
    blender --background --python render_blender.py -- \
        --input preprocessed/ --output renders/ \
        --resolution 480 270 --samples 16

    blender --background --python render_blender.py -- \
        --input preprocessed/ --output renders/ \
        --resolution 1920 1080 --samples 128

Smoothing controls:
    --blend-iters 5          # number of diffusion iterations (0 = off)
    --blend-self-weight 0.1  # how much a vertex keeps its own color vs neighbors
    --boundary-weight 0.1    # cross-label edge weight (0 = hard boundary, 1 = ignore labels)
    --smooth-shading          # enable smooth shading on fish meshes
    --color-domain vertex     # 'vertex' for smooth gradients, 'face' for flat per-triangle
"""

import os, sys, glob, argparse, subprocess, math, tempfile
import numpy as np
import bpy
from mathutils import Vector


# ==================================================================
#  SETTINGS
# ==================================================================

class Settings:

    # ── Resolution ────────────────────────────────────────────
    resolution      = (1920, 1080)
    render_samples  = 64        # 16=test, 128+=final
    render_engine   = "CYCLES"  # "CYCLES" or "BLENDER_EEVEE_NEXT"
    use_gpu         = True
    fps             = 60

    # ── Camera ────────────────────────────────────────────────
    ortho_padding      = 1.10
    camera_follow      = False
    camera_track_alpha = 1.00
    camera_height_min  = 8.0
    camera_height_mult = 2.3

    # ── Background ────────────────────────────────────────────
    world_color      = (0.0, 0.0, 0.0)
    world_strength   = 0.0
    checkerboard_enable = True
    checker_tiles_x  = 22
    checker_margin   = 0.20
    checker_cover_scale = 1.30
    checker_color_a  = (0.0, 0.0, 0.0)      # black tile
    checker_color_b  = (0.22, 0.22, 0.22)   # dark gray tile
    checker_emission = 1.0
    checker_z_offset = 0.10

    # ── Lighting ──────────────────────────────────────────────
    key_energy       = 520.0
    key_color        = (1.0, 0.95, 0.90)    # warm key
    key_size_factor  = 0.55

    fill_energy      = 130.0
    fill_color       = (0.90, 0.93, 1.0)    # cool fill
    fill_size_factor = 0.50

    rim_energy       = 120.0
    rim_color        = (1.0, 0.97, 0.94)    # warm rim
    rim_size_factor  = 0.45

    light_spread     = 1.05
    light_z_factor   = 0.86

    # ── Fish layout ───────────────────────────────────────────
    fish_y_gap      = 0.0

    # ── Color themes ──────────────────────────────────────────
    theme_fish1 = {
        "name": "StudioBlue",
        "default": (0.064, 0.176, 0.60),    # body
        0: (0.12, 0.20, 0.88),              # light blue
        1: (0.45, 0.08, 0.10),              # red
        2: (0.45, 0.08, 0.10),              # red
        3: (0.98, 0.66, 0.22),              # high-contrast golden yellow
        4: (0.31, 0.10, 0.38),              # deep violet
        5: (0.18, 0.34, 0.08),              # dark olive green
        6: (0.08, 0.32, 0.28),              # dark teal
        7: (0.40, 0.28, 0.04),              # dark amber
        8: (0.08, 0.30, 0.18),              # dark forest green
        9: (0.38, 0.08, 0.24),              # deep rose
        10: (0.42, 0.18, 0.06),             # dark burnt orange
    }

    theme_fish2 = {
        "name": "StudioBlue",
        "default": (0.064, 0.176, 0.60),
        0: (0.12, 0.20, 0.88),
        1: (0.45, 0.08, 0.10),
        2: (0.45, 0.08, 0.10),
        3: (0.98, 0.66, 0.22),
        4: (0.31, 0.10, 0.38),
        5: (0.18, 0.34, 0.08),
        6: (0.08, 0.32, 0.28),
        7: (0.40, 0.28, 0.04),
        8: (0.08, 0.30, 0.18),
        9: (0.38, 0.08, 0.24),
        10: (0.42, 0.18, 0.06),
    }

    # ── Material ──────────────────────────────────────────────
    roughness       = 0.85
    specular        = 0.05
    metallic        = 0.0
    emission_body   = 0.005
    emission_label  = 0.035
    color_attribute_name = "LabelColor"

    # ── Smoothing controls ────────────────────────────────────
    color_blend_iters      = 1      # diffusion iterations (0 = no smoothing)
    color_blend_self_weight = 10.00  # vertex self-retention vs neighbor average
    boundary_weight        = 0.01    # cross-label edge weight: 0=hard, 1=no distinction
    color_domain           = "vertex"  # "vertex" for smooth gradients, "face" for flat
    smooth_shading         = True   # mesh smooth shading
    auto_smooth_angle      = 48.0   # degrees, for auto-smooth

    # ── Reference lines (invisible) ───────────────────────────
    line_color      = (0.0, 0.0, 0.0)
    line_thickness  = 0.0
    line_extend     = 0.0
    line_emission   = 0.0

    # ── Color management ──────────────────────────────────────
    exposure        = 0.0
    gamma           = 1.0

    # ── Video ─────────────────────────────────────────────────
    video_codec     = "libx264"
    video_crf       = 17
    allow_missing_stream = False


CFG = Settings()


# ==================================================================
#  UTILITIES
# ==================================================================

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for b in list(bpy.data.meshes):
        if b.users == 0: bpy.data.meshes.remove(b)
    for b in list(bpy.data.materials):
        if b.users == 0: bpy.data.materials.remove(b)
    for b in list(bpy.data.lights):
        if b.users == 0: bpy.data.lights.remove(b)
    for b in list(bpy.data.cameras):
        if b.users == 0: bpy.data.cameras.remove(b)


def build_metadata_from_frames(frames):
    bbox_min = np.min([fr["vertices"].min(axis=0) for fr in frames], axis=0).astype(np.float64)
    bbox_max = np.max([fr["vertices"].max(axis=0) for fr in frames], axis=0).astype(np.float64)
    labels_all = np.concatenate([fr["labels"].astype(np.int32) for fr in frames], axis=0)
    unique_labels = np.unique(labels_all).astype(np.int32)
    return {
        "n_frames": np.array([len(frames)], dtype=np.int32),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "unique_labels": unique_labels,
    }


def atomic_savez(path, **arrays):
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".npz", dir=out_dir)
    os.close(fd)
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_metadata(meta_path, frames):
    meta = build_metadata_from_frames(frames)
    atomic_savez(meta_path, **meta)
    return meta


def parse_frame_npz(npz_obj, fp, fish_name):
    required = ("vertices", "triangles", "labels")
    missing = [k for k in required if k not in npz_obj]
    if missing:
        raise ValueError(f"[{fish_name}] {fp} missing keys: {missing}")

    vertices = np.asarray(npz_obj["vertices"], dtype=np.float32)
    triangles = np.asarray(npz_obj["triangles"], dtype=np.int32)
    labels = np.asarray(npz_obj["labels"], dtype=np.int32).reshape(-1)

    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError(f"[{fish_name}] invalid vertices in {fp}: {vertices.shape}")
    if vertices.shape[1] > 3:
        vertices = vertices[:, :3]

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"[{fish_name}] invalid triangles in {fp}: {triangles.shape}")
    if len(triangles) == 0:
        raise ValueError(f"[{fish_name}] empty triangles in {fp}")

    if len(labels) != len(triangles):
        if len(labels) == 1:
            labels = np.full((len(triangles),), int(labels[0]), dtype=np.int32)
        else:
            raise ValueError(
                f"[{fish_name}] label/triangle mismatch in {fp}: "
                f"{len(labels)} labels vs {len(triangles)} triangles"
            )

    return {"vertices": vertices, "triangles": triangles, "labels": labels}


def load_frames(input_dir, fish_name, allow_missing=False):
    fish_dir = os.path.join(input_dir, fish_name)
    if not os.path.isdir(fish_dir):
        if allow_missing:
            return [], None
        raise FileNotFoundError(f"[{fish_name}] directory not found: {fish_dir}")

    frame_paths = sorted(glob.glob(os.path.join(fish_dir, "frame_*.npz")))
    if not frame_paths:
        if allow_missing:
            return [], None
        raise FileNotFoundError(f"[{fish_name}] no frame_*.npz files in {fish_dir}")

    frames = []
    for fp in frame_paths:
        try:
            with np.load(fp, allow_pickle=False) as d:
                frames.append(parse_frame_npz(d, fp, fish_name))
        except Exception as exc:
            print(f"  [{fish_name}] skipping invalid frame {os.path.basename(fp)} ({exc})")

    if not frames:
        if allow_missing:
            return [], None
        raise ValueError(f"[{fish_name}] no valid frame data in {fish_dir}")

    meta_path = os.path.join(fish_dir, "metadata.npz")
    meta = None
    if os.path.isfile(meta_path):
        try:
            with np.load(meta_path, allow_pickle=False) as raw_meta:
                meta = {k: np.asarray(raw_meta[k]) for k in raw_meta.files}
            required = ("n_frames", "bbox_min", "bbox_max", "unique_labels")
            if any(k not in meta for k in required):
                raise ValueError("missing metadata fields")
            if tuple(np.asarray(meta["bbox_min"]).shape) != (3,) or tuple(np.asarray(meta["bbox_max"]).shape) != (3,):
                raise ValueError("invalid bbox shape")
            n_meta = int(np.asarray(meta["n_frames"]).reshape(-1)[0])
            if n_meta != len(frames):
                raise ValueError("frame count mismatch")
        except Exception as exc:
            print(f"  [{fish_name}] invalid metadata.npz ({exc}); rebuilding.")
            meta = save_metadata(meta_path, frames)
    else:
        print(f"  [{fish_name}] metadata.npz missing; rebuilding from frame files.")
        meta = save_metadata(meta_path, frames)

    return frames, meta


def set_node_input(node, name_or_names, value):
    names = (name_or_names,) if isinstance(name_or_names, str) else tuple(name_or_names)
    for nm in names:
        if nm in node.inputs:
            node.inputs[nm].default_value = value
            return True
    return False


def to_rgba(color):
    c = tuple(color)
    if len(c) == 4:
        return c
    if len(c) == 3:
        return (c[0], c[1], c[2], 1.0)
    raise ValueError(f"Expected RGB or RGBA color, got length {len(c)}: {c}")


def create_material(name, emission_strength):
    """Surface material driven by per-vertex/per-corner label colors."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for n in list(nodes): nodes.remove(n)

    output = nodes.new("ShaderNodeOutputMaterial")
    shader_add = nodes.new("ShaderNodeAddShader")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    emission = nodes.new("ShaderNodeEmission")
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = CFG.color_attribute_name

    set_node_input(principled, "Roughness", CFG.roughness)
    set_node_input(principled, ["Specular IOR Level", "Specular"], CFG.specular)
    set_node_input(principled, "Metallic", CFG.metallic)

    set_node_input(emission, "Strength", emission_strength)

    links.new(attr.outputs["Color"], principled.inputs["Base Color"])
    links.new(attr.outputs["Color"], emission.inputs["Color"])
    links.new(principled.outputs["BSDF"], shader_add.inputs[0])
    links.new(emission.outputs["Emission"], shader_add.inputs[1])
    links.new(shader_add.outputs["Shader"], output.inputs["Surface"])

    return mat


def build_materials(theme, prefix):
    return {
        "theme": theme,
        "surface": create_material(f"{prefix}_surface", CFG.emission_label),
    }


def mesh_from_arrays(name, vertices, triangles):
    m = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, m)
    bpy.context.collection.objects.link(obj)
    m.from_pydata(vertices.tolist(), [], triangles.tolist())
    m.update()
    return obj


def labels_to_face_colors(labels, theme):
    default = np.asarray(theme.get("default", (0.7, 0.7, 0.7)), dtype=np.float32)
    labels_i = labels.astype(np.int32, copy=False)
    face_colors = np.repeat(default[None, :], len(labels_i), axis=0)
    for lab in np.unique(labels_i):
        face_colors[labels_i == lab] = np.asarray(theme.get(int(lab), default), dtype=np.float32)
    return face_colors


def triangles_to_edges(triangles):
    tri = np.asarray(triangles, dtype=np.int64)
    if len(tri) == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


# ------------------------------------------------------------------
#  Vertex-color smoothing (boundary-aware Laplacian diffusion)
# ------------------------------------------------------------------

def _build_edge_weights(triangles, labels, n_verts, boundary_weight):
    """
    Build unique edge list with weights: 1.0 for same-label edges,
    `boundary_weight` for edges shared by faces with different labels.

    Correctly handles boundary edges (shared by only one face).
    """
    tri = np.asarray(triangles, dtype=np.int64)
    lab = np.asarray(labels, dtype=np.int32)
    n_faces = len(tri)

    # Each face contributes 3 directed half-edges
    # half_edges[i] = (v_lo, v_hi, face_label)
    he_v0 = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
    he_v1 = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
    he_lab = np.tile(lab, 3)

    # Sort each edge so v_lo < v_hi
    swap = he_v0 > he_v1
    he_v0_s = np.where(swap, he_v1, he_v0)
    he_v1_s = np.where(swap, he_v0, he_v1)

    # Encode edge as single int for grouping
    max_v = n_verts + 1
    edge_key = he_v0_s * max_v + he_v1_s

    # Sort by edge key to group half-edges belonging to the same edge
    order = np.argsort(edge_key)
    ek_sorted = edge_key[order]
    lab_sorted = he_lab[order]
    v0_sorted = he_v0_s[order]
    v1_sorted = he_v1_s[order]

    # Find unique edges
    unique_mask = np.empty(len(ek_sorted), dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = ek_sorted[1:] != ek_sorted[:-1]
    u_idx = np.where(unique_mask)[0]
    n_unique = len(u_idx)

    u_v0 = v0_sorted[u_idx]
    u_v1 = v1_sorted[u_idx]

    # Determine if edge is interior (shared by 2 faces) and whether labels match
    # For each unique edge start, check if next half-edge belongs to same edge
    # and whether labels differ.
    next_idx = np.minimum(u_idx + 1, len(ek_sorted) - 1)
    is_interior = (ek_sorted[next_idx] == ek_sorted[u_idx])  # True if 2+ half-edges
    lab_a = lab_sorted[u_idx]
    lab_b = lab_sorted[next_idx]
    same_label = (~is_interior) | (lab_a == lab_b)  # boundary edges count as "same"

    bw = float(np.clip(boundary_weight, 0.0, 1.0))
    edge_w = np.where(same_label, 1.0, bw).astype(np.float32)

    return u_v0, u_v1, edge_w


def blend_vertex_colors_boundary_aware(vertex_colors, triangles, labels,
                                        iters, self_weight, boundary_weight):
    """
    Laplacian diffusion of vertex colors with boundary-aware edge weights.

    Parameters
    ----------
    vertex_colors : (V, 3) float32
    triangles     : (F, 3) int
    labels        : (F,) int   — per-face labels
    iters         : int        — diffusion iterations; 0 = no-op
    self_weight   : float      — how much a vertex retains its color vs neighbor avg
    boundary_weight : float    — weight for edges crossing label boundaries
                                 0.0 = hard boundary (no diffusion across)
                                 1.0 = ignore labels entirely

    Returns
    -------
    (V, 3) float32, clipped to [0, 1]
    """
    if iters <= 0 or len(vertex_colors) == 0:
        return vertex_colors.copy()

    n_verts = len(vertex_colors)
    u_v0, u_v1, edge_w = _build_edge_weights(triangles, labels, n_verts, boundary_weight)

    if len(u_v0) == 0:
        return vertex_colors.copy()

    cur = vertex_colors.astype(np.float32, copy=True)
    sw = float(max(self_weight, 0.0))

    ew3 = edge_w[:, None]  # (E, 1) for broadcasting against (E, 3) colors

    for _ in range(int(iters)):
        nb_sum = np.zeros_like(cur)
        w_sum = np.zeros((n_verts, 1), dtype=np.float32)

        np.add.at(nb_sum, u_v0, cur[u_v1] * ew3)
        np.add.at(nb_sum, u_v1, cur[u_v0] * ew3)
        np.add.at(w_sum[:, 0], u_v0, edge_w)
        np.add.at(w_sum[:, 0], u_v1, edge_w)

        nb_avg = nb_sum / np.maximum(w_sum, 1e-8)
        cur = (sw * cur + nb_avg) / (sw + 1.0)

    return np.clip(cur, 0.0, 1.0)


# ------------------------------------------------------------------
#  Color attribute writing
# ------------------------------------------------------------------

def set_mesh_vertex_colors(mesh, vertex_colors, attr_name):
    """
    Write per-VERTEX colors into a POINT-domain color attribute.
    Blender interpolates across faces automatically → smooth gradients.

    vertex_colors : (V, 3) float32
    """
    attrs = mesh.color_attributes
    attr = attrs.get(attr_name)
    if attr is not None:
        attrs.remove(attr)
    attr = attrs.new(name=attr_name, type="FLOAT_COLOR", domain="POINT")

    n_verts = len(mesh.vertices)
    if n_verts == 0:
        return

    rgba = np.empty((n_verts, 4), dtype=np.float32)
    rgba[:, :3] = vertex_colors[:n_verts]
    rgba[:, 3] = 1.0
    attr.data.foreach_set("color", rgba.reshape(-1))


def set_mesh_face_colors(mesh, face_colors, attr_name):
    """
    Write per-FACE colors into a CORNER-domain color attribute.
    Each triangle's 3 corners get the same color → flat per-face look.

    face_colors : (F, 3) float32
    """
    attrs = mesh.color_attributes
    attr = attrs.get(attr_name)
    if attr is not None:
        attrs.remove(attr)
    attr = attrs.new(name=attr_name, type="FLOAT_COLOR", domain="CORNER")

    n_loops = len(mesh.loops)
    if n_loops == 0:
        return

    # Each face has 3 loops (corners); repeat face color for all 3
    loop_colors = np.repeat(face_colors, 3, axis=0)  # (F*3, 3)
    rgba = np.empty((n_loops, 4), dtype=np.float32)
    rgba[:, :3] = loop_colors[:n_loops]
    rgba[:, 3] = 1.0
    attr.data.foreach_set("color", rgba.reshape(-1))


# ------------------------------------------------------------------
#  Material + color assignment (the fixed pipeline)
# ------------------------------------------------------------------

def assign_materials(obj, labels, materials_dict):
    """
    Assign material and compute + write color attribute.

    Pipeline:
    1. Map face labels → face RGB colors via theme.
    2. Scatter face colors to vertices (area-based averaging).
    3. Smooth vertex colors with boundary-aware Laplacian diffusion.
    4. Write colors to mesh attribute in the chosen domain.
       - "vertex" domain: blended vertex colors → smooth gradients
       - "face" domain:   raw face colors → flat per-triangle
    """
    mesh = obj.data
    n_polys = len(mesh.polygons)
    if len(labels) != n_polys:
        raise ValueError(
            f"{obj.name}: {len(labels)} labels for {n_polys} polygons"
        )

    theme = materials_dict["theme"]
    surface_mat = materials_dict["surface"]
    if len(obj.data.materials) == 0 or obj.data.materials[0] is not surface_mat:
        obj.data.materials.clear()
        obj.data.materials.append(surface_mat)

    # Step 1: per-face colors from theme
    face_colors = labels_to_face_colors(labels, theme)  # (F, 3)

    if CFG.color_domain == "face" or CFG.color_blend_iters <= 0:
        # No smoothing — write flat face colors
        set_mesh_face_colors(mesh, face_colors, CFG.color_attribute_name)
        return

    # Step 2: scatter face colors to vertices
    triangles = np.array([poly.vertices[:] for poly in mesh.polygons], dtype=np.int64)
    n_verts = len(mesh.vertices)
    vcols = np.zeros((n_verts, 3), dtype=np.float32)
    counts = np.zeros((n_verts, 1), dtype=np.float32)
    for k in range(3):
        idx = triangles[:, k]
        np.add.at(vcols, idx, face_colors)
        np.add.at(counts[:, 0], idx, 1.0)
    vcols /= np.maximum(counts, 1.0)

    # Step 3: boundary-aware Laplacian smoothing
    vcols = blend_vertex_colors_boundary_aware(
        vcols, triangles, labels,
        CFG.color_blend_iters,
        CFG.color_blend_self_weight,
        CFG.boundary_weight,
    )

    # Step 4: write smoothed vertex colors (POINT domain → Blender interpolates)
    set_mesh_vertex_colors(mesh, vcols, CFG.color_attribute_name)


def update_mesh(obj, vertices, triangles):
    old = obj.data
    new = bpy.data.meshes.new(obj.name + "_tmp")
    new.from_pydata(vertices.tolist(), [], triangles.tolist())
    new.update()
    obj.data = new
    bpy.data.meshes.remove(old)


def ensure_outward_normals(obj):
    mesh = obj.data
    if len(mesh.polygons) == 0:
        return
    verts = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
    tris = np.array([p.vertices[:] for p in mesh.polygons], dtype=np.int64)
    tri = verts[tris]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    centers = tri.mean(axis=1)
    centroid = verts.mean(axis=0)
    orient_score = np.mean(np.einsum("ij,ij->i", normals, centers - centroid))
    if orient_score < 0:
        mesh.flip_normals()
        mesh.update()


def apply_surface_shading(obj):
    """Configure mesh shading: smooth or flat, with optional auto-smooth."""
    use_smooth = CFG.smooth_shading
    for poly in obj.data.polygons:
        poly.use_smooth = use_smooth
    if use_smooth and hasattr(obj.data, "use_auto_smooth"):
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = math.radians(CFG.auto_smooth_angle)


def smooth_track(values, alpha):
    if not values:
        return []
    a = min(max(alpha, 0.0), 1.0)
    out = [values[0]]
    cur = values[0]
    for v in values[1:]:
        cur = a * v + (1.0 - a) * cur
        out.append(cur)
    return out


def robust_y_span(frames, percentile=95.0):
    spans = []
    for fr in frames:
        vy = fr["vertices"][:, 1]
        spans.append(float(vy.max() - vy.min()))
    if not spans:
        return 1.0
    return float(np.percentile(np.array(spans), percentile))


def robust_center_y(frames):
    centers = []
    for fr in frames:
        centers.append(float(np.median(fr["vertices"][:, 1])))
    if not centers:
        return 0.0
    return float(np.median(np.array(centers)))


def compute_frame_coverage(frames1, frames2, y_off1, y_off2, n):
    frame_centers_x = []
    frame_centers_y = []
    frame_x_min = []
    frame_x_max = []
    frame_y_min = []
    frame_y_max = []
    x_min = float("inf")
    x_max = float("-inf")
    y_min = float("inf")
    y_max = float("-inf")
    z_min = float("inf")
    z_max = float("-inf")
    max_span_x = 0.0
    max_span_y = 0.0

    for fi in range(n):
        v1 = frames1[fi]["vertices"]
        v2 = frames2[fi]["vertices"]

        mn1 = v1.min(axis=0).astype(np.float64)
        mx1 = v1.max(axis=0).astype(np.float64)
        mn2 = v2.min(axis=0).astype(np.float64)
        mx2 = v2.max(axis=0).astype(np.float64)
        mn1[1] += y_off1; mx1[1] += y_off1
        mn2[1] += y_off2; mx2[1] += y_off2

        fmin = np.minimum(mn1, mn2)
        fmax = np.maximum(mx1, mx2)
        span = fmax - fmin

        frame_centers_x.append(float((fmin[0] + fmax[0]) * 0.5))
        frame_centers_y.append(float((fmin[1] + fmax[1]) * 0.5))
        frame_x_min.append(float(fmin[0]))
        frame_x_max.append(float(fmax[0]))
        frame_y_min.append(float(fmin[1]))
        frame_y_max.append(float(fmax[1]))
        x_min = min(x_min, float(fmin[0]))
        x_max = max(x_max, float(fmax[0]))
        y_min = min(y_min, float(fmin[1]))
        y_max = max(y_max, float(fmax[1]))
        z_min = min(z_min, float(fmin[2]))
        z_max = max(z_max, float(fmax[2]))
        max_span_x = max(max_span_x, float(span[0]))
        max_span_y = max(max_span_y, float(span[1]))

    aspect = CFG.resolution[0] / CFG.resolution[1]
    ortho_scale = max(
        max_span_x * CFG.ortho_padding,
        max_span_y * CFG.ortho_padding * aspect,
        1e-3,
    )

    return {
        "frame_centers_x": frame_centers_x,
        "frame_centers_y": frame_centers_y,
        "frame_x_min": frame_x_min,
        "frame_x_max": frame_x_max,
        "frame_y_min": frame_y_min,
        "frame_y_max": frame_y_max,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
        "z_span": max(z_max - z_min, 1e-3),
        "ortho_scale": ortho_scale,
    }


def compute_required_ortho_scale(coverage, x_track, y_track, aspect, padding):
    if aspect <= 0:
        aspect = 1.0
    pad = max(float(padding), 1.0)
    required = 1e-3
    for fi, (cx, cy) in enumerate(zip(x_track, y_track)):
        half_x = max(
            cx - coverage["frame_x_min"][fi],
            coverage["frame_x_max"][fi] - cx,
        )
        half_y = max(
            cy - coverage["frame_y_min"][fi],
            coverage["frame_y_max"][fi] - cy,
        )
        required = max(required, max(2.0 * half_x, 2.0 * half_y * aspect))
    return max(required * pad, 1e-3)


# ==================================================================
#  SCENE SETUP
# ==================================================================

def setup_world():
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new("ShaderNodeBackground")
    out = nodes.new("ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Color"].default_value = (*CFG.world_color, 1.0)
    bg.inputs["Strength"].default_value = CFG.world_strength


def setup_camera(track_x0, track_y, track_z, ortho_scale, z_span):
    rig = bpy.data.objects.new("CameraRig", None)
    bpy.context.collection.objects.link(rig)
    rig.location = Vector((track_x0, track_y, track_z))

    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_obj.parent = rig

    cam_data.type = "ORTHO"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.ortho_scale = ortho_scale

    cam_height = max(
        CFG.camera_height_min,
        z_span * CFG.camera_height_mult,
        ortho_scale * 2.0,
    )
    cam_obj.location = Vector((0.0, 0.0, cam_height))
    cam_obj.rotation_euler = (0, 0, 0)

    cam_data.clip_start = 0.01
    cam_data.clip_end = max(1000.0, cam_height * 40.0)

    return rig, cam_obj, cam_height


def setup_lights(rig, ortho_scale, cam_height):
    spread = max(ortho_scale * CFG.light_spread, 3.0)
    base_size = max(ortho_scale * 0.75, 1.5)
    light_z = max(cam_height * CFG.light_z_factor, spread * 1.2)
    energy_scale = max((ortho_scale / 6.0) ** 2, 0.35)

    def add_light(name, energy, color, size_factor, offset):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.energy = energy * energy_scale
        ld.color = color
        ld.shape = 'DISK'
        ld.size = base_size * size_factor
        lo = bpy.data.objects.new(name, ld)
        bpy.context.collection.objects.link(lo)
        lo.parent = rig
        lo.location = Vector((offset[0], offset[1], light_z + offset[2]))
        lo.rotation_euler = (0, 0, 0)
        return lo

    add_light("Key", CFG.key_energy, CFG.key_color, CFG.key_size_factor,
              (-spread * 0.40, -spread * 0.40, 0.0))
    add_light("Fill", CFG.fill_energy, CFG.fill_color, CFG.fill_size_factor,
              (spread * 0.42, spread * 0.12, -spread * 0.12))
    add_light("Rim", CFG.rim_energy, CFG.rim_color, CFG.rim_size_factor,
              (0.0, spread * 0.52, -spread * 0.18))


def create_reference_line(x_min, x_max, y_pos, z_pos):
    x_ext = max(x_max - x_min, 1e-3)
    x_lo = x_min - x_ext * CFG.line_extend
    x_hi = x_max + x_ext * CFG.line_extend
    length = max(x_hi - x_lo, 1e-3)
    cx = (x_lo + x_hi) / 2.0

    bpy.ops.mesh.primitive_cylinder_add(
        radius=CFG.line_thickness, depth=length,
        location=(cx, y_pos, z_pos),
        rotation=(0, math.pi / 2, 0),
    )
    line = bpy.context.active_object
    line.name = f"Line_y{y_pos:.3f}"
    mat = bpy.data.materials.new(f"LineMat_{y_pos:.3f}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    out = nodes.new("ShaderNodeOutputMaterial")
    em = nodes.new("ShaderNodeEmission")
    set_node_input(em, "Color", to_rgba(CFG.line_color))
    set_node_input(em, "Strength", CFG.line_emission)
    links.new(em.outputs["Emission"], out.inputs["Surface"])
    line.data.materials.append(mat)
    return line


def create_checkerboard_background(x_min, x_max, y_min, y_max, z_pos):
    if not CFG.checkerboard_enable:
        return None

    span_x = max(x_max - x_min, 1e-3)
    span_y = max(y_max - y_min, 1e-3)
    ext_x = span_x * (1.0 + 2.0 * CFG.checker_margin)
    ext_y = span_y * (1.0 + 2.0 * CFG.checker_margin)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(cx, cy, z_pos))
    plane = bpy.context.active_object
    plane.name = "ScaleCheckerboard"
    plane.scale = (0.5 * ext_x, 0.5 * ext_y, 1.0)

    mat = bpy.data.materials.new("ScaleCheckerboardMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    checker = nodes.new("ShaderNodeTexChecker")
    emission = nodes.new("ShaderNodeEmission")
    out = nodes.new("ShaderNodeOutputMaterial")

    tile_size = max(span_x / max(CFG.checker_tiles_x, 2), 1e-3)
    set_node_input(mapping, "Scale", (ext_x / tile_size, ext_y / tile_size, 1.0))
    set_node_input(checker, "Color1", to_rgba(CFG.checker_color_a))
    set_node_input(checker, "Color2", to_rgba(CFG.checker_color_b))
    set_node_input(checker, "Scale", 1.0)
    set_node_input(emission, "Strength", CFG.checker_emission)

    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], out.inputs["Surface"])

    plane.data.materials.append(mat)
    return plane


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = CFG.render_engine
    scene.render.resolution_x = CFG.resolution[0]
    scene.render.resolution_y = CFG.resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.use_motion_blur = False

    for vt in ("Standard", "AgX", "Filmic"):
        try:
            scene.view_settings.view_transform = vt
            break
        except Exception:
            continue
    for lk in ("High Contrast", "Medium High Contrast", "Medium Contrast", "None"):
        try:
            scene.view_settings.look = lk
            break
        except Exception:
            continue
    scene.view_settings.exposure = CFG.exposure
    scene.view_settings.gamma = CFG.gamma

    if CFG.render_engine == "CYCLES":
        scene.cycles.samples = CFG.render_samples
        scene.cycles.use_denoising = True
        if hasattr(scene.cycles, "use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling = True
        if hasattr(scene.cycles, "adaptive_threshold"):
            scene.cycles.adaptive_threshold = 0.015
        if CFG.use_gpu:
            prefs = bpy.context.preferences.addons.get("cycles")
            if prefs:
                cp = prefs.preferences
                for ct in ["OPTIX", "CUDA", "HIP", "METAL"]:
                    try:
                        cp.compute_device_type = ct; cp.get_devices()
                        for d in cp.devices: d.use = True
                        scene.cycles.device = 'GPU'
                        print(f"  GPU: {ct}"); break
                    except: continue
                else:
                    print("  No GPU, using CPU"); scene.cycles.device = 'CPU'
    elif "EEVEE" in CFG.render_engine:
        if hasattr(scene, 'eevee'):
            scene.eevee.taa_render_samples = CFG.render_samples

    scene.render.fps = CFG.fps


# ==================================================================
#  MAIN RENDER
# ==================================================================

def clone_stream(frames, meta):
    frames_out = []
    for fr in frames:
        frames_out.append({
            "vertices": fr["vertices"].copy(),
            "triangles": fr["triangles"].copy(),
            "labels": fr["labels"].copy(),
        })
    meta_out = {
        "n_frames": np.array([len(frames_out)], dtype=np.int32),
        "bbox_min": np.asarray(meta["bbox_min"], dtype=np.float64).copy(),
        "bbox_max": np.asarray(meta["bbox_max"], dtype=np.float64).copy(),
        "unique_labels": np.asarray(meta["unique_labels"], dtype=np.int32).copy(),
    }
    return frames_out, meta_out


def render_all(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading fish 1..."); frames1, meta1 = load_frames(input_dir, "fish1", allow_missing=True)
    print("Loading fish 2..."); frames2, meta2 = load_frames(input_dir, "fish2", allow_missing=True)
    if not frames1 and not frames2:
        raise FileNotFoundError(
            f"No preprocessed data found in {input_dir}. Expected fish1/fish2 frame_*.npz files."
        )
    if not frames1 or not frames2:
        if CFG.allow_missing_stream:
            if not frames1:
                print("  [fish1] missing: duplicating fish2 (--allow-missing-stream enabled).")
                frames1, meta1 = clone_stream(frames2, meta2)
            if not frames2:
                print("  [fish2] missing: duplicating fish1 (--allow-missing-stream enabled).")
                frames2, meta2 = clone_stream(frames1, meta1)
        else:
            missing = "fish1" if not frames1 else "fish2"
            raise FileNotFoundError(
                f"Missing required stream '{missing}' in {input_dir}. "
                "Run preprocess.py for both simulations, or pass --allow-missing-stream."
            )

    n = min(len(frames1), len(frames2))
    if len(frames1) != len(frames2):
        print(f"  Frame-count mismatch: fish1={len(frames1)} fish2={len(frames2)}; truncating to {n}")
    print(f"Rendering {n} frames")
    if n == 0:
        return 0

    # ── Y centering ───────────────────────────────────────────
    y_c1 = robust_center_y(frames1)
    y_c2 = robust_center_y(frames2)

    if CFG.fish_y_gap <= 0:
        y_ext = max(robust_y_span(frames1), robust_y_span(frames2))
        CFG.fish_y_gap = y_ext * 1.35

    y_line1 = CFG.fish_y_gap / 2.0
    y_line2 = -CFG.fish_y_gap / 2.0
    y_off1 = y_line1 - y_c1
    y_off2 = y_line2 - y_c2
    print(f"  Lane gap: {CFG.fish_y_gap:.4f}")
    print(f"  Y offsets: fish1={y_off1:+.4f}  fish2={y_off2:+.4f}")

    coverage = compute_frame_coverage(frames1, frames2, y_off1, y_off2, n)
    track_z = (coverage["z_min"] + coverage["z_max"]) * 0.5
    z_checker = coverage["z_min"] - max(0.05, CFG.checker_z_offset * coverage["z_span"])
    z_line = coverage["z_max"] + max(0.02, coverage["z_span"] * 0.01)

    if abs(y_line1 - y_line2) < 1e-6:
        y_line2 -= max(0.05, coverage["ortho_scale"] * 0.03)

    aspect = CFG.resolution[0] / CFG.resolution[1]
    if CFG.camera_follow:
        x_track = smooth_track(coverage["frame_centers_x"], CFG.camera_track_alpha)
        y_track = smooth_track(coverage["frame_centers_y"], CFG.camera_track_alpha)
        camera_mode = "follow"
    else:
        x_fixed = 0.5 * (coverage["x_min"] + coverage["x_max"])
        y_fixed = 0.5 * (y_line1 + y_line2)
        x_track = [x_fixed] * n
        y_track = [y_fixed] * n
        camera_mode = "fixed"

    camera_ortho = max(
        coverage["ortho_scale"],
        compute_required_ortho_scale(
            coverage, x_track, y_track, aspect, CFG.ortho_padding
        ),
    )

    cam_half_x = 0.5 * camera_ortho
    cam_half_y = cam_half_x / aspect
    cover = max(CFG.checker_cover_scale, 1.0)
    bg_half_x = cam_half_x * cover
    bg_half_y = cam_half_y * cover
    bg_x_min = min(x_track) - bg_half_x
    bg_x_max = max(x_track) + bg_half_x
    bg_y_min = min(y_track) - bg_half_y
    bg_y_max = max(y_track) + bg_half_y

    print(f"  Camera mode: {camera_mode}")
    print(f"  Ortho scale: {camera_ortho:.3f}")
    print(f"  Trajectory X: {coverage['x_min']:.3f} → {coverage['x_max']:.3f}")
    print(f"  Smoothing: domain={CFG.color_domain}  iters={CFG.color_blend_iters}  "
          f"self_w={CFG.color_blend_self_weight}  boundary_w={CFG.boundary_weight}")
    print(f"  Mesh shading: {'smooth' if CFG.smooth_shading else 'flat'}")
    print(f"  Background: {'checkerboard' if CFG.checkerboard_enable else 'black'}")

    # ── Scene ─────────────────────────────────────────────────
    clear_scene()
    setup_world()
    setup_render()
    rig, _cam, cam_height = setup_camera(
        x_track[0], y_track[0], track_z, camera_ortho, coverage["z_span"],
    )
    setup_lights(rig, camera_ortho, cam_height)
    create_checkerboard_background(
        bg_x_min, bg_x_max,
        bg_y_min, bg_y_max,
        z_checker,
    )

    create_reference_line(coverage["x_min"], coverage["x_max"], y_line1, z_line)
    create_reference_line(coverage["x_min"], coverage["x_max"], y_line2, z_line)

    mats1 = build_materials(CFG.theme_fish1, "fish1")
    mats2 = build_materials(CFG.theme_fish2, "fish2")

    f1 = frames1[0]; f2 = frames2[0]
    v1 = f1["vertices"].copy(); v1[:, 1] += y_off1
    v2 = f2["vertices"].copy(); v2[:, 1] += y_off2

    obj1 = mesh_from_arrays("Fish1", v1, f1["triangles"])
    ensure_outward_normals(obj1)
    assign_materials(obj1, f1["labels"], mats1)
    apply_surface_shading(obj1)

    obj2 = mesh_from_arrays("Fish2", v2, f2["triangles"])
    ensure_outward_normals(obj2)
    assign_materials(obj2, f2["labels"], mats2)
    apply_surface_shading(obj2)

    # ── Render loop ───────────────────────────────────────────
    print(f"\n{CFG.resolution[0]}×{CFG.resolution[1]}  samples={CFG.render_samples}  engine={CFG.render_engine}\n")

    for fi in range(n):
        path = os.path.join(output_dir, f"frame_{fi:05d}.png")

        if fi > 0:
            f1 = frames1[fi]; f2 = frames2[fi]
            v1 = f1["vertices"].copy(); v1[:, 1] += y_off1
            v2 = f2["vertices"].copy(); v2[:, 1] += y_off2

            update_mesh(obj1, v1, f1["triangles"])
            ensure_outward_normals(obj1)
            assign_materials(obj1, f1["labels"], mats1)
            apply_surface_shading(obj1)
            update_mesh(obj2, v2, f2["triangles"])
            ensure_outward_normals(obj2)
            assign_materials(obj2, f2["labels"], mats2)
            apply_surface_shading(obj2)

        rig.location.x = x_track[fi]
        rig.location.y = y_track[fi]
        rig.location.z = track_z

        bpy.context.scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print(f"  {fi+1:5d}/{n} → {os.path.basename(path)}")

    print(f"\n{n} frames rendered to {output_dir}")
    return n


def make_video(output_dir, n):
    vp = os.path.join(output_dir, "animation.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(CFG.fps),
        "-i", os.path.join(output_dir, "frame_%05d.png"),
        "-c:v", CFG.video_codec, "-crf", str(CFG.video_crf),
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        vp,
    ]
    print(f"\nStitching → {vp}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0: print(f"Video: {vp}")
    else: print(f"ffmpeg error:\n{r.stderr}")
    return vp


# ==================================================================
#  CLI
# ==================================================================

def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(
        description="Headless Blender renderer for two-fish FEM comparison."
    )
    ap.add_argument("--input", required=True, help="Directory with fish1/ fish2/ preprocessed frames")
    ap.add_argument("--output", required=True, help="Output directory for rendered frames")
    ap.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
    ap.add_argument("--samples", type=int, default=None, help="Render samples (16=test, 128+=final)")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--engine", choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"], default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--y-gap", type=float, default=None, help="Y gap between fish lanes")
    ap.add_argument("--allow-missing-stream", action="store_true")

    # ── Smoothing controls ────────────────────────────────────
    smooth = ap.add_argument_group("smoothing", "Color smoothing and shading controls")
    smooth.add_argument("--blend-iters", type=int, default=None,
                        help=f"Laplacian diffusion iterations (default: {CFG.color_blend_iters}). "
                             "0 disables smoothing entirely.")
    smooth.add_argument("--blend-self-weight", type=float, default=None,
                        help=f"Self-retention weight during diffusion (default: {CFG.color_blend_self_weight}). "
                             "Higher = vertex keeps more of its own color.")
    smooth.add_argument("--boundary-weight", type=float, default=None,
                        help=f"Cross-label edge weight (default: {CFG.boundary_weight}). "
                             "0.0 = hard boundary (no bleed across labels). "
                             "1.0 = ignore labels (full diffusion everywhere).")
    smooth.add_argument("--color-domain", choices=["vertex", "face"], default=None,
                        help=f"Color attribute domain (default: {CFG.color_domain}). "
                             "'vertex' = smooth interpolated gradients. "
                             "'face' = flat per-triangle colors (no smoothing).")
    smooth.add_argument("--smooth-shading", action="store_true", default=None,
                        help="Enable smooth mesh shading (default: on)")
    smooth.add_argument("--flat-shading", action="store_true",
                        help="Force flat mesh shading")
    smooth.add_argument("--auto-smooth-angle", type=float, default=None,
                        help=f"Auto-smooth angle in degrees (default: {CFG.auto_smooth_angle})")

    args = ap.parse_args(argv)

    # Apply to settings
    if args.resolution is not None: CFG.resolution = tuple(args.resolution)
    if args.samples is not None:    CFG.render_samples = args.samples
    if args.fps is not None:        CFG.fps = args.fps
    if args.engine is not None:     CFG.render_engine = args.engine
    if args.y_gap is not None:      CFG.fish_y_gap = args.y_gap
    CFG.allow_missing_stream = bool(args.allow_missing_stream)

    # Smoothing
    if args.blend_iters is not None:      CFG.color_blend_iters = args.blend_iters
    if args.blend_self_weight is not None: CFG.color_blend_self_weight = args.blend_self_weight
    if args.boundary_weight is not None:   CFG.boundary_weight = args.boundary_weight
    if args.color_domain is not None:      CFG.color_domain = args.color_domain
    if args.auto_smooth_angle is not None: CFG.auto_smooth_angle = args.auto_smooth_angle
    if args.flat_shading:
        CFG.smooth_shading = False
    elif args.smooth_shading:
        CFG.smooth_shading = True

    return args


if __name__ == "__main__":
    try:
        args = parse_args()
        print("=" * 50)
        print(f"  Input:  {args.input}")
        print(f"  Output: {args.output}")
        print(f"  {CFG.resolution[0]}×{CFG.resolution[1]}  samples={CFG.render_samples}")
        print(f"  Smoothing: domain={CFG.color_domain}  iters={CFG.color_blend_iters}  "
              f"self_w={CFG.color_blend_self_weight}  boundary_w={CFG.boundary_weight}")
        print(f"  Shading: {'smooth' if CFG.smooth_shading else 'flat'} "
              f"(auto-smooth angle={CFG.auto_smooth_angle}°)")
        print("=" * 50)
        n = render_all(args.input, args.output)
        if not args.no_video and n > 0:
            make_video(args.output, n)
        print("\nDone!")
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
