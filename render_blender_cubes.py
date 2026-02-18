#!/usr/bin/env python3
"""
render_blender_cubes.py — Headless Blender renderer for two jumping/flipping cubes.
Angled orthographic camera, bright opaque checkerboard ground, reduced color smoothing.

Usage:
    blender --background --python render_blender_cubes.py -- \
        --input preprocessed/ --output renders/ \
        --resolution 480 270 --samples 16

    blender --background --python render_blender_cubes.py -- \
        --input preprocessed/ --output renders/ \
        --resolution 1920 1080 --samples 128
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

    # ── Camera (orthographic at a jump-friendly angle) ────────
    ortho_padding       = 1.20
    camera_follow       = False
    camera_track_alpha  = 0.75
    camera_elevation_deg = 18.0   # lower camera for stronger low-angle look
    camera_azimuth_deg   = 224.0
    camera_distance_min  = 9.0
    camera_distance_mult = 2.5

    # ── Background ────────────────────────────────────────────
    world_color      = (0.0, 0.0, 0.0)
    world_strength   = 0.0
    checkerboard_enable = True
    checker_tiles_x  = 14
    checker_margin   = 0.25
    checker_cover_scale = 1.45
    checker_color_a  = (0.0, 0.0, 0.0)
    checker_color_b  = (0.10, 0.10, 0.10)
    checker_roughness = 0.92
    checker_specular  = 0.06
    checker_z_offset = 0.05

    # ── Lighting ──────────────────────────────────────────────
    # Three-point area rig (parented to camera rig).
    key_energy       = 430.0
    key_color        = (1.0, 0.97, 0.94)
    key_size_factor  = 0.55

    fill_energy      = 220.0
    fill_color       = (0.90, 0.95, 1.0)
    fill_size_factor = 0.50

    rim_energy       = 180.0
    rim_color        = (0.96, 0.98, 1.0)
    rim_size_factor  = 0.45

    light_spread     = 1.12
    light_z_factor   = 0.72

    # ── Cube layout ───────────────────────────────────────────
    cube_y_gap      = 0.0       # 0 = auto from stream extent

    # ── Color themes ──────────────────────────────────────────
    # label → (R, G, B) in 0–1.
    # Shared palette between both cubes for direct material comparison.
    theme_cube1 = {
        "name": "StudioBlue",
        "default": (0.064, 0.176, 0.60),
        0: (0.00, 0.00, 0.40),              # body (label 0)
        1: (0.45, 0.08, 0.10),
        2: (0.08, 0.32, 0.28),
        3: (0.40, 0.28, 0.04),
        4: (0.10, 0.18, 0.45),
        5: (0.18, 0.34, 0.08),
        6: (0.26, 0.10, 0.38),
        7: (0.42, 0.18, 0.06),
        8: (0.08, 0.30, 0.18),
        9: (0.38, 0.08, 0.24),
    }

    theme_cube2 = {
        "name": "StudioBlue",
        "default": (0.064, 0.176, 0.60),
        0: (0.00, 0.00, 0.40),              # body (label 0)
        1: (0.45, 0.08, 0.10),
        2: (0.08, 0.32, 0.28),
        3: (0.40, 0.28, 0.04),
        4: (0.10, 0.18, 0.45),
        5: (0.18, 0.34, 0.08),
        6: (0.26, 0.10, 0.38),
        7: (0.42, 0.18, 0.06),
        8: (0.08, 0.30, 0.18),
        9: (0.38, 0.08, 0.24),
    }

    # ── Material ──────────────────────────────────────────────
    roughness       = 0.85
    specular        = 0.05
    metallic        = 0.0
    emission_body   = 0.005
    emission_label  = 0.035
    color_attribute_name = "LabelColor"
    # Keep smoothing but weaker than fish setup to preserve material changes.
    color_blend_iters    = 1
    color_blend_self_weight = 6.0
    auto_smooth_angle_deg = 30.0
    # Kept for compatibility with helper utilities not used in cube scene.
    line_color      = (0.80, 0.86, 0.95)
    line_thickness  = 0.004
    line_extend     = 0.24
    line_emission   = 0.0

    # ── Color management ──────────────────────────────────────
    exposure        = 0.08
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


def parse_frame_npz(npz_obj, fp, stream_name):
    required = ("vertices", "triangles", "labels")
    missing = [k for k in required if k not in npz_obj]
    if missing:
        raise ValueError(f"[{stream_name}] {fp} missing keys: {missing}")

    vertices = np.asarray(npz_obj["vertices"], dtype=np.float32)
    triangles = np.asarray(npz_obj["triangles"], dtype=np.int32)
    labels = np.asarray(npz_obj["labels"], dtype=np.int32).reshape(-1)

    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError(f"[{stream_name}] invalid vertices in {fp}: {vertices.shape}")
    if vertices.shape[1] > 3:
        vertices = vertices[:, :3]

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"[{stream_name}] invalid triangles in {fp}: {triangles.shape}")
    if len(triangles) == 0:
        raise ValueError(f"[{stream_name}] empty triangles in {fp}")

    if len(labels) != len(triangles):
        if len(labels) == 1:
            labels = np.full((len(triangles),), int(labels[0]), dtype=np.int32)
        else:
            raise ValueError(
                f"[{stream_name}] label/triangle mismatch in {fp}: "
                f"{len(labels)} labels vs {len(triangles)} triangles"
            )

    return {"vertices": vertices, "triangles": triangles, "labels": labels}


def load_frames(input_dir, stream_name, allow_missing=False):
    stream_dir = os.path.join(input_dir, stream_name)
    if not os.path.isdir(stream_dir):
        if allow_missing:
            return [], None
        raise FileNotFoundError(f"[{stream_name}] directory not found: {stream_dir}")

    frame_paths = sorted(glob.glob(os.path.join(stream_dir, "frame_*.npz")))
    if not frame_paths:
        if allow_missing:
            return [], None
        raise FileNotFoundError(f"[{stream_name}] no frame_*.npz files in {stream_dir}")

    frames = []
    for fp in frame_paths:
        try:
            with np.load(fp, allow_pickle=False) as d:
                frames.append(parse_frame_npz(d, fp, stream_name))
        except Exception as exc:
            print(f"  [{stream_name}] skipping invalid frame {os.path.basename(fp)} ({exc})")

    if not frames:
        if allow_missing:
            return [], None
        raise ValueError(f"[{stream_name}] no valid frame data in {stream_dir}")

    meta_path = os.path.join(stream_dir, "metadata.npz")
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
            print(f"  [{stream_name}] invalid metadata.npz ({exc}); rebuilding.")
            meta = save_metadata(meta_path, frames)
    else:
        print(f"  [{stream_name}] metadata.npz missing; rebuilding from frame files.")
        meta = save_metadata(meta_path, frames)

    return frames, meta


def set_node_input(node, name_or_names, value):
    names = (name_or_names,) if isinstance(name_or_names, str) else tuple(name_or_names)
    for nm in names:
        if nm in node.inputs:
            node.inputs[nm].default_value = value
            return True
    return False


def create_material(name, emission_strength):
    """Surface material driven by per-vertex label colors."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear defaults
    for n in list(nodes): nodes.remove(n)

    # Principled + emission, both fed by vertex-color attribute.
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


def blend_vertex_colors(vertex_colors, triangles, iters, self_weight):
    if iters <= 0 or len(vertex_colors) == 0:
        return vertex_colors
    edges = triangles_to_edges(triangles)
    if len(edges) == 0:
        return vertex_colors

    cur = vertex_colors.astype(np.float32, copy=True)
    sw = float(max(self_weight, 0.0))
    for _ in range(int(iters)):
        nb_sum = np.zeros_like(cur)
        deg = np.zeros((len(cur), 1), dtype=np.float32)

        np.add.at(nb_sum, edges[:, 0], cur[edges[:, 1]])
        np.add.at(nb_sum, edges[:, 1], cur[edges[:, 0]])
        np.add.at(deg[:, 0], edges[:, 0], 1.0)
        np.add.at(deg[:, 0], edges[:, 1], 1.0)

        nb_avg = nb_sum / np.maximum(deg, 1.0)
        cur = (sw * cur + nb_avg) / (sw + 1.0)

    return np.clip(cur, 0.0, 1.0)


def set_mesh_color_attribute(mesh, vertex_colors, attr_name):
    if not hasattr(mesh, "color_attributes"):
        return

    attrs = mesh.color_attributes
    attr = attrs.get(attr_name)
    make_new = attr is None
    if attr is not None:
        try:
            make_new = (attr.domain != "CORNER") or (getattr(attr, "data_type", "FLOAT_COLOR") != "FLOAT_COLOR")
        except Exception:
            make_new = False
    if make_new:
        if attr is not None:
            attrs.remove(attr)
        attr = attrs.new(name=attr_name, type="FLOAT_COLOR", domain="CORNER")

    n_loops = len(mesh.loops)
    if n_loops == 0:
        return
    loop_vidx = np.empty(n_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vidx)

    rgba = np.empty((n_loops, 4), dtype=np.float32)
    rgba[:, :3] = vertex_colors[loop_vidx]
    rgba[:, 3] = 1.0
    attr.data.foreach_set("color", rgba.reshape(-1))


def assign_materials(obj, labels, materials_dict):
    mesh = obj.data
    if len(labels) != len(mesh.polygons):
        raise ValueError(
            f"{obj.name}: {len(labels)} labels for {len(mesh.polygons)} polygons"
        )

    theme = materials_dict["theme"]
    surface_mat = materials_dict["surface"]
    if len(obj.data.materials) == 0 or obj.data.materials[0] is not surface_mat:
        obj.data.materials.clear()
        obj.data.materials.append(surface_mat)

    triangles = np.array([poly.vertices[:] for poly in mesh.polygons], dtype=np.int64)
    face_colors = labels_to_face_colors(labels, theme)

    n_vertices = len(mesh.vertices)
    vcols = np.zeros((n_vertices, 3), dtype=np.float32)
    counts = np.zeros((n_vertices, 1), dtype=np.float32)
    for k in range(3):
        idx = triangles[:, k]
        np.add.at(vcols, idx, face_colors)
        np.add.at(counts[:, 0], idx, 1.0)
    vcols = vcols / np.maximum(counts, 1.0)
    vcols = blend_vertex_colors(vcols, triangles, CFG.color_blend_iters, CFG.color_blend_self_weight)
    set_mesh_color_attribute(mesh, vcols, CFG.color_attribute_name)


def update_mesh(obj, vertices, triangles):
    old = obj.data
    new = bpy.data.meshes.new(obj.name + "_tmp")
    new.from_pydata(vertices.tolist(), [], triangles.tolist())
    new.update()
    obj.data = new
    bpy.data.meshes.remove(old)


def ensure_outward_normals(obj):
    """Fallback for preprocessed data with inverted winding."""
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
    """Smooth shading with restrained auto-smooth to keep cube edges readable."""
    for poly in obj.data.polygons:
        poly.use_smooth = True
    if hasattr(obj.data, "use_auto_smooth"):
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = math.radians(CFG.auto_smooth_angle_deg)


def smooth_track(values, alpha):
    """Exponential smoothing to reduce camera jitter."""
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
    """Frame-wise bounds for stable camera scale and tracking."""
    frame_centers_x = []
    frame_centers_y = []
    frame_x_min = []
    frame_x_max = []
    frame_y_min = []
    frame_y_max = []
    frame_z_min = []
    frame_z_max = []
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
        frame_z_min.append(float(fmin[2]))
        frame_z_max.append(float(fmax[2]))
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
        "frame_z_min": frame_z_min,
        "frame_z_max": frame_z_max,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
        "z_span": max(z_max - z_min, 1e-3),
        "ortho_scale": ortho_scale,
    }


def camera_basis(elevation_deg, azimuth_deg):
    """Return camera placement vector and projection basis for a fixed angle."""
    el = math.radians(float(elevation_deg))
    az = math.radians(float(azimuth_deg))

    to_camera = np.array([
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el),
    ], dtype=np.float64)
    to_camera /= max(np.linalg.norm(to_camera), 1e-12)

    # Camera looks back toward the rig center.
    forward = -to_camera
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= max(np.linalg.norm(up), 1e-12)

    return to_camera, right, up


def compute_required_ortho_scale_angled(coverage, x_track, y_track, track_z, right, up, aspect, padding):
    """Width-style ortho scale needed for an angled orthographic camera."""
    if aspect <= 0:
        aspect = 1.0
    pad = max(float(padding), 1.0)
    required = 1e-3
    for fi in range(len(x_track)):
        cx, cy = float(x_track[fi]), float(y_track[fi])
        corners = (
            (coverage["frame_x_min"][fi], coverage["frame_y_min"][fi], coverage["frame_z_min"][fi]),
            (coverage["frame_x_min"][fi], coverage["frame_y_min"][fi], coverage["frame_z_max"][fi]),
            (coverage["frame_x_min"][fi], coverage["frame_y_max"][fi], coverage["frame_z_min"][fi]),
            (coverage["frame_x_min"][fi], coverage["frame_y_max"][fi], coverage["frame_z_max"][fi]),
            (coverage["frame_x_max"][fi], coverage["frame_y_min"][fi], coverage["frame_z_min"][fi]),
            (coverage["frame_x_max"][fi], coverage["frame_y_min"][fi], coverage["frame_z_max"][fi]),
            (coverage["frame_x_max"][fi], coverage["frame_y_max"][fi], coverage["frame_z_min"][fi]),
            (coverage["frame_x_max"][fi], coverage["frame_y_max"][fi], coverage["frame_z_max"][fi]),
        )
        max_abs_x = 0.0
        max_abs_y = 0.0
        for xw, yw, zw in corners:
            dv = np.array([xw - cx, yw - cy, zw - track_z], dtype=np.float64)
            px = abs(float(np.dot(dv, right)))
            py = abs(float(np.dot(dv, up)))
            max_abs_x = max(max_abs_x, px)
            max_abs_y = max(max_abs_y, py)
        required = max(required, max(2.0 * max_abs_x, 2.0 * max_abs_y * aspect))
    return max(required * pad, 1e-3)


# ==================================================================
#  SCENE SETUP
# ==================================================================

def setup_world():
    """Neutral bright world, with most spatial cue coming from the checker ground."""
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


def setup_camera(track_x0, track_y, track_z, ortho_scale, scene_span, to_camera):
    """Angled orthographic camera parented to a tracking rig."""
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

    cam_distance = max(
        CFG.camera_distance_min,
        scene_span * CFG.camera_distance_mult,
        ortho_scale * 1.75,
    )
    cam_obj.location = Vector(tuple(to_camera * cam_distance))
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)

    track = cam_obj.constraints.new(type='TRACK_TO')
    track.target = rig
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    cam_data.clip_start = 0.01
    cam_data.clip_end = max(1000.0, cam_distance * 80.0)

    return rig, cam_obj, cam_distance


def setup_lights(rig, ortho_scale, cam_distance):
    """Three-point area-light rig that follows the camera rig."""
    spread = max(ortho_scale * CFG.light_spread, 3.0)
    base_size = max(ortho_scale * 0.75, 1.5)
    light_z = max(cam_distance * CFG.light_z_factor, spread * 1.2)
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

    add_light(
        "Key", CFG.key_energy, CFG.key_color, CFG.key_size_factor,
        (-spread * 0.40, -spread * 0.40, 0.0),
    )
    add_light(
        "Fill", CFG.fill_energy, CFG.fill_color, CFG.fill_size_factor,
        (spread * 0.42, spread * 0.12, -spread * 0.12),
    )
    add_light(
        "Rim", CFG.rim_energy, CFG.rim_color, CFG.rim_size_factor,
        (0.0, spread * 0.52, -spread * 0.18),
    )


def create_reference_line(x_min, x_max, y_pos, z_pos):
    """Thin emissive cylinder along X."""
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
    set_node_input(em, "Color", (*CFG.line_color, 1.0))
    set_node_input(em, "Strength", CFG.line_emission)
    links.new(em.outputs["Emission"], out.inputs["Surface"])
    line.data.materials.append(mat)
    return line


def create_checkerboard_background(x_min, x_max, y_min, y_max, z_pos):
    """Opaque checkerboard ground plane in bright sober tones."""
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
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    out = nodes.new("ShaderNodeOutputMaterial")

    tile_size = max(span_x / max(CFG.checker_tiles_x, 2), 1e-3)
    set_node_input(mapping, "Scale", (ext_x / tile_size, ext_y / tile_size, 1.0))
    set_node_input(checker, "Color1", (*CFG.checker_color_a, 1.0))
    set_node_input(checker, "Color2", (*CFG.checker_color_b, 1.0))
    set_node_input(checker, "Scale", 1.0)
    set_node_input(bsdf, "Roughness", CFG.checker_roughness)
    set_node_input(bsdf, ["Specular IOR Level", "Specular"], CFG.checker_specular)
    set_node_input(bsdf, "Metallic", 0.0)

    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

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

    # Color management with gentle contrast for presentation frames.
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


def render_all(input_dir, output_dir, stream1="cube1", stream2="cube2"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {stream1}..."); frames1, meta1 = load_frames(input_dir, stream1, allow_missing=True)
    print(f"Loading {stream2}..."); frames2, meta2 = load_frames(input_dir, stream2, allow_missing=True)
    loaded1, loaded2 = stream1, stream2

    # Compatibility fallback if the user reuses preprocess.py (fish1/fish2 folders).
    if not frames1 and not frames2 and (stream1, stream2) == ("cube1", "cube2"):
        print("  cube1/cube2 not found. Trying fish1/fish2 for compatibility.")
        frames1, meta1 = load_frames(input_dir, "fish1", allow_missing=True)
        frames2, meta2 = load_frames(input_dir, "fish2", allow_missing=True)
        loaded1, loaded2 = "fish1", "fish2"

    if not frames1 and not frames2:
        raise FileNotFoundError(
            f"No preprocessed data found in {input_dir}. "
            f"Expected {stream1}/{stream2} (or fish1/fish2 for compatibility) frame_*.npz files."
        )
    if not frames1 or not frames2:
        if CFG.allow_missing_stream:
            if not frames1:
                print(f"  [{loaded1}] missing: duplicating {loaded2} (--allow-missing-stream enabled).")
                frames1, meta1 = clone_stream(frames2, meta2)
            if not frames2:
                print(f"  [{loaded2}] missing: duplicating {loaded1} (--allow-missing-stream enabled).")
                frames2, meta2 = clone_stream(frames1, meta1)
        else:
            missing = loaded1 if not frames1 else loaded2
            raise FileNotFoundError(
                f"Missing required stream '{missing}' in {input_dir}. "
                "Run preprocess_cubes_hex.py (or preprocess_cubes.py / preprocess.py), "
                "or pass --allow-missing-stream."
            )

    n = min(len(frames1), len(frames2))
    if len(frames1) != len(frames2):
        print(f"  Frame-count mismatch: {loaded1}={len(frames1)} {loaded2}={len(frames2)}; truncating to {n}")
    print(f"Rendering {n} frames")
    if n == 0:
        return 0

    # ── Y centering ───────────────────────────────────────────
    y_c1 = robust_center_y(frames1)
    y_c2 = robust_center_y(frames2)

    if CFG.cube_y_gap <= 0:
        y_ext = max(robust_y_span(frames1), robust_y_span(frames2))
        CFG.cube_y_gap = y_ext * 1.45

    y_line1 = CFG.cube_y_gap / 2.0
    y_line2 = -CFG.cube_y_gap / 2.0
    y_off1 = y_line1 - y_c1
    y_off2 = y_line2 - y_c2
    print(f"  Lane gap: {CFG.cube_y_gap:.4f}")
    print(f"  Y offsets: cube1={y_off1:+.4f}  cube2={y_off2:+.4f}")

    coverage = compute_frame_coverage(frames1, frames2, y_off1, y_off2, n)
    track_z = (coverage["z_min"] + coverage["z_max"]) * 0.5
    z_ground = coverage["z_min"] - max(0.02, CFG.checker_z_offset * coverage["z_span"])

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

    to_camera, cam_right, cam_up = camera_basis(
        CFG.camera_elevation_deg, CFG.camera_azimuth_deg
    )
    camera_ortho = max(
        coverage["ortho_scale"],
        compute_required_ortho_scale_angled(
            coverage, x_track, y_track, track_z, cam_right, cam_up, aspect, CFG.ortho_padding
        ),
    )

    span_xy = max(
        coverage["x_max"] - coverage["x_min"],
        coverage["y_max"] - coverage["y_min"],
        camera_ortho,
        1e-3,
    )
    ground_pad = span_xy * max(CFG.checker_cover_scale, 1.0)
    bg_x_min = min(x_track) - ground_pad
    bg_x_max = max(x_track) + ground_pad
    bg_y_min = min(y_track) - ground_pad
    bg_y_max = max(y_track) + ground_pad

    print(f"  Camera mode: {camera_mode}")
    print(f"  Camera angle: elevation={CFG.camera_elevation_deg:.1f}° azimuth={CFG.camera_azimuth_deg:.1f}°")
    print(f"  Ortho scale: {camera_ortho:.3f}")
    print(f"  Trajectory X: {coverage['x_min']:.3f} → {coverage['x_max']:.3f}")
    print(f"  Color blending: iters={CFG.color_blend_iters} self_weight={CFG.color_blend_self_weight}")
    print(f"  Ground: {'checkerboard' if CFG.checkerboard_enable else 'none'}")

    # ── Scene ─────────────────────────────────────────────────
    clear_scene()
    setup_world()
    setup_render()
    scene_span = max(
        coverage["x_max"] - coverage["x_min"],
        coverage["y_max"] - coverage["y_min"],
        coverage["z_span"],
        1e-3,
    )
    rig, _cam, cam_distance = setup_camera(
        x_track[0], y_track[0], track_z, camera_ortho, scene_span, to_camera,
    )
    setup_lights(rig, camera_ortho, cam_distance)
    create_checkerboard_background(
        bg_x_min, bg_x_max,
        bg_y_min, bg_y_max,
        z_ground,
    )

    # Materials
    mats1 = build_materials(CFG.theme_cube1, "cube1")
    mats2 = build_materials(CFG.theme_cube2, "cube2")

    # Initial meshes
    f1 = frames1[0]; f2 = frames2[0]
    v1 = f1["vertices"].copy(); v1[:, 1] += y_off1
    v2 = f2["vertices"].copy(); v2[:, 1] += y_off2

    obj1 = mesh_from_arrays("Cube1", v1, f1["triangles"])
    ensure_outward_normals(obj1)
    assign_materials(obj1, f1["labels"], mats1)
    apply_surface_shading(obj1)

    obj2 = mesh_from_arrays("Cube2", v2, f2["triangles"])
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stream1", default="cube1")
    ap.add_argument("--stream2", default="cube2")
    ap.add_argument("--resolution", type=int, nargs=2, default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--engine", choices=["CYCLES","BLENDER_EEVEE_NEXT","BLENDER_EEVEE"], default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--y-gap", type=float, default=None)
    ap.add_argument("--camera-follow", action="store_true")
    ap.add_argument("--allow-missing-stream", action="store_true")
    args = ap.parse_args(argv)
    if args.resolution is not None: CFG.resolution = tuple(args.resolution)
    if args.samples is not None:    CFG.render_samples = args.samples
    if args.fps is not None:        CFG.fps = args.fps
    if args.engine is not None:     CFG.render_engine = args.engine
    if args.y_gap is not None:      CFG.cube_y_gap = args.y_gap
    CFG.camera_follow = bool(args.camera_follow)
    CFG.allow_missing_stream = bool(args.allow_missing_stream)
    return args


if __name__ == "__main__":
    try:
        args = parse_args()
        print("=" * 50)
        print(f"  Input:  {args.input}")
        print(f"  Output: {args.output}")
        print(f"  {CFG.resolution[0]}×{CFG.resolution[1]}  samples={CFG.render_samples}")
        print("=" * 50)
        n = render_all(args.input, args.output, stream1=args.stream1, stream2=args.stream2)
        if not args.no_video and n > 0:
            make_video(args.output, n)
        print("\nDone!")
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
