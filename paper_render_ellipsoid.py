#!/usr/bin/env python3
"""
paper_render_ellipsoid.py — Render one preprocessed ellipsoid frame as PNG.

Run:
    blender --background --python paper_render_ellipsoid.py

All parameters are defined at the top of this file.
"""

import glob
import os
import sys

import bpy
import numpy as np

# Reuse the exact rendering pipeline/style from paper_render_swimmer_frames.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import paper_render_swimmer_frames as PR  # noqa: E402


# ==================================================================
#  SETTINGS (edit these)
# ==================================================================


class Settings:
    # ── Input / Output ────────────────────────────────────────────
    input_dir = "preprocessed/ellipsoid"   # folder with frame_*.npz
    stream_subdir = ""                     # optional, e.g. "fish1"
    frame_pattern = "frame_*.npz"
    frame_index = 0                         # supports negatives (-1 = last)
    output_png = "renders/ellipsoid.png"
    object_name = "Ellipsoid"
    center_mesh_xy = False

    # ── Resolution / engine ───────────────────────────────────────
    resolution = (1920, 1080)
    render_samples = 64
    render_engine = "CYCLES"               # "CYCLES" or "BLENDER_EEVEE_NEXT"
    use_gpu = True

    # ── Camera / framing ──────────────────────────────────────────
    ortho_padding = 1.10
    camera_height_min = 8.0
    camera_height_mult = 2.3

    # ── Background ────────────────────────────────────────────────
    world_color = (1.0, 1.0, 1.0)
    world_strength = 0.0

    # ── Grid ──────────────────────────────────────────────────────
    grid_enable = True
    grid_color = (0.55, 0.55, 0.55)
    grid_bg_color = (1.0, 1.0, 1.0)
    grid_emission = 1.0
    grid_line_width = 0.10
    grid_min_pixels = 5.0
    grid_spacing = 1.5
    grid_margin = 0.60
    grid_cover_scale = 1.30
    grid_z_offset = 0.10

    # ── Lighting ──────────────────────────────────────────────────
    key_energy = 520.0
    key_color = (1.0, 0.95, 0.90)
    key_size_factor = 0.55

    fill_energy = 130.0
    fill_color = (0.90, 0.93, 1.0)
    fill_size_factor = 0.50

    rim_energy = 120.0
    rim_color = (1.0, 0.97, 0.94)
    rim_size_factor = 0.45

    light_spread = 1.05
    light_z_factor = 0.86

    # ── Material / color ──────────────────────────────────────────
    roughness = 0.85
    specular = 0.05
    metallic = 0.0
    emission_body = 0.005
    emission_label = 0.035
    color_attribute_name = "LabelColor"

    # Keep same smoothing pipeline as swimmer script
    color_blend_iters = 1
    color_blend_self_weight = 10.0
    boundary_weight = 0.01
    color_domain = "vertex"                # "vertex" or "face"
    smooth_shading = True
    auto_smooth_angle = 48.0

    # ── Color management ──────────────────────────────────────────
    exposure = 0.0
    gamma = 1.0

    # ── Theme (same palette as paper_render_swimmer_frames.py fish1)
    theme = {
        "name": "StudioBlue",
        "default": (0.064, 0.176, 0.60),
        0: (0.72, 0.80, 0.88),
        1: (0.04, 0.04, 0.04),
        2: (0.15, 0.00, 0.00),
        3: (0.98, 0.66, 0.22),
        4: (0.31, 0.10, 0.38),
        5: (0.18, 0.34, 0.08),
        6: (0.08, 0.32, 0.28),
        7: (0.40, 0.28, 0.04),
        8: (0.08, 0.30, 0.18),
        9: (0.38, 0.08, 0.24),
        10: (0.42, 0.18, 0.06),
    }


CFG = Settings()


# ==================================================================
#  CONFIG + LOAD
# ==================================================================


def apply_settings_to_shared_renderer():
    s = PR.CFG

    s.resolution = tuple(CFG.resolution)
    s.render_samples = int(CFG.render_samples)
    s.render_engine = CFG.render_engine
    s.use_gpu = bool(CFG.use_gpu)

    s.ortho_padding = float(CFG.ortho_padding)
    s.camera_height_min = float(CFG.camera_height_min)
    s.camera_height_mult = float(CFG.camera_height_mult)

    s.world_color = tuple(CFG.world_color)
    s.world_strength = float(CFG.world_strength)

    s.grid_enable = bool(CFG.grid_enable)
    s.grid_color = tuple(CFG.grid_color)
    s.grid_bg_color = tuple(CFG.grid_bg_color)
    s.grid_emission = float(CFG.grid_emission)
    s.grid_line_width = float(CFG.grid_line_width)
    s.grid_min_pixels = float(CFG.grid_min_pixels)
    s.grid_spacing = float(CFG.grid_spacing) if CFG.grid_spacing is not None else None
    s.grid_margin = float(CFG.grid_margin)
    s.grid_cover_scale = float(CFG.grid_cover_scale)
    s.grid_z_offset = float(CFG.grid_z_offset)

    s.key_energy = float(CFG.key_energy)
    s.key_color = tuple(CFG.key_color)
    s.key_size_factor = float(CFG.key_size_factor)

    s.fill_energy = float(CFG.fill_energy)
    s.fill_color = tuple(CFG.fill_color)
    s.fill_size_factor = float(CFG.fill_size_factor)

    s.rim_energy = float(CFG.rim_energy)
    s.rim_color = tuple(CFG.rim_color)
    s.rim_size_factor = float(CFG.rim_size_factor)

    s.light_spread = float(CFG.light_spread)
    s.light_z_factor = float(CFG.light_z_factor)

    s.roughness = float(CFG.roughness)
    s.specular = float(CFG.specular)
    s.metallic = float(CFG.metallic)
    s.emission_body = float(CFG.emission_body)
    s.emission_label = float(CFG.emission_label)
    s.color_attribute_name = CFG.color_attribute_name

    s.color_blend_iters = int(CFG.color_blend_iters)
    s.color_blend_self_weight = float(CFG.color_blend_self_weight)
    s.boundary_weight = float(CFG.boundary_weight)
    s.color_domain = CFG.color_domain
    s.smooth_shading = bool(CFG.smooth_shading)
    s.auto_smooth_angle = float(CFG.auto_smooth_angle)

    s.exposure = float(CFG.exposure)
    s.gamma = float(CFG.gamma)


def resolve_frame_dir():
    frame_dir = os.path.abspath(CFG.input_dir)
    if CFG.stream_subdir:
        frame_dir = os.path.join(frame_dir, CFG.stream_subdir)
    return frame_dir


def load_single_frame():
    frame_dir = resolve_frame_dir()
    if not os.path.isdir(frame_dir):
        raise FileNotFoundError(f"Input directory not found: {frame_dir}")

    frame_paths = sorted(glob.glob(os.path.join(frame_dir, CFG.frame_pattern)))
    if not frame_paths:
        raise FileNotFoundError(
            f"No files matching '{CFG.frame_pattern}' in {frame_dir}"
        )

    idx = int(CFG.frame_index)
    if idx < 0:
        idx += len(frame_paths)
    if idx < 0 or idx >= len(frame_paths):
        raise IndexError(
            f"frame_index={CFG.frame_index} out of range [0, {len(frame_paths) - 1}]"
        )

    frame_path = frame_paths[idx]
    with np.load(frame_path, allow_pickle=False) as d:
        frame = PR.parse_frame_npz(d, frame_path, "ellipsoid")

    return frame, frame_path, len(frame_paths)


# ==================================================================
#  RENDER
# ==================================================================


def render_single_ellipsoid():
    apply_settings_to_shared_renderer()
    frame, frame_path, n_total = load_single_frame()

    verts = frame["vertices"].copy()
    tris = frame["triangles"]
    labels = frame["labels"]

    if CFG.center_mesh_xy:
        center_xy = 0.5 * (verts.min(axis=0)[:2] + verts.max(axis=0)[:2])
        verts[:, 0] -= center_xy[0]
        verts[:, 1] -= center_xy[1]

    single_stream = [[{"vertices": verts, "triangles": tris, "labels": labels}]]
    y_offsets = [0.0]

    coverage = PR.compute_frame_coverage(single_stream, y_offsets, 1)
    track_z = 0.5 * (coverage["z_min"] + coverage["z_max"])
    z_ground = coverage["z_min"] - max(0.05, PR.CFG.grid_z_offset * coverage["z_span"])

    aspect = PR.CFG.resolution[0] / PR.CFG.resolution[1]
    x_fixed = 0.5 * (coverage["x_min"] + coverage["x_max"])
    y_fixed = 0.5 * (coverage["y_min"] + coverage["y_max"])

    x_track = [x_fixed]
    y_track = [y_fixed]
    camera_ortho = max(
        coverage["ortho_scale"],
        PR.compute_required_ortho_scale(coverage, x_track, y_track, aspect, PR.CFG.ortho_padding),
    )

    cam_half_x = 0.5 * camera_ortho
    cam_half_y = cam_half_x / max(aspect, 1e-6)
    cover = max(PR.CFG.grid_cover_scale, 1.0)
    bg_x_min = x_fixed - cam_half_x * cover
    bg_x_max = x_fixed + cam_half_x * cover
    bg_y_min = y_fixed - cam_half_y * cover
    bg_y_max = y_fixed + cam_half_y * cover

    print("=" * 50)
    print(f"  Input dir: {resolve_frame_dir()}")
    print(f"  Frame:     {os.path.basename(frame_path)} ({CFG.frame_index} of {n_total})")
    print(f"  Output:    {os.path.abspath(CFG.output_png)}")
    print(f"  Res:       {PR.CFG.resolution[0]}x{PR.CFG.resolution[1]} samples={PR.CFG.render_samples}")
    print(f"  Smoothing: domain={PR.CFG.color_domain} iters={PR.CFG.color_blend_iters} "
          f"self_w={PR.CFG.color_blend_self_weight} boundary_w={PR.CFG.boundary_weight}")
    print(f"  Shading:   {'smooth' if PR.CFG.smooth_shading else 'flat'}")
    print("=" * 50)

    PR.clear_scene()
    PR.setup_world()
    PR.setup_render()
    rig, _cam, cam_height = PR.setup_camera(x_fixed, y_fixed, track_z, camera_ortho, coverage["z_span"])
    PR.setup_lights(rig, camera_ortho, cam_height)
    PR.create_grid_ground(bg_x_min, bg_x_max, bg_y_min, bg_y_max, z_ground, camera_ortho=camera_ortho)

    mats = PR.build_materials(CFG.theme, "ellipsoid")
    obj = PR.mesh_from_arrays(CFG.object_name, verts, tris)
    PR.ensure_outward_normals(obj)
    PR.assign_materials(obj, labels, mats)
    PR.apply_surface_shading(obj)

    rig.location.x = x_fixed
    rig.location.y = y_fixed
    rig.location.z = track_z

    out_path = os.path.abspath(CFG.output_png)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

    print(f"\nRendered image written to {out_path}")


if __name__ == "__main__":
    try:
        render_single_ellipsoid()
        print("Done")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise
