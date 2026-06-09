"""Resize a generated mesh and build a side-by-side scale-comparison GLB against a
human (or custom) reference. FBX references are converted via Blender."""
import os
import subprocess

import numpy as np
import trimesh

BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REF_FBX = r"C:\Users\reese\Downloads\animations\character models\mannequin.fbx"
DEFAULT_REF_GLB = os.path.join(HERE, "assets", "refs", "mannequin.glb")


def _as_mesh(path):
    g = trimesh.load(path)
    if isinstance(g, trimesh.Scene):
        if len(g.geometry) == 0:
            raise ValueError(f"no geometry in {path}")
        g = g.to_geometry()  # bakes scene-graph node transforms into the geometry
    return g


def fbx_to_glb(fbx_path, out_glb=None):
    if out_glb is None:
        out_glb = os.path.splitext(fbx_path)[0] + ".glb"
    if os.path.exists(out_glb) and os.path.getmtime(out_glb) >= os.path.getmtime(fbx_path):
        return out_glb
    os.makedirs(os.path.dirname(out_glb) or ".", exist_ok=True)
    subprocess.run([BLENDER, "--background", "--python", os.path.join(HERE, "tools", "fbx2glb.py"),
                    "--", fbx_path, out_glb], check=True, capture_output=True, timeout=300)
    return out_glb


def ensure_glb(path):
    """Return a GLB path for a reference given as .glb or .fbx (converting if needed)."""
    if path and path.lower().endswith(".fbx"):
        return fbx_to_glb(path)
    return path


def _rot_axis_to_yup(mesh, up):
    up = (up or "y").lower()
    if up == "y":
        return
    if up == "z":
        R = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    elif up == "x":
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    else:
        return
    mesh.apply_transform(R)


def _ground_center(m):
    b = m.bounds
    m.apply_translation([-(b[0][0] + b[1][0]) / 2.0, -b[0][1], -(b[0][2] + b[1][2]) / 2.0])


def build_scale_compare(asset_path, target_m, ref_path, ref_height_m, ref_up, out_path):
    """Scale `asset` so its longest axis == target_m, stand `ref` at ref_height_m,
    place them side by side, export a combined GLB. Returns (out_path, asset_dims_m)."""
    asset = _as_mesh(asset_path)
    aext = asset.bounding_box.extents
    if max(aext) > 0:
        asset.apply_scale(float(target_m) / float(max(aext)))

    ref = _as_mesh(ensure_glb(ref_path))
    _rot_axis_to_yup(ref, ref_up)
    rh = float(ref.bounding_box.extents[1])  # Y is up after rotation
    if rh > 0:
        ref.apply_scale(float(ref_height_m) / rh)

    _ground_center(asset)
    _ground_center(ref)
    aw = float(asset.bounding_box.extents[0])
    rw = float(ref.bounding_box.extents[0])
    gap = max(0.3, 0.15 * float(target_m))
    asset.apply_translation([-(aw / 2.0 + gap / 2.0), 0, 0])
    ref.apply_translation([(rw / 2.0 + gap / 2.0), 0, 0])

    scene = trimesh.Scene()
    scene.add_geometry(asset, geom_name="asset")
    scene.add_geometry(ref, geom_name="reference")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    scene.export(out_path)
    return out_path, [round(float(x), 3) for x in asset.bounding_box.extents]


def resize_export(asset_path, target_m, out_path):
    """Scale a mesh so its longest axis == target_m and export. Returns (out_path, dims)."""
    m = _as_mesh(asset_path)
    ext = m.bounding_box.extents
    if max(ext) > 0:
        m.apply_scale(float(target_m) / float(max(ext)))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    m.export(out_path)
    return out_path, [round(float(x), 3) for x in m.bounding_box.extents]
