"""Render a finished textured mesh from several angles and grade it with a local
vision model. Returns a structured verdict used to approve or refine.

The repo's MeshRender.render() is incompatible with the custom_rasterizer ('cr')
backend, so we rasterize on the GPU and sample the texture with grid_sample
(no OpenGL/EGL needed -- works headless)."""
import numpy as np
from PIL import Image

from . import ollama_client as oc

_RENDER = None  # lazily-created MeshRender, reused across calls


def _get_render(resolution=512):
    global _RENDER
    if _RENDER is None:
        from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
        _RENDER = MeshRender(default_resolution=resolution, texture_size=1024,
                             camera_type='orth', device='cuda')
    return _RENDER


def _render_one(r, elev, azim):
    """GPU rasterize + texture-sample one view of the mesh loaded in MeshRender r."""
    import torch
    import torch.nn.functional as F
    from hy3dgen.texgen.differentiable_renderer.camera_utils import get_mv_matrix, transform_pos

    res = r.default_resolution
    mvp = np.matmul(r.camera_proj_mat,
                    get_mv_matrix(elev=elev, azim=azim, camera_distance=r.camera_distance)
                    ).astype(np.float32)
    pos_clip = transform_pos(mvp, r.vtx_pos)
    rast_out, _ = r.raster_rasterize(pos_clip, r.pos_idx, resolution=list(res))
    texc, _ = r.raster_interpolate(r.vtx_uv[None, ...], rast_out, r.uv_idx)
    mask = torch.clamp(rast_out[..., -1:], 0, 1)[0]
    grid = texc * 2 - 1
    tex_in = r.tex.permute(2, 0, 1)[None].float()
    color = F.grid_sample(tex_in, grid, mode='bilinear',
                          padding_mode='border', align_corners=False)[0].permute(1, 2, 0)
    out = color * mask + torch.ones_like(color) * (1 - mask)
    return (out.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def _to_pil(arr):
    a = np.asarray(arr)
    while a.ndim > 3:                       # drop leading batch dims
        a = a[0]
    if a.ndim == 3 and a.shape[-1] > 3:     # drop alpha if present
        a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.clip(a, 0.0, 1.0) * 255.0
        a = a.astype(np.uint8)
    return Image.fromarray(a)


def _extract_texture(mesh):
    """Pull the baked texture image (PIL) out of a trimesh, if present."""
    vis = getattr(mesh, "visual", None)
    if vis is None:
        return None
    mat = getattr(vis, "material", None)
    if mat is not None:
        for attr in ("baseColorTexture", "image"):
            img = getattr(mat, attr, None)
            if img is not None:
                return img
    return getattr(vis, "image", None)


def render_views(textured_mesh, n_views=4, elev=15, resolution=512):
    """Return a list of PIL images of the textured mesh from evenly-spaced azimuths."""
    r = _get_render(resolution)
    r.load_mesh(textured_mesh)
    # The repo's load_mesh never sets the texture; apply it ourselves.
    tex = _extract_texture(textured_mesh)
    if tex is not None:
        r.set_texture(tex)
    else:
        r.set_texture(np.ones((16, 16, 3), dtype=np.float32) * 0.6)  # neutral fallback
    imgs = []
    for i in range(n_views):
        azim = (360.0 / n_views) * i
        try:
            imgs.append(_to_pil(_render_one(r, elev, azim)))
        except Exception as e:
            print(f"[qa] render view {i} failed: {e}", flush=True)
    return imgs


def contact_sheet(images, cell=512):
    """Tile up to 4 images into a 2x2 sheet (one PIL image)."""
    if not images:
        return None
    imgs = [im.convert("RGB").resize((cell, cell)) for im in images[:4]]
    while len(imgs) < 4:
        imgs.append(Image.new("RGB", (cell, cell), (255, 255, 255)))
    sheet = Image.new("RGB", (cell * 2, cell * 2), (255, 255, 255))
    for idx, im in enumerate(imgs):
        sheet.paste(im, ((idx % 2) * cell, (idx // 2) * cell))
    return sheet


_QA_SYSTEM = (
    "You are a strict 3D game-asset reviewer. You are shown multiple rendered views "
    "of ONE generated 3D model on a white background. Judge whether it is a usable, "
    "high-quality game asset that matches the intended description and theme. "
    "Penalize: wrong/unrecognizable object, broken or incomplete geometry, holes, "
    "floating disconnected pieces, melted/blobby shape, smeared or wrong textures, "
    "multiple merged objects, or poor proportions. Reward: clean recognizable form, "
    "solid geometry, coherent materials, on-theme. "
    'Respond ONLY as JSON: {"score": <0-10 number>, "matches": <true/false>, '
    '"issues": ["short issue", ...], "suggestions": "concrete prompt changes to improve it"}'
)


def evaluate(images, asset_name, prompt, theme, model="qwen2.5vl:7b"):
    """Grade the renders. Returns dict: score(float), matches(bool), issues, suggestions, raw."""
    sheet = contact_sheet(images)
    review_imgs = [sheet] if sheet is not None else images
    user = (
        f"Intended asset: {asset_name}\n"
        f"Game theme/aesthetic: {theme}\n"
        f"Full description that was requested:\n{prompt}\n\n"
        "Review the rendered 3D model in the image(s) and return your JSON verdict."
    )
    data = oc.vision_json(model, user, review_imgs, system=_QA_SYSTEM,
                          options={"temperature": 0.2})
    try:
        score = float(data.get("score"))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(10.0, score))
    issues = data.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]
    return {
        "score": score,
        "matches": bool(data.get("matches", score >= 6)),
        "issues": issues,
        "suggestions": (data.get("suggestions") or "").strip(),
        "raw": data,
    }
