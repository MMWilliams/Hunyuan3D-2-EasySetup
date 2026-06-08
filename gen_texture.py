import sys, os, time, trimesh
from PIL import Image
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
from hy3dgen.shapegen.postprocessors import FaceReducer
from hy3dgen.rembg import BackgroundRemover

mesh_path = sys.argv[1]
img_path = sys.argv[2]
out_path = sys.argv[3]
faces = int(sys.argv[4]) if len(sys.argv) > 4 else 60000
res = int(sys.argv[5]) if len(sys.argv) > 5 else 1024

t0 = time.time()
print("loading paint pipeline...", flush=True)
pipe = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
# Lower render/texture resolution (2048 -> res) — huge speedup, still sharp.
pipe.config.render_size = res
pipe.config.texture_size = res
pipe.render = MeshRender(default_resolution=res, texture_size=res)
print(f"pipeline ready {time.time()-t0:.0f}s", flush=True)

mesh = trimesh.load(mesh_path, force="mesh")
print(f"loaded mesh: {len(mesh.faces)} faces; decimating -> {faces}...", flush=True)
t1 = time.time()
mesh = FaceReducer()(mesh, max_facenum=faces)  # 1.8M -> ~60k: rasterize/bake far faster
print(f"decimated in {time.time()-t1:.0f}s", flush=True)

img = BackgroundRemover()(Image.open(img_path).convert("RGB"))
print("painting texture...", flush=True)
t2 = time.time()
mesh = pipe(mesh, image=img)
print(f"painted in {time.time()-t2:.0f}s", flush=True)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
mesh.export(out_path)
print(f"DONE -> {out_path}  (total {time.time()-t0:.0f}s)", flush=True)
