import sys, os, time
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.rembg import BackgroundRemover

img_path = sys.argv[1] if len(sys.argv) > 1 else "inputs/building1.png"
out_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/building1_shape.glb"
model = "tencent/Hunyuan3D-2"

t0 = time.time()
print("loading shapegen pipeline (downloads weights on first run)...", flush=True)
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model)
print(f"pipeline ready {time.time()-t0:.0f}s; prepping image...", flush=True)

img = Image.open(img_path).convert("RGB")
img = BackgroundRemover()(img)  # → RGBA with alpha matte

print("generating mesh...", flush=True)
mesh = pipe(image=img)[0]
os.makedirs(os.path.dirname(out_path), exist_ok=True)
mesh.export(out_path)
print(f"DONE -> {out_path}  ({time.time()-t0:.0f}s total)", flush=True)
