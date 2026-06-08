import sys, time, torch
from diffusers import StableDiffusionXLPipeline

prompt = sys.argv[1]
out = sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 30

SUFFIX = ", full building centered and entirely visible, isolated on a plain solid white background, architectural product photography, soft even daylight, sharp focus, high detail"
NEG = "cropped, cut off, partial, multiple buildings, people, cars, street, ground, clutter, text, watermark, blurry, lowres"

t0 = time.time()
print("loading SDXL (downloads on first run)...", flush=True)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
print(f"ready {time.time()-t0:.0f}s; generating...", flush=True)
img = pipe(prompt=prompt + SUFFIX, negative_prompt=NEG,
           height=1024, width=1024, num_inference_steps=steps, guidance_scale=6.5).images[0]
img.save(out)
print(f"DONE -> {out} ({time.time()-t0:.0f}s)", flush=True)
