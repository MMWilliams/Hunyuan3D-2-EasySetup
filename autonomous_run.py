"""Headless CLI for the Autonomous Asset Factory (same loop as the web-UI panel).

Example:
    python autonomous_run.py --theme "Ghost in the Shell cyberpunk props" --count 2

Requires Ollama running with a chat model and a vision model. Reuses the same
Hunyuan3D pipelines the web app loads.
"""
import argparse
import random

import torch

from autonomous.runner import AutonomousRunner


def build_generate_fn(args):
    from hy3dgen.text2image import HunyuanDiTPipeline
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from hy3dgen.shapegen import FaceReducer, Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen.pipelines import export_to_trimesh
    from hy3dgen.rembg import BackgroundRemover

    print("loading pipelines (cached weights)...", flush=True)
    t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled', device=args.device)
    rmbg = BackgroundRemover()
    i23d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path, subfolder=args.subfolder, use_safetensors=True, device=args.device)
    face_reducer = FaceReducer()
    texgen = Hunyuan3DPaintPipeline.from_pretrained(args.texgen_model_path)
    print("pipelines ready.", flush=True)

    def generate(prompt, params):
        p = params or {}
        image = t2i(prompt)
        image = rmbg(image.convert('RGB'))
        gen = torch.Generator().manual_seed(random.randint(0, 10_000_000))
        outputs = i23d(image=image,
                       num_inference_steps=int(p.get('steps', 30)),
                       guidance_scale=float(p.get('guidance_scale', 5.0)),
                       generator=gen,
                       octree_resolution=int(p.get('octree_resolution', 256)),
                       num_chunks=int(p.get('num_chunks', 8000)),
                       output_type='mesh')
        mesh = export_to_trimesh(outputs)[0]
        mesh = face_reducer(mesh, int(p.get('target_faces', 40000)))
        textured = texgen(mesh, image)
        return textured, image, {}

    return generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theme', required=True)
    ap.add_argument('--asset_types', default='')
    ap.add_argument('--count', type=int, default=5)
    ap.add_argument('--threshold', type=float, default=7.0)
    ap.add_argument('--retries', type=int, default=2)
    ap.add_argument('--llm', default='qwen2.5:14b')
    ap.add_argument('--vision', default='qwen2.5vl:7b')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--out', default='autonomous_assets')
    ap.add_argument('--model_path', default='tencent/Hunyuan3D-2')
    ap.add_argument('--subfolder', default='hunyuan3d-dit-v2-0')
    ap.add_argument('--texgen_model_path', default='tencent/Hunyuan3D-2')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    generate = build_generate_fn(args)
    runner = AutonomousRunner(out_root=args.out)
    gen_params = {'steps': args.steps, 'guidance_scale': 5.0,
                  'octree_resolution': 256, 'target_faces': 40000, 'num_chunks': 8000}
    last = None
    for log, gallery, preview, counters in runner.run(
            generate, args.theme, args.asset_types, args.count,
            args.threshold, args.retries, args.llm, args.vision, gen_params):
        if counters != last:
            print("   progress:", counters, flush=True)
            last = counters
    print("FINISHED:", runner.counters, flush=True)


if __name__ == '__main__':
    main()
