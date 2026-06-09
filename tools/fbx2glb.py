"""Run inside Blender:  blender --background --python fbx2glb.py -- <in.fbx> <out.glb>
Converts an FBX to a GLB (Y-up, applied transforms)."""
import sys
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=src)
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB",
                          export_yup=True, export_apply=True)
print("FBX2GLB_DONE", dst)
