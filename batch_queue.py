"""Run a queue of themed autonomous batches back-to-back, loading the Hunyuan
pipelines only once. Each batch saves into the same combined run folder."""
import types

from autonomous_run import build_generate_fn
from autonomous.runner import AutonomousRunner

RUN_DIR = (r"C:\Users\reese\Hunyuan3D-2\autonomous_assets"
           r"\the_game_takes_place_in_a_sprawling_near_future_megacity_whe_20260608_193944")

SLUM_TYPES = ("slum buildings, container tenements, modular housing capsules, ramshackle storefronts, "
              "corrugated shacks, market arcades, street vendor stalls, street-food carts, food trucks, "
              "noodle stands, neon signs, hanging shop signs, holographic billboards, hand-painted boards, "
              "traffic signs and signals, street lights, lamp posts, utility and power boxes, jury-rigged "
              "generators, cable junctions, satellite dishes, AC units, water tanks, dumpsters, trash piles, "
              "garbage bags, oil-drum fire barrels, scrap heaps, vending machines, salvaged robots and drones, "
              "repair benches, makeshift barricades and scrap cover")
WAREHOUSE_TYPES = ("rusted forklifts, broken pallet racks, collapsed shelving, scattered pallets, old crates, "
                   "cargo containers, rusted oil drums and barrels, abandoned conveyors, hanging chains, "
                   "broken hand trucks, dust-covered machinery, dead control panels, cracked pillars, debris "
                   "piles, rubble, torn tarps, broken loading-dock equipment, defunct robots, hanging work "
                   "lights, broken roller doors")
LAB_TYPES = ("robotic arms, robot assembly and test stations, humanoid robot frames on stands, prototype "
             "service robots and drones, diagnostic and calibration rigs, lab workbenches, tool and component "
             "cabinets, server and compute racks, holographic display terminals, 3D printers and fabricators, "
             "robotic charging docks, parts bins, precision testing rigs, control consoles, inspection "
             "stations, overhead gantry cranes, sterile storage units")

# (name, theme_file, asset_types, count, threshold, retries)
JOBS = [
    ("slum",      "slum_theme.txt",      SLUM_TYPES,      100, 8.0, 2),
    ("warehouse", "warehouse_theme.txt", WAREHOUSE_TYPES,  20, 8.0, 2),
    ("lab",       "robotics_lab_theme.txt", LAB_TYPES,     50, 8.0, 2),
]


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def main():
    args = types.SimpleNamespace(device="cuda", model_path="tencent/Hunyuan3D-2",
                                 subfolder="hunyuan3d-dit-v2-0",
                                 texgen_model_path="tencent/Hunyuan3D-2")
    generate = build_generate_fn(args)
    runner = AutonomousRunner(out_root="autonomous_assets")
    gen_params = {"steps": 30, "guidance_scale": 5.0, "octree_resolution": 256,
                  "target_faces": 40000, "num_chunks": 8000}

    for name, theme_file, types_str, count, thr, retries in JOBS:
        theme = read(theme_file)
        print(f"\n===== BATCH START: {name} (count {count}, threshold {thr}) =====", flush=True)
        last = None
        for log, gallery, preview, counters in runner.run(
                generate, theme, types_str, count, thr, retries,
                "qwen2.5:14b", "qwen2.5vl:7b", gen_params, run_dir=RUN_DIR):
            if counters != last:
                print(f"  [{name}] {counters}", flush=True)
                last = counters
        print(f"===== BATCH DONE: {name} -> {runner.counters} =====", flush=True)

    print("\nALL BATCHES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
