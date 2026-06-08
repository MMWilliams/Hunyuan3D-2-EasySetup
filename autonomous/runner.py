"""The autonomous loop: plan prompts (LLM) -> generate (Hunyuan) -> render+grade
(vision) -> approve & save, or refine the prompt and retry."""
import json
import os
import threading
import time
import traceback

from . import ollama_client as oc
from . import prompt_gen as pg
from . import qa as qa_mod


def _ts():
    return time.strftime("%Y%m%d_%H%M%S")


class AutonomousRunner:
    def __init__(self, out_root):
        self.out_root = out_root
        self.stop_flag = threading.Event()
        self.busy = False
        self.log_lines = []
        self.approved = []          # list of dicts {name, glb, sheet, score}
        self.counters = {"planned": 0, "generated": 0, "approved": 0, "rejected": 0}

    # ---- helpers --------------------------------------------------------
    def stop(self):
        self.stop_flag.set()
        self._log("Stop requested - will halt after the current asset.")

    def _log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.log_lines.append(line)
        self.log_lines = self.log_lines[-400:]
        print("[autonomous] " + msg, flush=True)

    def _log_text(self):
        return "\n".join(reversed(self.log_lines))

    def _state(self, preview=None):
        gallery = [(a["sheet"], f"{a['name']}  ({a['score']:.1f})") for a in self.approved
                   if a.get("sheet") and os.path.exists(a["sheet"])]
        return self._log_text(), gallery[::-1], preview, dict(self.counters)

    def _unique_path(self, run_dir, slug, ext):
        path = os.path.join(run_dir, f"{slug}.{ext}")
        i = 2
        while os.path.exists(path):
            path = os.path.join(run_dir, f"{slug}-{i}.{ext}")
            i += 1
        return path

    # ---- main loop ------------------------------------------------------
    def run(self, generate_fn, theme, asset_types, target_count,
            threshold, max_retries, llm_model, vision_model, gen_params):
        """Generator yielding (log_text, gallery, preview_img, counters)."""
        self.stop_flag.clear()
        self.busy = True
        self.log_lines, self.approved = [], []
        self.counters = {"planned": 0, "generated": 0, "approved": 0, "rejected": 0}

        run_dir = os.path.join(self.out_root, f"{pg.slugify(theme, 'run')}_{_ts()}")
        rej_dir = os.path.join(run_dir, "rejected")
        os.makedirs(rej_dir, exist_ok=True)
        manifest = os.path.join(run_dir, "manifest.jsonl")
        self._log(f"Run folder: {run_dir}")
        if not oc.is_up():
            self._log("ERROR: Ollama is not reachable at " + oc.OLLAMA_HOST)
            self.busy = False
            yield self._state()
            return
        self._log(f"Theme: {theme[:120]}")
        self._log(f"LLM: {llm_model} | Vision: {vision_model} | "
                  f"threshold {threshold} | retries {max_retries} | "
                  f"target {'∞' if target_count <= 0 else target_count}")
        yield self._state()

        avoid, queue = [], []
        unlimited = target_count <= 0

        while not self.stop_flag.is_set() and (unlimited or self.counters["approved"] < target_count):
            # refill the prompt queue from the planner
            if not queue:
                self._log("Planning new assets with the LLM...")
                yield self._state()
                try:
                    batch = pg.plan_assets(theme, asset_types, n=5,
                                           avoid_names=avoid, model=llm_model)
                except Exception as e:
                    self._log(f"Planner error: {e}; retrying in 5s")
                    yield self._state()
                    time.sleep(5)
                    continue
                if not batch:
                    self._log("Planner returned nothing; retrying in 5s")
                    yield self._state()
                    time.sleep(5)
                    continue
                queue.extend(batch)
                self.counters["planned"] += len(batch)
                self._log(f"Planned {len(batch)} assets: "
                          + ", ".join(b["name"] for b in batch))
                yield self._state()

            item = queue.pop(0)
            name, prompt = item["name"], item["prompt"]
            avoid.append(name)
            slug = pg.slugify(name)
            approved = False
            best = None  # (score, sheet_path, verdict)

            for attempt in range(1, max_retries + 2):
                if self.stop_flag.is_set():
                    break
                self._log(f"[{name}] attempt {attempt}: generating...")
                yield self._state()
                try:
                    mesh, concept, stats = generate_fn(prompt, gen_params)
                except Exception as e:
                    self._log(f"[{name}] generation failed: {e}")
                    traceback.print_exc()
                    yield self._state()
                    break
                self.counters["generated"] += 1

                self._log(f"[{name}] rendering + grading with {vision_model}...")
                yield self._state()
                try:
                    views = qa_mod.render_views(mesh, n_views=4)
                    sheet = qa_mod.contact_sheet(views)
                    verdict = qa_mod.evaluate(views, name, prompt, theme, model=vision_model)
                except Exception as e:
                    self._log(f"[{name}] QA failed: {e}; treating as low score")
                    traceback.print_exc()
                    sheet, verdict = None, {"score": 0.0, "issues": ["qa error"],
                                            "suggestions": "", "matches": False}

                score = verdict["score"]
                sheet_path = self._unique_path(run_dir if score >= threshold else rej_dir,
                                               f"{slug}_a{attempt}", "png")
                if sheet is not None:
                    sheet.save(sheet_path)
                self._log(f"[{name}] score {score:.1f}/10"
                          + (f" | issues: {', '.join(verdict['issues'][:3])}" if verdict["issues"] else ""))
                if best is None or score > best[0]:
                    best = (score, sheet_path, verdict)
                yield self._state(preview=sheet_path if sheet is not None else None)

                if score >= threshold:
                    # ---- approve & save ----
                    glb = self._unique_path(run_dir, slug, "glb")
                    try:
                        mesh.export(glb)
                    except Exception as e:
                        self._log(f"[{name}] export failed: {e}")
                        break
                    final_sheet = self._unique_path(run_dir, slug, "png")
                    if sheet is not None:
                        sheet.save(final_sheet)
                    concept_path = self._unique_path(run_dir, slug + "_concept", "png")
                    try:
                        if concept is not None:
                            concept.convert("RGB").save(concept_path)
                    except Exception:
                        concept_path = None
                    with open(self._unique_path(run_dir, slug, "prompt.txt"), "w", encoding="utf-8") as f:
                        f.write(prompt)
                    with open(self._unique_path(run_dir, slug, "qa.json"), "w", encoding="utf-8") as f:
                        json.dump({"name": name, "score": score, "attempts": attempt,
                                   "verdict": verdict["raw"], "stats": stats}, f, indent=2)
                    with open(manifest, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"name": name, "slug": slug, "glb": os.path.basename(glb),
                                            "score": score, "attempts": attempt,
                                            "prompt": prompt}) + "\n")
                    self.counters["approved"] += 1
                    self.approved.append({"name": name, "glb": glb,
                                          "sheet": final_sheet if sheet is not None else None,
                                          "score": score})
                    self._log(f"[{name}] APPROVED ({score:.1f}) -> {os.path.basename(glb)}")
                    approved = True
                    yield self._state(preview=final_sheet if sheet is not None else None)
                    break

                # ---- refine and retry ----
                if attempt <= max_retries:
                    self._log(f"[{name}] below {threshold}; refining prompt...")
                    yield self._state()
                    try:
                        prompt = pg.refine_prompt(name, prompt, verdict["issues"],
                                                  verdict["suggestions"], theme, model=llm_model)
                    except Exception as e:
                        self._log(f"[{name}] refine failed: {e}")

            if not approved and not self.stop_flag.is_set():
                self.counters["rejected"] += 1
                bs = best[0] if best else 0.0
                self._log(f"[{name}] REJECTED after {max_retries + 1} attempts (best {bs:.1f})")
                yield self._state()

        self._log("Stopped." if self.stop_flag.is_set() else "Target reached - done.")
        self.busy = False
        yield self._state()
