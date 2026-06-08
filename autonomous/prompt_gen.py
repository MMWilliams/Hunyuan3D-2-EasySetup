"""Use a local LLM to author detailed, 3D-friendly asset prompts from a brief,
and to refine a prompt given vision-QA feedback."""
import re

from . import ollama_client as oc

# Appended to every prompt so HunyuanDiT makes a clean, single, fully-visible
# object that converts well to a 3D mesh.
CAPTURE_SUFFIX = (
    " The object is a single complete item, centered and entirely visible, "
    "isolated on a plain solid neutral background, studio product photography, "
    "soft even lighting, sharp focus, high detail, no other objects."
)

_PLANNER_SYSTEM = (
    "You are an art director generating prompts for a text-to-3D model that makes "
    "individual game assets (one object per prompt). For the given game/theme brief, "
    "invent distinct, useful assets that fit the setting and aesthetic. "
    "Each asset gets:\n"
    "  - 'name': a short, specific, file-safe name (2-5 words, e.g. 'rusted fuel barrel').\n"
    "  - 'prompt': ONE vivid paragraph (4-7 sentences) describing the object's form, "
    "materials, colors, wear, and distinctive details. Describe ONLY a single object. "
    "Do NOT mention cameras, backgrounds, lighting, or rendering -- that is added "
    "automatically. Do NOT include scenes, characters, or multiple items.\n"
    "Maximize variety; avoid repeating earlier assets. Respond ONLY as JSON: "
    '{"assets": [{"name": "...", "prompt": "..."}, ...]}'
)


def plan_assets(theme, asset_types, n, avoid_names=None, model="qwen2.5:14b"):
    """Return a list of {name, prompt} dicts (prompt already has CAPTURE_SUFFIX)."""
    avoid_names = avoid_names or []
    avoid = ""
    if avoid_names:
        avoid = ("\nAlready generated (do NOT repeat these, make different ones): "
                 + "; ".join(avoid_names[-60:]))
    asset_line = f"\nAsset types/focus requested: {asset_types}" if asset_types else ""
    user = (
        f"Game / theme / aesthetic:\n{theme}{asset_line}\n\n"
        f"Generate {n} new, distinct assets that fit this.{avoid}"
    )
    data = oc.chat_json(model, _PLANNER_SYSTEM, user,
                        options={"temperature": 0.9, "top_p": 0.95})
    assets = data.get("assets") if isinstance(data, dict) else None
    if not assets and isinstance(data, list):
        assets = data
    out = []
    for a in (assets or []):
        if not isinstance(a, dict):
            continue
        name = (a.get("name") or "").strip()
        prompt = (a.get("prompt") or "").strip()
        if not name or not prompt:
            continue
        out.append({"name": name, "prompt": prompt + CAPTURE_SUFFIX})
    return out


_REFINE_SYSTEM = (
    "You revise a single text-to-3D asset prompt to fix problems a vision reviewer "
    "found in the generated 3D model. Keep the same asset identity and theme, but "
    "rewrite the description to directly address the issues (clearer silhouette, "
    "simpler/cleaner form, stronger material cues, fix proportions, etc.). "
    "Describe ONLY one object; do not mention cameras/background/lighting. "
    'Respond ONLY as JSON: {"prompt": "..."}'
)


def refine_prompt(name, prev_prompt, issues, suggestions, theme, model="qwen2.5:14b"):
    base = prev_prompt.replace(CAPTURE_SUFFIX, "").strip()
    issues_txt = "; ".join(issues) if isinstance(issues, list) else str(issues or "")
    user = (
        f"Theme: {theme}\nAsset: {name}\n\nPrevious prompt:\n{base}\n\n"
        f"Reviewer issues: {issues_txt}\n"
        f"Reviewer suggestions: {suggestions}\n\n"
        "Rewrite the prompt to fix these."
    )
    data = oc.chat_json(model, _REFINE_SYSTEM, user, options={"temperature": 0.7})
    new = (data.get("prompt") or "").strip() if isinstance(data, dict) else ""
    if not new:
        return prev_prompt
    return new + CAPTURE_SUFFIX


def slugify(name, fallback="asset"):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return s[:60] or fallback
