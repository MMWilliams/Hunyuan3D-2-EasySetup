"""Tiny Ollama HTTP client (chat + vision), stdlib-only except `requests`."""
import base64
import io
import json
import os

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST


def _img_to_b64(img):
    """Accept a PIL.Image, a numpy array, a path, or raw bytes -> base64 str."""
    from PIL import Image
    import numpy as np

    if isinstance(img, str):
        with open(img, "rb") as f:
            return base64.b64encode(f.read()).decode()
    if isinstance(img, bytes):
        return base64.b64encode(img).decode()
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype("uint8"))
    if isinstance(img, Image.Image):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    raise TypeError(f"unsupported image type: {type(img)}")


def is_up(timeout=3):
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def list_models():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def chat(model, messages, fmt=None, options=None, timeout=600):
    """messages: list of {role, content, [images: [b64,...]]}. Returns content str."""
    payload = {"model": model, "messages": messages, "stream": False}
    if fmt:
        payload["format"] = fmt
    if options:
        payload["options"] = options
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def chat_json(model, system, user, options=None, timeout=600):
    """Chat forcing a JSON object response; returns a parsed dict (best effort)."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    raw = chat(model, msgs, fmt="json", options=options, timeout=timeout)
    return _loads_loose(raw)


def vision_json(model, prompt, images, system=None, options=None, timeout=600):
    """Vision chat (images attached to the user turn), JSON response -> dict."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({
        "role": "user",
        "content": prompt,
        "images": [_img_to_b64(im) for im in images],
    })
    raw = chat(model, msgs, fmt="json", options=options, timeout=timeout)
    return _loads_loose(raw)


def _loads_loose(raw):
    """Parse JSON that may be wrapped in prose or code fences."""
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    # strip code fences
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("\n") + 1:] if "\n" in raw else raw
    # grab the outermost {...}
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(raw[s:e + 1])
        except Exception:
            pass
    return {}
