"""Models selectable in Jarvis: the "brain" controls the camera through tools, the "eyes" describe the photo."""

import json
from pathlib import Path

import httpx

BIELIK_URL = "http://localhost:8000"        # serve_bielik.sh (vLLM, checkpoint selected by VARIANT)
OLLAMA_OPENCLAW = "http://localhost:11434"  # existing ollama-with-openclaw container (Ollama 0.17)
OLLAMA_JARVIS = "http://localhost:11435"    # serve_ollama.sh (Ollama 0.34, because Gemma 4 needs a newer one)

BRAINS = {
    "bielik": {"label": "Bielik 1.5B", "backend": "vllm", "base": BIELIK_URL},
    "gemma4": {"label": "Gemma 4 E4B", "backend": "ollama", "base": OLLAMA_JARVIS, "model": "gemma4:e4b"},
}
VISIONS = {
    "gemma3": {"label": "Gemma 3 12B", "base": OLLAMA_OPENCLAW, "model": "jaro-jm/vlm"},
    "gemma4": {"label": "Gemma 4 E4B", "base": OLLAMA_JARVIS, "model": "gemma4:e4b"},
}
DEFAULT = {"brain": "bielik", "vision": "gemma3"}
CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config():
    try:
        saved = json.loads(CONFIG_PATH.read_text())
    except (OSError, ValueError):
        saved = {}
    return {
        "brain": saved.get("brain") if saved.get("brain") in BRAINS else DEFAULT["brain"],
        "vision": saved.get("vision") if saved.get("vision") in VISIONS else DEFAULT["vision"],
    }


def save_config(config):
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")


_capabilities = {}


def capabilities(base, model):
    """Capabilities of a model in Ollama (vision, tools, thinking...); empty tuple when they cannot be checked."""
    if (base, model) not in _capabilities:
        try:
            r = httpx.post(f"{base}/api/show", json={"model": model}, timeout=5)
            r.raise_for_status()
            _capabilities[base, model] = tuple(r.json().get("capabilities", []))
        except (httpx.HTTPError, ValueError):
            return ()
    return _capabilities[base, model]


def unload(base, model):
    """Frees the model from VRAM (keep_alive=0); a loaded Gemma keeps the GPU at ~100% even with no requests."""
    try:
        httpx.post(f"{base}/api/generate", json={"model": model, "keep_alive": 0}, timeout=10)
    except httpx.HTTPError:
        pass
