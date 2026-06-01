"""Flask server for the chat UI.

This is adapted from the root-level test.py placeholder server. The UI sends
POST /generate with {"prompt": "..."} and expects a PNG response. This version
uses the quantum-conditioned diffusion POC instead of drawing placeholder text.
"""

from __future__ import annotations

import re
from pathlib import Path

from flask import Flask, request, send_file
from flask_cors import CORS

from quantum_only_generator import (
    generate_quantum_only_for_prompt,
    save_quantum_only_metrics,
    save_quantum_only_report,
)
from latent_quantum_generator import (
    generate_latent_quantum_for_prompt,
    save_latent_quantum_report,
)
from trained_quantum_generator import (
    generate_trained_quantum_for_prompt,
    save_trained_quantum_report,
)


app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
        }
    },
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "ui"
SHOT_MODES = {
    "fast": 256,
    "balanced": 768,
    "deepthinking": 1024,
}
DEFAULT_SHOT_MODE = "balanced"
QUANTUM_ONLY_DEPTH = 4


def _safe_name(prompt: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.lower()).strip("_")
    return name[:80] or "generated_image"


def _shots_for_mode(mode: str) -> tuple[str, int]:
    normalized = re.sub(r"[^a-zA-Z]", "", mode.lower())
    if normalized not in SHOT_MODES:
        normalized = DEFAULT_SHOT_MODE
    return normalized, SHOT_MODES[normalized]


def generate_prompt_image(prompt: str, mode: str = DEFAULT_SHOT_MODE) -> Path:
    """Generate a quantum-only PNG for a UI prompt and return its path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_prompt = _safe_name(prompt)
    mode_name, shots = _shots_for_mode(mode)
    latent_result = generate_latent_quantum_for_prompt(prompt, shots=shots)
    if latent_result is not None:
        image_path = OUTPUT_DIR / f"{safe_prompt}_{mode_name}_latent_vqg_report.png"
        save_latent_quantum_report(latent_result, image_path)
        return image_path

    trained_result = generate_trained_quantum_for_prompt(prompt, shots=shots, seed=7)
    if trained_result is not None:
        image_path = OUTPUT_DIR / f"{safe_prompt}_{mode_name}_trained_vqg_report.png"
        save_trained_quantum_report(trained_result, image_path)
        return image_path

    result = generate_quantum_only_for_prompt(
        prompt,
        shots=shots,
        depth=QUANTUM_ONLY_DEPTH,
        seed=7,
    )
    image_path = OUTPUT_DIR / f"{safe_prompt}_{mode_name}_quantum_only_report.png"
    metrics_path = OUTPUT_DIR / f"{safe_prompt}_{mode_name}_quantum_only_metrics.json"
    save_quantum_only_report(result, image_path)
    save_quantum_only_metrics(result, metrics_path)
    return image_path


@app.route("/ping", methods=["GET"])
def ping():
    return "ok", 200


@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    mode = data.get("mode", DEFAULT_SHOT_MODE)
    if not prompt:
        prompt = "yellow star"

    image_path = generate_prompt_image(prompt, mode)
    return send_file(image_path, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
