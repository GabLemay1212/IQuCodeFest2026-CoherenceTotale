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

from quantum_diffusion_poc import (
    build_digit_prototypes,
    generate_for_prompt,
    load_digit_data,
    parse_prompt,
    save_metrics,
    save_visual_report,
    train_evaluator,
)
from shape_diffusion import (
    generate_abstract_for_prompt,
    generate_shape_for_prompt,
    parse_shape_prompt,
    save_shape_metrics,
    save_shape_visual_report,
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
_DIGIT_CONTEXT: tuple[dict[int, object], object] | None = None


def _safe_name(prompt: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.lower()).strip("_")
    return name[:80] or "generated_image"


def _get_digit_context():
    """Load digit prototypes and evaluator once for the Flask process."""
    global _DIGIT_CONTEXT
    if _DIGIT_CONTEXT is None:
        images, labels = load_digit_data()
        prototypes = build_digit_prototypes(images, labels)
        evaluator = train_evaluator(images, labels)
        _DIGIT_CONTEXT = (prototypes, evaluator)
    return _DIGIT_CONTEXT


def _detect_prompt_type(prompt: str) -> str:
    try:
        parse_prompt(prompt)
        return "digit"
    except ValueError:
        pass

    try:
        parse_shape_prompt(prompt)
        return "shape"
    except ValueError:
        pass

    return "abstract"


def generate_prompt_image(prompt: str) -> Path:
    """Generate a report PNG for a UI prompt and return its path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_prompt = _safe_name(prompt)
    prompt_type = _detect_prompt_type(prompt)

    if prompt_type == "digit":
        prototypes, evaluator = _get_digit_context()
        result = generate_for_prompt(
            prompt,
            prototypes,
            evaluator,
            shots=256,
            steps=20,
            seed=7,
        )
        image_path = OUTPUT_DIR / f"{safe_prompt}_digit_report.png"
        metrics_path = OUTPUT_DIR / f"{safe_prompt}_digit_metrics.json"
        save_visual_report([result], image_path)
        save_metrics([result], metrics_path)
        return image_path

    if prompt_type == "shape":
        result = generate_shape_for_prompt(prompt, shots=256, steps=20, seed=7)
        suffix = "shape"
    else:
        result = generate_abstract_for_prompt(prompt, shots=256, steps=20, seed=7)
        suffix = "abstract"

    image_path = OUTPUT_DIR / f"{safe_prompt}_{suffix}_report.png"
    metrics_path = OUTPUT_DIR / f"{safe_prompt}_{suffix}_metrics.json"
    save_shape_visual_report(result, image_path)
    save_shape_metrics(result, metrics_path)
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
    if not prompt:
        prompt = "yellow star"

    image_path = generate_prompt_image(prompt)
    return send_file(image_path, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
