"""Flask server for the quantum-simulator-trained Fashion-MNIST demo.

This server mirrors the simple UI API used by `slopgpt.html`:

POST /generate
{
    "prompt": "sneaker",
    "mode": "balanced"
}

It only uses the checkpoint trained in this folder. It does not call the older
Tiny ImageNet, FRQI, shape, or IBM hardware paths.
"""

from __future__ import annotations

import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from generate_quantum_sim_sample import (
    CHECKPOINT_PATH,
    attributes_to_string,
    generate_image,
    match_prompt,
    parse_prompt_attributes,
    safe_name,
    save_image,
)


THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR / "outputs" / "ui"

MODE_CONFIGS = {
    "fast": {"shots": 64, "candidates": 1, "latent_scale": 0.7},
    "balanced": {"shots": 128, "candidates": 4, "latent_scale": 1.0},
    "deepthinking": {"shots": 256, "candidates": 12, "latent_scale": 1.4},
    "realquantumdemo": {"shots": 256, "candidates": 12, "latent_scale": 1.4},
}
DEFAULT_MODE = "balanced"
SUPPORTED_PROMPTS = (
    "t-shirt, tshirt, shirt, trouser, pants, pullover, dress, coat, "
    "sandal, sneaker, shoe, bag, ankle boot, boot"
)


@dataclass(frozen=True)
class GeneratedImage:
    path: Path
    metadata: dict[str, Any]


class UnsupportedPromptError(ValueError):
    """Raised when the prompt cannot be mapped to a Fashion-MNIST class."""


app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "expose_headers": [
                "X-Generator",
                "X-Matched-Class",
                "X-Class-Id",
                "X-Shots",
                "X-Seed",
                "X-Variation",
                "X-Detected-Attributes",
                "X-Output-Path",
            ],
        }
    },
)


def mode_config(mode: str | None) -> tuple[str, dict[str, int]]:
    if not mode:
        return DEFAULT_MODE, MODE_CONFIGS[DEFAULT_MODE]
    normalized = re.sub(r"[^a-zA-Z]", "", mode.lower())
    if normalized not in MODE_CONFIGS:
        normalized = DEFAULT_MODE
    return normalized, MODE_CONFIGS[normalized]


def request_json() -> dict[str, Any]:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def send_png(generated: GeneratedImage):
    response = send_file(generated.path, mimetype="image/png")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Generator"] = "quantum_simulator_trained_vqg"
    response.headers["X-Output-Path"] = str(generated.path)
    for key, value in generated.metadata.items():
        response.headers["X-" + key.replace("_", "-").title()] = str(value)
    return response


def generate_prompt_image(
    prompt: str,
    mode: str,
    *,
    seed: int,
    variation: float,
) -> GeneratedImage:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. Train first with "
            "python SlopGPT-QuantumTrain/train_quantum_sim_fashion_mnist_16.py "
            "--epochs 1 --samples-per-class 1 --shots 64 --max-steps 10 "
            "--data-dir \"C:\\Personal Files\\Universite\\Quantum\\IQuCodeFest2026-CoherenceTotale\\Fashion-MNIST\""
        )

    try:
        label, class_name = match_prompt(prompt)
    except ValueError as exc:
        raise UnsupportedPromptError(
            f"Unsupported prompt. Try one of: {SUPPORTED_PROMPTS}."
        ) from exc

    mode_name, config = mode_config(mode)
    shots = int(config["shots"])
    candidates = int(config["candidates"])
    latent_scale = float(config["latent_scale"])
    attributes = parse_prompt_attributes(prompt)
    image = generate_image(
        label,
        shots=shots,
        seed=seed,
        prompt=prompt,
        variation=variation,
        latent_scale=latent_scale,
        candidates=candidates,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        OUTPUT_DIR
        / f"{safe_name(prompt)}_{mode_name}_quantum_sim_trained_{shots}shots.png"
    )
    save_image(image, output_path)
    return GeneratedImage(
        path=output_path,
        metadata={
            "matched_class": class_name,
            "class_id": label,
            "shots": shots,
            "seed": seed,
            "variation": variation,
            "detected_attributes": attributes_to_string(attributes),
            "candidates": candidates,
            "latent_scale": latent_scale,
        },
    )


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify(
        {
            "status": "ok",
            "generator": "quantum_simulator_trained_vqg",
            "checkpoint_available": CHECKPOINT_PATH.exists(),
            "checkpoint_path": str(CHECKPOINT_PATH),
            "output_dir": str(OUTPUT_DIR),
            "supported_prompts": SUPPORTED_PROMPTS,
            "modes": {name: config["shots"] for name, config in MODE_CONFIGS.items()},
            "mode_configs": MODE_CONFIGS,
        }
    ), 200


@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return "", 204

    data = request_json()
    prompt = str(data.get("prompt", "sneaker")).strip() or "sneaker"
    mode = str(data.get("mode", DEFAULT_MODE)).strip()
    seed_raw = data.get("seed", None)
    variation_raw = data.get("variation", 0.05)

    if seed_raw is None or seed_raw == "":
        seed = time.time_ns() % (2**31 - 1)
    else:
        try:
            seed = int(seed_raw)
        except (TypeError, ValueError):
            seed = time.time_ns() % (2**31 - 1)

    try:
        variation = float(variation_raw)
    except (TypeError, ValueError):
        variation = 0.05
    variation = max(0.0, variation)

    try:
        generated = generate_prompt_image(prompt, mode, seed=seed, variation=variation)
        return send_png(generated)
    except UnsupportedPromptError as exc:
        return jsonify(
            {
                "error": "unsupported_prompt",
                "message": str(exc),
                "prompt": prompt,
                "supported_prompts": SUPPORTED_PROMPTS,
            }
        ), 400
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify(
            {
                "error": "generation_failed",
                "message": str(exc),
                "prompt": prompt,
                "mode": mode,
            }
        ), 500


if __name__ == "__main__":
    print("Starting SlopGPT QuantumTrain UI server")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    app.run(host="127.0.0.1", port=8080, debug=True)
