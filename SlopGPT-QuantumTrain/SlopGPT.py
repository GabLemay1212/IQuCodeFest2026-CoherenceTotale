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
import random
import time
import traceback
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from generate_quantum_sim_sample import (
    CHECKPOINT_PATH,
    IBMJobPendingError,
    attributes_to_string,
    generate_result,
    match_prompt,
    parse_prompt_attributes,
    safe_name,
    save_image,
)


THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR / "outputs" / "ui"

MODE_CONFIGS = {
    "fast": {"shots": 64, "candidates": 1, "latent_scale": 0.7, "backend": "simulator"},
    "balanced": {"shots": 128, "candidates": 4, "latent_scale": 1.0, "backend": "simulator"},
    "deepthinking": {"shots": 256, "candidates": 12, "latent_scale": 1.4, "backend": "simulator"},
    "ibmfast": {
        "shots": 256,
        "candidates": 1,
        "latent_scale": 0.7,
        "backend": "ibm",
        "ibm_timeout_seconds": 45,
    },
    "ibmbalanced": {
        "shots": 512,
        "candidates": 1,
        "latent_scale": 1.0,
        "backend": "ibm",
        "ibm_timeout_seconds": 45,
    },
    "ibmdeepthinking": {
        "shots": 1024,
        "candidates": 1,
        "latent_scale": 1.4,
        "backend": "ibm",
        "ibm_timeout_seconds": 45,
    },
}
DEFAULT_MODE = "balanced"
MODEL_CONFIGS = {
    "fashion": {
        "label": "Model 1 - Fashion pixels",
        "generator": "quantum_simulator_trained_vqg",
    },
    "vector": {
        "label": "Model 2 - Vector shapes",
        "generator": "quantum_vector_svg",
    },
}
DEFAULT_MODEL = "fashion"
VECTOR_STEP_CONFIGS = {
    "fast": 20,
    "balanced": 80,
    "deepthinking": 160,
    "ibmfast": 20,
    "ibmbalanced": 80,
    "ibmdeepthinking": 160,
}
SUPPORTED_PROMPTS = (
    "t-shirt, tshirt, shirt, trouser, pants, pullover, dress, coat, "
    "sandal, sneaker, shoe, bag, ankle boot, boot"
)
SUPPORTED_VECTOR_PROMPTS = (
    "circle, star, hexagon, triangle, square, diamond, cross, pentagon, "
    "octagon, heart with colors like red, blue, yellow, green, purple, pink"
)
VECTOR_COLORS = {
    "red": "#ef4444",
    "orange": "#f97316",
    "yellow": "#eab308",
    "green": "#22c55e",
    "blue": "#3b82f6",
    "purple": "#a855f7",
    "pink": "#ec4899",
    "black": "#111827",
    "white": "#f8fafc",
    "gray": "#6b7280",
}
VECTOR_SHAPES = {
    "circle",
    "star",
    "hexagon",
    "triangle",
    "square",
    "diamond",
    "cross",
    "pentagon",
    "octagon",
    "heart",
}


@dataclass(frozen=True)
class GeneratedImage:
    path: Path
    metadata: dict[str, Any]
    mime_type: str = "image/png"


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
                "X-Quantum-Backend",
                "X-Ibm-Backend",
                "X-Ibm-Job-Id",
                "X-Ibm-Status",
                "X-Ibm-Fallback-Reason",
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
    response = send_file(generated.path, mimetype=generated.mime_type)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Generator"] = "quantum_simulator_trained_vqg"
    response.headers["X-Output-Path"] = str(generated.path)
    for key, value in generated.metadata.items():
        response.headers["X-" + key.replace("_", "-").title()] = str(value)
    return response


def model_config(model: str | None) -> tuple[str, dict[str, str]]:
    if not model:
        return DEFAULT_MODEL, MODEL_CONFIGS[DEFAULT_MODEL]
    normalized = re.sub(r"[^a-zA-Z0-9]", "", model.lower())
    if normalized in {"model1", "first", "fashion", "pixel", "pixels"}:
        normalized = "fashion"
    elif normalized in {"model2", "second", "vector", "svg", "shape", "shapes"}:
        normalized = "vector"
    if normalized not in MODEL_CONFIGS:
        normalized = DEFAULT_MODEL
    return normalized, MODEL_CONFIGS[normalized]


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
    backend = "ibm" if str(config.get("backend", "simulator")) == "ibm" else "simulator"
    ibm_timeout_seconds = float(config.get("ibm_timeout_seconds", 45))
    attributes = parse_prompt_attributes(prompt)
    result = generate_result(
        label,
        shots=shots,
        seed=seed,
        prompt=prompt,
        variation=variation,
        latent_scale=latent_scale,
        candidates=candidates,
        backend=backend,
        ibm_timeout_seconds=ibm_timeout_seconds,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        OUTPUT_DIR
        / f"{safe_name(prompt)}_{mode_name}_quantum_sim_trained_{shots}shots.png"
    )
    save_image(result.image, output_path)
    return GeneratedImage(
        path=output_path,
        metadata={
            "matched_class": class_name,
            "class_id": label,
            "shots": shots,
            "seed": result.seed,
            "variation": variation,
            "detected_attributes": attributes_to_string(attributes),
            "candidates": candidates,
            "latent_scale": latent_scale,
            "quantum_backend": result.quantum_backend,
            "ibm_backend": result.ibm_backend_name,
            "ibm_job_id": result.ibm_job_id,
            "ibm_status": result.ibm_status,
            "ibm_fallback_reason": result.ibm_fallback_reason,
        },
    )


def parse_light_vector_prompt(prompt: str, item_count: int | None = None) -> list[tuple[str, str]]:
    parts = re.split(r"\band\b|,", prompt.lower().replace("-", " "))
    items: list[tuple[str, str]] = []
    for part in parts:
        tokens = part.split()
        shape = next((token for token in tokens if token in VECTOR_SHAPES), None)
        color = next((token for token in tokens if token in VECTOR_COLORS), None)
        if shape:
            items.append((shape, color or "blue"))
    items = items or [("circle", "blue")]
    if item_count is None:
        return items
    total = max(1, min(20, int(item_count)))
    return [items[index % len(items)] for index in range(total)]


def polygon_points(cx: float, cy: float, radius: float, sides: int, rotate: float = -90.0) -> str:
    import math

    points = []
    for index in range(sides):
        angle = math.radians(rotate + index * 360.0 / sides)
        points.append(f"{cx + math.cos(angle) * radius:.2f},{cy + math.sin(angle) * radius:.2f}")
    return " ".join(points)


def star_points(cx: float, cy: float, outer: float, inner: float) -> str:
    import math

    points = []
    for index in range(10):
        radius = outer if index % 2 == 0 else inner
        angle = math.radians(-90 + index * 36)
        points.append(f"{cx + math.cos(angle) * radius:.2f},{cy + math.sin(angle) * radius:.2f}")
    return " ".join(points)


def add_light_svg_shape(svg: list[str], shape: str, color: str, rng: random.Random, index: int, count: int) -> None:
    size = 512
    spacing = size / max(count + 1, 2)
    cx = spacing * (index + 1) + rng.uniform(-28, 28)
    cy = size * 0.5 + rng.uniform(-80, 80)
    radius = rng.uniform(48, 86)
    fill = VECTOR_COLORS[color]
    stroke = "#111827"
    opacity = rng.uniform(0.78, 0.94)
    angle = rng.uniform(-28, 28)
    common = f'fill="{fill}" fill-opacity="{opacity:.3f}" stroke="{stroke}" stroke-width="5"'

    if shape == "circle":
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" {common}/>')
    elif shape == "square":
        side = radius * 1.75
        svg.append(
            f'<rect x="{cx - side / 2:.2f}" y="{cy - side / 2:.2f}" width="{side:.2f}" '
            f'height="{side:.2f}" transform="rotate({angle:.2f},{cx:.2f},{cy:.2f})" {common}/>'
        )
    elif shape == "diamond":
        svg.append(f'<polygon points="{polygon_points(cx, cy, radius, 4, 45)}" {common}/>')
    elif shape == "triangle":
        svg.append(f'<polygon points="{polygon_points(cx, cy, radius, 3)}" {common}/>')
    elif shape == "hexagon":
        svg.append(f'<polygon points="{polygon_points(cx, cy, radius, 6)}" {common}/>')
    elif shape == "pentagon":
        svg.append(f'<polygon points="{polygon_points(cx, cy, radius, 5)}" {common}/>')
    elif shape == "octagon":
        svg.append(f'<polygon points="{polygon_points(cx, cy, radius, 8)}" {common}/>')
    elif shape == "star":
        svg.append(f'<polygon points="{star_points(cx, cy, radius, radius * 0.42)}" {common}/>')
    elif shape == "cross":
        arm = radius * 0.38
        long = radius
        path = (
            f"M {cx-arm:.2f},{cy-long:.2f} H {cx+arm:.2f} V {cy-arm:.2f} "
            f"H {cx+long:.2f} V {cy+arm:.2f} H {cx+arm:.2f} "
            f"V {cy+long:.2f} H {cx-arm:.2f} V {cy+arm:.2f} "
            f"H {cx-long:.2f} V {cy-arm:.2f} H {cx-arm:.2f} Z"
        )
        svg.append(f'<path d="{path}" transform="rotate({angle:.2f},{cx:.2f},{cy:.2f})" {common}/>')
    elif shape == "heart":
        path = (
            f"M {cx:.2f},{cy + radius * 0.65:.2f} "
            f"C {cx - radius * 1.45:.2f},{cy - radius * 0.10:.2f} {cx - radius:.2f},{cy - radius:.2f} {cx:.2f},{cy - radius * 0.35:.2f} "
            f"C {cx + radius:.2f},{cy - radius:.2f} {cx + radius * 1.45:.2f},{cy - radius * 0.10:.2f} {cx:.2f},{cy + radius * 0.65:.2f} Z"
        )
        svg.append(f'<path d="{path}" {common}/>')


def generate_light_vector_svg(
    prompt: str,
    output_dir: Path,
    *,
    seed: int,
    item_count: int | None = None,
) -> Path:
    items = parse_light_vector_prompt(prompt, item_count)
    rng = random.Random(seed)
    safe_prompt = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.lower()).strip("_") or "vector"
    output_path = output_dir / f"vector_fallback_{safe_prompt[:48]}_{seed}.svg"
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<title>{escape(prompt)}</title>',
    ]
    for index, (shape, color) in enumerate(items):
        add_light_svg_shape(svg, shape, color, rng, index, len(items))
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")
    return output_path


def generate_vector_image(
    prompt: str,
    mode: str,
    *,
    seed: int,
    variation: float,
    item_count: int | None = None,
) -> GeneratedImage:
    vector_backend = "pennylane_vector"
    output_path: Path
    try:
        from VectorImage import generate_vector_svg
    except Exception:  # noqa: BLE001
        generate_vector_svg = None
        vector_backend = "lightweight_vector_fallback"

    mode_name, _ = mode_config(mode)
    steps = VECTOR_STEP_CONFIGS.get(mode_name, VECTOR_STEP_CONFIGS[DEFAULT_MODE])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if generate_vector_svg is None:
        output_path = generate_light_vector_svg(
            prompt,
            OUTPUT_DIR,
            seed=seed,
            item_count=item_count,
        )
    else:
        output_path = Path(
            generate_vector_svg(
                prompt,
                OUTPUT_DIR,
                seed=seed,
                steps=steps,
                n_samples=1,
                item_count=item_count,
            )
        )
    return GeneratedImage(
        path=output_path,
        mime_type="image/svg+xml",
        metadata={
            "matched_class": "Vector shapes",
            "class_id": "vector",
            "shots": "n/a",
            "seed": seed,
            "variation": variation,
            "detected_attributes": "n/a",
            "candidates": 1,
            "item_count": item_count or "prompt",
            "latent_scale": "n/a",
            "quantum_backend": vector_backend,
            "vector_steps": steps,
            "ibm_backend": None,
            "ibm_job_id": None,
            "ibm_status": None,
            "ibm_fallback_reason": None,
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
            "supported_vector_prompts": SUPPORTED_VECTOR_PROMPTS,
            "models": MODEL_CONFIGS,
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
    model = str(data.get("model", DEFAULT_MODEL)).strip()
    item_count_raw = data.get("item_count", 1)
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
        item_count = max(1, min(20, int(item_count_raw)))
    except (TypeError, ValueError):
        item_count = 1

    try:
        model_name, _ = model_config(model)
        if model_name == "vector":
            generated = generate_vector_image(
                prompt,
                mode,
                seed=seed,
                variation=variation,
                item_count=item_count,
            )
        else:
            generated = generate_prompt_image(prompt, mode, seed=seed, variation=variation)
        return send_png(generated)
    except IBMJobPendingError as exc:
        metadata = exc.metadata
        return jsonify(
            {
                "error": "ibm_job_pending",
                "message": str(exc),
                "prompt": prompt,
                "mode": mode,
                "model": model,
                "ibm_backend": metadata.get("ibm_backend_name"),
                "ibm_job_id": metadata.get("ibm_job_id"),
                "ibm_status": metadata.get("ibm_status"),
            }
        ), 202
    except UnsupportedPromptError as exc:
        return jsonify(
            {
                "error": "unsupported_prompt",
                "message": str(exc),
                "prompt": prompt,
                "model": model,
                "supported_prompts": SUPPORTED_PROMPTS,
                "supported_vector_prompts": SUPPORTED_VECTOR_PROMPTS,
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
                "model": model,
            }
        ), 500


if __name__ == "__main__":
    print("Starting SlopGPT QuantumTrain UI server")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    app.run(host="127.0.0.1", port=8080, debug=True)
