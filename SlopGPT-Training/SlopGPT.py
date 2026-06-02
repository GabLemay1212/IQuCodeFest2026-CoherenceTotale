"""Flask server for the quantum image generation chat UI.

The UI sends:

POST /generate
{
    "prompt": "goldfish",
    "mode": "balanced"
}

and receives a PNG image response.

This training UI intentionally uses only the Fashion-MNIST 16x16 grayscale model.
Older Tiny ImageNet/FRQI/POC generators remain in the repo, but unsupported
prompts return a clear error instead of silently falling through to another
image format.
"""

from __future__ import annotations

import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent

# Allows imports whether this server is inside POC_DiffusionModel or SlopGPT-Training.
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT_DIR / "POC_DiffusionModel"))
sys.path.insert(0, str(ROOT_DIR / "SlopGPT-Training"))


# ---------------------------------------------------------------------
# Legacy / fallback generators
# ---------------------------------------------------------------------

from quantum_only_generator import (  # noqa: E402
    generate_quantum_only_for_prompt,
    save_quantum_only_metrics,
    save_quantum_only_report,
)

from latent_quantum_generator import (  # noqa: E402
    generate_latent_quantum_for_prompt,
    save_latent_quantum_report,
)

from trained_quantum_generator import (  # noqa: E402
    generate_trained_quantum_for_prompt,
    save_trained_quantum_report,
)


# ---------------------------------------------------------------------
# Fashion-MNIST 16x16 grayscale model
# ---------------------------------------------------------------------

FASHION_MODEL_IMPORT_ERROR: Exception | None = None

try:
    from fashion_mnist_generate import (  # noqa: E402
        PROMPT_ALIASES,
        generate as generate_fashion_mnist,
        match_prompt as match_fashion_mnist_prompt,
        model_available as fashion_mnist_available,
        save_images as save_fashion_mnist_images,
    )
except Exception as exc:  # noqa: BLE001
    FASHION_MODEL_IMPORT_ERROR = exc
    PROMPT_ALIASES = {}
    generate_fashion_mnist = None
    match_fashion_mnist_prompt = None
    fashion_mnist_available = None
    save_fashion_mnist_images = None


# ---------------------------------------------------------------------
# FRQI 64x64 trained model
# ---------------------------------------------------------------------

FRQI_MODEL_IMPORT_ERROR: Exception | None = None

try:
    from frqi_generate import (  # noqa: E402
        generate as generate_frqi_64,
        model_available as frqi_64_available,
        save_report as save_frqi_64_report,
    )
except Exception as exc:  # noqa: BLE001
    FRQI_MODEL_IMPORT_ERROR = exc
    generate_frqi_64 = None
    frqi_64_available = None
    save_frqi_64_report = None


# ---------------------------------------------------------------------
# Full Tiny ImageNet trained model
# ---------------------------------------------------------------------

FULL_MODEL_IMPORT_ERROR: Exception | None = None

try:
    from full_model_generate import (  # noqa: E402
        generate as generate_full_tiny_imagenet,
        model_available as full_tiny_imagenet_available,
        save_report as save_full_tiny_imagenet_report,
    )
except Exception as exc:  # noqa: BLE001
    FULL_MODEL_IMPORT_ERROR = exc
    generate_full_tiny_imagenet = None
    full_tiny_imagenet_available = None
    save_full_tiny_imagenet_report = None


# ---------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------

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
                "X-Class-ID",
                "X-Seed",
                "X-Shots",
                "X-Quantum-Backend",
                "X-Ibm-Backend",
                "X-Ibm-Job-Id",
                "X-Output-Path",
            ],
        }
    },
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

OUTPUT_DIR = THIS_DIR / "outputs" / "ui"

MODE_CONFIGS = {
    "fast": {
        "shots": 128,
        "latent_scale": 0.7,
        "candidates": 1,
        "backend": "simulator",
    },
    "balanced": {
        "shots": 512,
        "latent_scale": 1.0,
        "candidates": 2,
        "backend": "simulator",
    },
    "deepthinking": {
        "shots": 1024,
        "latent_scale": 1.7,
        "candidates": 24,
        "backend": "simulator",
    },
    "realquantumdemo": {
        "shots": 128,
        "latent_scale": 1.0,
        "candidates": 1,
        "backend": "ibm",
    },
}
SHOT_MODES = {name: config["shots"] for name, config in MODE_CONFIGS.items()}

DEFAULT_SHOT_MODE = "balanced"
DEFAULT_PROMPT = "sneaker"
QUANTUM_ONLY_DEPTH = 4

SUPPORTED_FASHION_PROMPTS = (
    "t-shirt, tshirt, shirt, trouser, pants, pullover, dress, coat, "
    "sandal, sneaker, shoe, bag, ankle boot, boot"
)


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GeneratedImage:
    path: Path
    generator: str
    metadata: dict[str, Any]


class UnsupportedPromptError(ValueError):
    """Raised when a prompt is outside the Fashion-MNIST vocabulary."""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_name(prompt: str) -> str:
    """Convert a prompt into a safe filename fragment."""
    name = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.lower()).strip("_")
    return name[:80] or "generated_image"


def _config_for_mode(mode: str | None) -> tuple[str, dict[str, float | int]]:
    """Normalize mode and return generation settings."""
    if not mode:
        return DEFAULT_SHOT_MODE, MODE_CONFIGS[DEFAULT_SHOT_MODE]

    normalized = re.sub(r"[^a-zA-Z]", "", mode.lower())

    if normalized not in MODE_CONFIGS:
        normalized = DEFAULT_SHOT_MODE

    return normalized, MODE_CONFIGS[normalized]


def _request_json() -> dict[str, Any]:
    """Read request JSON safely."""
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return {}

    return data


def _save_clean_grayscale_image(image: np.ndarray, output_path: Path, *, scale: int = 40) -> None:
    """Save a clean enlarged PNG for the UI without matplotlib titles."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = np.clip(image, 0.0, 1.0)
    pixel_image = Image.fromarray((image * 255).astype(np.uint8), mode="L")

    height, width = image.shape
    pixel_image = pixel_image.resize(
        (width * scale, height * scale),
        Image.Resampling.NEAREST,
    )

    pixel_image.save(output_path)


def _send_png(generated: GeneratedImage):
    """Send PNG with useful no-cache and debug headers."""
    response = send_file(generated.path, mimetype="image/png")

    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    response.headers["X-Generator"] = generated.generator
    response.headers["X-Output-Path"] = str(generated.path)

    for key, value in generated.metadata.items():
        if value is not None:
            header_name = "X-" + key.replace("_", "-").title()
            response.headers[header_name] = str(value)

    return response


# ---------------------------------------------------------------------
# Generator pipeline
# ---------------------------------------------------------------------

def _try_full_tiny_imagenet(
    prompt: str,
    *,
    mode_name: str,
    shots: int,
    seed: int | None,
    debug_report: bool,
) -> GeneratedImage | None:
    """Try the full Tiny ImageNet trained model first."""
    if generate_full_tiny_imagenet is None:
        return None

    if full_tiny_imagenet_available is not None and not full_tiny_imagenet_available():
        return None

    sample = generate_full_tiny_imagenet(prompt, shots=shots, seed=seed)

    if sample is None:
        return None

    safe_prompt = _safe_name(prompt)

    if debug_report:
        output_path = (
            OUTPUT_DIR
            / f"{safe_prompt}_{mode_name}_full_tiny_imagenet_seed{sample.latent_seed}_report.png"
        )
        save_full_tiny_imagenet_report(sample, output_path)
    else:
        output_path = (
            OUTPUT_DIR
            / f"{safe_prompt}_{mode_name}_full_tiny_imagenet_seed{sample.latent_seed}.png"
        )
        _save_clean_grayscale_image(sample.image, output_path)

    return GeneratedImage(
        path=output_path,
        generator="full_tiny_imagenet_vqg",
        metadata={
            "matched_class": sample.class_name,
            "class_id": sample.class_id,
            "seed": sample.latent_seed,
            "shots": sample.shots,
        },
    )


def _try_fashion_mnist(
    prompt: str,
    *,
    mode_name: str,
    shots: int,
    latent_scale: float,
    candidates: int,
    backend: str,
    seed: int | None,
) -> GeneratedImage | None:
    """Try the Fashion-MNIST 16x16 grayscale checkpoint first."""
    if generate_fashion_mnist is None or save_fashion_mnist_images is None:
        return None

    if fashion_mnist_available is not None and not fashion_mnist_available():
        return None

    sample = generate_fashion_mnist(
        prompt,
        shots=shots,
        seed=seed,
        latent_scale=latent_scale,
        candidates=candidates,
        backend=backend,
    )
    if sample is None:
        return None

    safe_prompt = _safe_name(prompt)
    output_prefix = (
        OUTPUT_DIR
        / f"{safe_prompt}_{mode_name}_fashion16_seed{sample.seed}"
    )
    grayscale_path, binary_debug_path = save_fashion_mnist_images(sample, output_prefix)

    return GeneratedImage(
        path=grayscale_path,
        generator="fashion_mnist_16x16_grayscale_vqg",
        metadata={
            "matched_class": sample.class_name,
            "class_id": sample.label,
            "seed": sample.seed,
            "shots": sample.shots,
            "latent_scale": sample.latent_scale,
            "candidates": sample.candidates,
            "candidate_score": f"{sample.candidate_score:.4f}",
            "quantum_backend": sample.quantum_backend,
            "ibm_backend": sample.ibm_backend_name,
            "ibm_job_id": sample.ibm_job_id,
            "checkpoint_step": sample.checkpoint_step,
            "binary_debug_path": binary_debug_path,
        },
    )


def _try_frqi_64(
    prompt: str,
    *,
    mode_name: str,
    seed: int | None,
) -> GeneratedImage | None:
    """Try the FRQI 64x64 RGB checkpoint first."""
    if generate_frqi_64 is None:
        return None

    if frqi_64_available is not None and not frqi_64_available():
        return None

    sample = generate_frqi_64(prompt, seed=seed)

    if sample is None:
        return None

    safe_prompt = _safe_name(prompt)
    output_path = (
        OUTPUT_DIR
        / f"{safe_prompt}_{mode_name}_frqi64_seed{sample.latent_seed}_report.png"
    )
    save_frqi_64_report(sample, output_path)

    return GeneratedImage(
        path=output_path,
        generator="frqi_64x64_rgb_vqg",
        metadata={
            "matched_class": sample.class_name,
            "class_id": sample.class_id,
            "seed": sample.latent_seed,
            "frqi_qubits_per_channel": sample.frqi_qubits_per_channel,
        },
    )


def _try_latent_quantum(prompt: str, *, mode_name: str, shots: int) -> GeneratedImage | None:
    """Try the older latent quantum generator."""
    latent_result = generate_latent_quantum_for_prompt(prompt, shots=shots)

    if latent_result is None:
        return None

    safe_prompt = _safe_name(prompt)
    output_path = OUTPUT_DIR / f"{safe_prompt}_{mode_name}_latent_vqg_report.png"

    save_latent_quantum_report(latent_result, output_path)

    return GeneratedImage(
        path=output_path,
        generator="legacy_latent_vqg",
        metadata={
            "shots": shots,
        },
    )


def _try_trained_quantum(
    prompt: str,
    *,
    mode_name: str,
    shots: int,
    seed: int,
) -> GeneratedImage | None:
    """Try the older trained quantum generator."""
    trained_result = generate_trained_quantum_for_prompt(prompt, shots=shots, seed=seed)

    if trained_result is None:
        return None

    safe_prompt = _safe_name(prompt)
    output_path = OUTPUT_DIR / f"{safe_prompt}_{mode_name}_trained_vqg_report.png"

    save_trained_quantum_report(trained_result, output_path)

    return GeneratedImage(
        path=output_path,
        generator="legacy_trained_vqg",
        metadata={
            "seed": seed,
            "shots": shots,
        },
    )


def _quantum_only_fallback(
    prompt: str,
    *,
    mode_name: str,
    shots: int,
    seed: int,
) -> GeneratedImage:
    """Final fallback. Always tries to produce something."""
    result = generate_quantum_only_for_prompt(
        prompt,
        shots=shots,
        depth=QUANTUM_ONLY_DEPTH,
        seed=seed,
    )

    safe_prompt = _safe_name(prompt)

    image_path = OUTPUT_DIR / f"{safe_prompt}_{mode_name}_quantum_only_report.png"
    metrics_path = OUTPUT_DIR / f"{safe_prompt}_{mode_name}_quantum_only_metrics.json"

    save_quantum_only_report(result, image_path)
    save_quantum_only_metrics(result, metrics_path)

    return GeneratedImage(
        path=image_path,
        generator="quantum_only_fallback",
        metadata={
            "seed": seed,
            "shots": shots,
            "metrics_path": metrics_path,
        },
    )


def generate_prompt_image(
    prompt: str,
    mode: str = DEFAULT_SHOT_MODE,
    *,
    seed: int | None = None,
    debug_report: bool = False,
) -> GeneratedImage:
    """Generate a Fashion-MNIST image for the UI prompt."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mode_name, mode_config = _config_for_mode(mode)
    shots = int(mode_config["shots"])
    latent_scale = float(mode_config["latent_scale"])
    candidates = int(mode_config["candidates"])
    backend = str(mode_config.get("backend", "simulator"))

    if match_fashion_mnist_prompt is None:
        raise RuntimeError(
            "Fashion-MNIST generator could not be imported. "
            f"Import error: {FASHION_MODEL_IMPORT_ERROR!r}"
        )

    if match_fashion_mnist_prompt(prompt) is None:
        raise UnsupportedPromptError(
            "Unsupported prompt. Try one of: "
            f"{SUPPORTED_FASHION_PROMPTS}."
        )

    if fashion_mnist_available is not None and not fashion_mnist_available():
        raise FileNotFoundError(
            "Fashion-MNIST checkpoint not found. Train first with "
            "python SlopGPT-Training/train_fashion_mnist_quantum_16.py "
            "--epochs 1 --batch-size 128 --max-steps 100 "
            "--data-dir \"C:\\Personal Files\\Universite\\Quantum\\IQuCodeFest2026-CoherenceTotale\\Fashion-MNIST\""
        )

    fashion_result = _try_fashion_mnist(
        prompt,
        mode_name=mode_name,
        shots=shots,
        latent_scale=latent_scale,
        candidates=candidates,
        backend=backend,
        seed=seed,
    )

    if fashion_result is not None:
        return fashion_result

    raise RuntimeError(
        "Fashion-MNIST generation failed even though the prompt and checkpoint were valid."
    )


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.route("/ping", methods=["GET"])
def ping():
    fashion_ready = False
    frqi_ready = False
    full_model_ready = False

    if fashion_mnist_available is not None:
        try:
            fashion_ready = bool(fashion_mnist_available())
        except Exception:  # noqa: BLE001
            fashion_ready = False

    if frqi_64_available is not None:
        try:
            frqi_ready = bool(frqi_64_available())
        except Exception:  # noqa: BLE001
            frqi_ready = False

    if full_tiny_imagenet_available is not None:
        try:
            full_model_ready = bool(full_tiny_imagenet_available())
        except Exception:  # noqa: BLE001
            full_model_ready = False

    return jsonify(
        {
            "status": "ok",
            "output_dir": str(OUTPUT_DIR),
            "fashion_mnist_imported": generate_fashion_mnist is not None,
            "fashion_mnist_available": fashion_ready,
            "fashion_mnist_import_error": (
                repr(FASHION_MODEL_IMPORT_ERROR) if FASHION_MODEL_IMPORT_ERROR else None
            ),
            "frqi_64_imported": generate_frqi_64 is not None,
            "frqi_64_available": frqi_ready,
            "frqi_64_import_error": (
                repr(FRQI_MODEL_IMPORT_ERROR) if FRQI_MODEL_IMPORT_ERROR else None
            ),
            "full_tiny_imagenet_imported": generate_full_tiny_imagenet is not None,
            "full_tiny_imagenet_available": full_model_ready,
            "full_tiny_imagenet_import_error": (
                repr(FULL_MODEL_IMPORT_ERROR) if FULL_MODEL_IMPORT_ERROR else None
            ),
            "supported_prompts": SUPPORTED_FASHION_PROMPTS,
            "modes": SHOT_MODES,
            "mode_configs": MODE_CONFIGS,
        }
    ), 200


@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return "", 204

    data = _request_json()

    prompt = str(data.get("prompt", "")).strip()
    mode = str(data.get("mode", DEFAULT_SHOT_MODE)).strip()

    # Optional:
    # - seed: fixed seed for repeatability
    # - debug_report: true to show prompt/class/shots title on image
    seed_raw = data.get("seed", None)
    debug_report = bool(data.get("debug_report", False))

    if not prompt:
        prompt = DEFAULT_PROMPT

    seed: int | None
    if seed_raw is None or seed_raw == "":
        seed = None
    else:
        try:
            seed = int(seed_raw)
        except ValueError:
            return jsonify({"error": "seed must be an integer"}), 400

    try:
        generated = generate_prompt_image(
            prompt,
            mode,
            seed=seed,
            debug_report=debug_report,
        )

        return _send_png(generated)

    except UnsupportedPromptError as exc:
        return jsonify(
            {
                "error": "unsupported_prompt",
                "message": str(exc),
                "prompt": prompt,
                "supported_prompts": SUPPORTED_FASHION_PROMPTS,
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting quantum image generation UI server")
    print(f"Output directory: {OUTPUT_DIR}")

    if FULL_MODEL_IMPORT_ERROR:
        print("Full Tiny ImageNet model import failed:")
        print(repr(FULL_MODEL_IMPORT_ERROR))

    if FASHION_MODEL_IMPORT_ERROR:
        print("Fashion-MNIST model import failed:")
        print(repr(FASHION_MODEL_IMPORT_ERROR))

    if FRQI_MODEL_IMPORT_ERROR:
        print("FRQI 64x64 model import failed:")
        print(repr(FRQI_MODEL_IMPORT_ERROR))

    app.run(host="127.0.0.1", port=8080, debug=True)
