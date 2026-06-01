"""Prompt-driven shape generation for the quantum diffusion POC."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from quantum_diffusion_poc import (
    IMAGE_SHAPE,
    colorize_image,
    denoise_from_prompt,
    parse_color,
    quantum_latent_mask,
)


SHAPE_NAMES = [
    "circle",
    "square",
    "triangle",
    "diamond",
    "star",
    "cross",
    "plus",
    "x",
]

SHAPE_TO_ID = {name: idx for idx, name in enumerate(SHAPE_NAMES)}


@dataclass(frozen=True)
class ShapeGenerationResult:
    prompt: str
    shape: str
    color: str
    prototype: np.ndarray
    classical_image: np.ndarray
    quantum_mask: np.ndarray
    quantum_image: np.ndarray
    classical_mae: float
    quantum_mae: float


def parse_shape_prompt(prompt: str) -> str:
    """Find a supported shape name in free text."""
    normalized = prompt.lower().replace("-", " ").replace("_", " ")
    aliases = {
        "rhombus": "diamond",
        "pentagram": "star",
        "plus sign": "plus",
        "letter x": "x",
    }
    for alias, canonical in aliases.items():
        if alias in normalized:
            return canonical

    for shape in SHAPE_NAMES:
        if shape in normalized.split() or shape in normalized:
            return shape

    supported = ", ".join(SHAPE_NAMES)
    raise ValueError(f"Prompt must include one supported shape: {supported}.")


def _star_points(center: tuple[float, float], outer: float, inner: float) -> list[tuple[float, float]]:
    cx, cy = center
    points = []
    for i in range(10):
        radius = outer if i % 2 == 0 else inner
        angle = -np.pi / 2 + i * np.pi / 5
        points.append((cx + radius * np.cos(angle), cy + radius * np.sin(angle)))
    return points


def render_shape_prototype(shape: str, size: int = IMAGE_SHAPE[0], high_res: int = 128) -> np.ndarray:
    """Render a clean grayscale shape and downsample it to the POC canvas."""
    canvas = Image.new("L", (high_res, high_res), 0)
    draw = ImageDraw.Draw(canvas)
    pad = int(high_res * 0.18)
    box = (pad, pad, high_res - pad, high_res - pad)

    if shape == "circle":
        draw.ellipse(box, fill=255)
    elif shape == "square":
        draw.rectangle(box, fill=255)
    elif shape == "triangle":
        draw.polygon(
            [
                (high_res / 2, pad),
                (high_res - pad, high_res - pad),
                (pad, high_res - pad),
            ],
            fill=255,
        )
    elif shape == "diamond":
        draw.polygon(
            [
                (high_res / 2, pad),
                (high_res - pad, high_res / 2),
                (high_res / 2, high_res - pad),
                (pad, high_res / 2),
            ],
            fill=255,
        )
    elif shape == "star":
        draw.polygon(_star_points((high_res / 2, high_res / 2), high_res * 0.36, high_res * 0.16), fill=255)
    elif shape in {"cross", "plus"}:
        bar = int(high_res * 0.18)
        mid = high_res // 2
        draw.rectangle((mid - bar, pad, mid + bar, high_res - pad), fill=255)
        draw.rectangle((pad, mid - bar, high_res - pad, mid + bar), fill=255)
    elif shape == "x":
        width = int(high_res * 0.18)
        draw.line((pad, pad, high_res - pad, high_res - pad), fill=255, width=width)
        draw.line((high_res - pad, pad, pad, high_res - pad), fill=255, width=width)
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    small = canvas.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(small, dtype=float) / 255.0


def generate_shape_for_prompt(
    prompt: str,
    *,
    shots: int = 512,
    steps: int = 20,
    seed: int = 7,
) -> ShapeGenerationResult:
    """Generate a shape image from a prompt."""
    shape = parse_shape_prompt(prompt)
    shape_id = SHAPE_TO_ID[shape]
    color = parse_color(prompt, default_index=shape_id)
    prototype = render_shape_prototype(shape)
    quantum_mask = quantum_latent_mask(
        shape_id,
        shots=shots,
        seed=seed + 1000,
        guidance_image=prototype,
    )
    classical_conditioning = np.full(IMAGE_SHAPE, fill_value=(shape_id + 1) / (len(SHAPE_NAMES) + 1))

    classical_image = denoise_from_prompt(
        prototype,
        classical_conditioning,
        steps=steps,
        seed=seed + 300 + shape_id,
        conditioning_weight=0.10,
    )
    quantum_image = denoise_from_prompt(
        prototype,
        quantum_mask,
        steps=steps,
        seed=seed + 400 + shape_id,
        conditioning_weight=0.42,
    )

    return ShapeGenerationResult(
        prompt=prompt,
        shape=shape,
        color=color,
        prototype=prototype,
        classical_image=classical_image,
        quantum_mask=quantum_mask,
        quantum_image=quantum_image,
        classical_mae=float(np.mean(np.abs(prototype - classical_image))),
        quantum_mae=float(np.mean(np.abs(prototype - quantum_image))),
    )


def save_shape_visual_report(result: ShapeGenerationResult, output_path: Path) -> None:
    """Save a one-prompt shape generation report."""
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.4))
    panels = [
        ("Shape prototype", result.prototype),
        ("Classical baseline", result.classical_image),
        ("Quantum latent", result.quantum_mask),
        ("Quantum output", result.quantum_image),
    ]

    for axis, (title, panel) in zip(axes, panels):
        axis.imshow(
            colorize_image(panel, SHAPE_TO_ID[result.shape], result.color),
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(
        f"{result.prompt} -> {result.shape} | "
        f"C MAE={result.classical_mae:.3f}, Q MAE={result.quantum_mae:.3f}"
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_shape_metrics(result: ShapeGenerationResult, output_path: Path) -> None:
    payload = {
        "prompt": result.prompt,
        "shape": result.shape,
        "color": result.color,
        "classical_mae": result.classical_mae,
        "quantum_mae": result.quantum_mae,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
