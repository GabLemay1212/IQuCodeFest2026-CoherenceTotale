"""Quantum-conditioned diffusion proof of concept on sklearn Digits.

This prototype is intentionally small enough to run during a hackathon.
It uses the 8x8 grayscale Digits dataset, treats prompts such as
"generate digit 3" as text conditioning, and compares:

1. A classical label-conditioned denoising baseline.
2. A quantum-assisted variant where a Qiskit circuit produces a
   label-conditioned latent mask used during denoising.

The quantum circuit is not pretending to be a full text-to-image model.
Its role is narrow and explicit: label-conditioned probabilistic latent
sampling for the generation pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


IMAGE_SHAPE = (32, 32)
NUM_PIXELS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
LATENT_SHAPE = (8, 8)
NUM_LATENT_QUBITS = LATENT_SHAPE[0] * LATENT_SHAPE[1]

COLOR_PALETTES = {
    "blue": np.array([0.10, 0.45, 0.95]),
    "red": np.array([0.90, 0.20, 0.20]),
    "green": np.array([0.15, 0.70, 0.35]),
    "orange": np.array([0.95, 0.55, 0.10]),
    "purple": np.array([0.55, 0.25, 0.90]),
    "cyan": np.array([0.00, 0.65, 0.75]),
    "pink": np.array([0.95, 0.30, 0.65]),
    "lime": np.array([0.55, 0.80, 0.20]),
    "yellow": np.array([0.98, 0.86, 0.10]),
    "brown": np.array([0.70, 0.38, 0.16]),
    "white": np.array([0.92, 0.94, 0.96]),
}

LABEL_COLORS = list(COLOR_PALETTES.values())


@dataclass(frozen=True)
class GenerationResult:
    label: int
    prompt: str
    classical_image: np.ndarray
    quantum_image: np.ndarray
    target_prototype: np.ndarray
    quantum_mask: np.ndarray
    classical_mae: float
    quantum_mae: float
    classical_mse: float
    quantum_mse: float
    classical_pred: int
    quantum_pred: int


def load_digit_data() -> tuple[np.ndarray, np.ndarray]:
    """Load sklearn's digits dataset and upscale images to the POC canvas."""
    digits = load_digits()
    images = np.array(
        [
            np.asarray(
                Image.fromarray((image / 16.0 * 255).astype(np.uint8)).resize(
                    IMAGE_SHAPE[::-1],
                    Image.Resampling.BICUBIC,
                ),
                dtype=float,
            )
            / 255.0
            for image in digits.images
        ]
    )
    labels = digits.target.astype(int)
    return images, labels


def build_digit_prototypes(images: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Average training examples per label to get simple target prototypes."""
    prototypes: dict[int, np.ndarray] = {}
    for label in sorted(np.unique(labels)):
        prototypes[int(label)] = images[labels == label].mean(axis=0)
    return prototypes


def parse_prompt(prompt: str) -> int:
    """Extract the requested digit from a tiny text prompt."""
    for token in prompt.lower().replace(",", " ").split():
        if token.isdigit():
            label = int(token)
            if 0 <= label <= 9:
                return label
    raise ValueError(f"Prompt must contain a digit from 0 to 9, got: {prompt!r}")


def parse_color(prompt: str, default_index: int = 0) -> str:
    """Extract a supported color from free text."""
    normalized = prompt.lower().replace("-", " ").replace("_", " ")
    for color in COLOR_PALETTES:
        if color in normalized.split() or color in normalized:
            return color
    return list(COLOR_PALETTES)[default_index % len(COLOR_PALETTES)]


def _downsample_to_latent(image: np.ndarray) -> np.ndarray:
    """Downsample the current canvas to the quantum latent grid."""
    img = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
    small = img.resize(LATENT_SHAPE[::-1], Image.Resampling.BICUBIC)
    return np.asarray(small, dtype=float) / 255.0


def quantum_latent_mask(
    label: int,
    shots: int = 512,
    seed: int = 7,
    guidance_image: np.ndarray | None = None,
) -> np.ndarray:
    """Generate a 32x32 latent mask from a label-conditioned quantum circuit.

    We use an 8x8 quantum latent grid and upsample it to the 32x32 image grid.
    The prompt label and optional target prototype both change the rotation
    pattern. A shallow local entangling layer introduces correlations between
    neighboring latent qubits. Measurement frequencies become the latent mask.
    """
    rng = np.random.default_rng(seed + label)
    label_phase = (label + 1) / 10.0
    latent_guidance = _downsample_to_latent(guidance_image) if guidance_image is not None else None
    simulator = AerSimulator(seed_simulator=seed + label)
    latent_mask = np.zeros(LATENT_SHAPE, dtype=float)

    # Simulate one 8-qubit quantum row at a time. This gives a larger 8x8
    # quantum latent while keeping each circuit small enough for fast demos.
    for row in range(LATENT_SHAPE[0]):
        qreg = QuantumRegister(LATENT_SHAPE[1], f"row_{row}")
        creg = ClassicalRegister(LATENT_SHAPE[1], f"meas_{row}")
        circuit = QuantumCircuit(qreg, creg)

        for col in range(LATENT_SHAPE[1]):
            spatial_wave = 0.5 * (
                np.sin((row + 1) * label_phase) + np.cos((col + 1) * label_phase)
            )
            jitter = rng.uniform(-0.08, 0.08)
            if latent_guidance is None:
                probability = np.clip(0.50 + 0.18 * spatial_wave + jitter, 0.05, 0.95)
            else:
                target_probability = latent_guidance[row, col]
                probability = np.clip(0.78 * target_probability + 0.14 * spatial_wave + jitter, 0.03, 0.97)
            theta = 2.0 * np.arcsin(np.sqrt(probability))
            circuit.ry(theta, qreg[col])

        for col in range(LATENT_SHAPE[1] - 1):
            circuit.cx(qreg[col], qreg[col + 1])

        circuit.measure(qreg, creg)
        counts = simulator.run(circuit, shots=shots).result().get_counts()

        ones_count = np.zeros(LATENT_SHAPE[1], dtype=float)
        for bitstring, count in counts.items():
            for col, bit in enumerate(bitstring[::-1]):
                if bit == "1":
                    ones_count[col] += count
        latent_mask[row] = ones_count / shots

    latent_img = Image.fromarray((latent_mask * 255).astype(np.uint8))
    upsampled = latent_img.resize(IMAGE_SHAPE[::-1], Image.Resampling.BICUBIC)
    return np.asarray(upsampled, dtype=float) / 255.0


def colorize_image(
    image: np.ndarray,
    label: int | None = None,
    color_name: str | None = None,
) -> np.ndarray:
    """Turn a grayscale image into RGB for friendlier visual outputs."""
    accent = COLOR_PALETTES.get(color_name or "", LABEL_COLORS[(label or 0) % len(LABEL_COLORS)])
    background = np.array([0.04, 0.05, 0.07])
    value = np.clip(np.asarray(image, dtype=float), 0.0, 1.0)[..., None]
    rgb = background * (1.0 - value) + accent * value
    return np.clip(rgb, 0.0, 1.0)


def denoise_from_prompt(
    prototype: np.ndarray,
    conditioning: np.ndarray,
    *,
    steps: int = 20,
    seed: int = 7,
    conditioning_weight: float = 0.22,
) -> np.ndarray:
    """Tiny diffusion-like denoising process.

    This is a compact educational approximation, not a trained DDPM. It starts
    from Gaussian noise and repeatedly moves toward a prompt-conditioned target.
    """
    rng = np.random.default_rng(seed)
    image = rng.normal(loc=0.5, scale=0.35, size=IMAGE_SHAPE)

    prototype_weight = 1.0 - conditioning_weight
    conditioned_target = np.clip(
        prototype_weight * prototype + conditioning_weight * conditioning,
        0.0,
        1.0,
    )
    for step in range(steps):
        progress = (step + 1) / steps
        noise_scale = 0.12 * (1.0 - progress)
        image = 0.82 * image + 0.18 * conditioned_target
        image += rng.normal(0.0, noise_scale, size=IMAGE_SHAPE)
        image = np.clip(image, 0.0, 1.0)

    return image


def train_evaluator(images: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Train a simple classifier used only to evaluate generated prompt accuracy."""
    x_train, x_test, y_train, y_test = train_test_split(
        images.reshape(len(images), -1),
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    clf = LogisticRegression(max_iter=3000, solver="lbfgs")
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    print(f"Evaluator accuracy on held-out real digits: {accuracy_score(y_test, preds):.3f}")
    return clf


def generate_for_prompt(
    prompt: str,
    prototypes: dict[int, np.ndarray],
    evaluator: LogisticRegression,
    *,
    shots: int,
    steps: int,
    seed: int,
) -> GenerationResult:
    """Generate classical and quantum-assisted images for one prompt."""
    label = parse_prompt(prompt)
    prototype = prototypes[label]

    classical_conditioning = np.full(IMAGE_SHAPE, fill_value=(label + 1) / 11.0)
    quantum_conditioning = quantum_latent_mask(label, shots=shots, seed=seed, guidance_image=prototype)

    classical_image = denoise_from_prompt(
        prototype,
        classical_conditioning,
        steps=steps,
        seed=seed + 100 + label,
        conditioning_weight=0.10,
    )
    quantum_image = denoise_from_prompt(
        prototype,
        quantum_conditioning,
        steps=steps,
        seed=seed + 200 + label,
        conditioning_weight=0.42,
    )

    classical_pred = int(evaluator.predict(classical_image.reshape(1, -1))[0])
    quantum_pred = int(evaluator.predict(quantum_image.reshape(1, -1))[0])

    return GenerationResult(
        label=label,
        prompt=prompt,
        classical_image=classical_image,
        quantum_image=quantum_image,
        target_prototype=prototype,
        quantum_mask=quantum_conditioning,
        classical_mae=float(mean_absolute_error(prototype.ravel(), classical_image.ravel())),
        quantum_mae=float(mean_absolute_error(prototype.ravel(), quantum_image.ravel())),
        classical_mse=float(mean_squared_error(prototype.ravel(), classical_image.ravel())),
        quantum_mse=float(mean_squared_error(prototype.ravel(), quantum_image.ravel())),
        classical_pred=classical_pred,
        quantum_pred=quantum_pred,
    )


def save_visual_report(results: list[GenerationResult], output_path: Path) -> None:
    """Save a grid comparing target, baseline, quantum mask, and quantum output."""
    rows = len(results)
    fig, axes = plt.subplots(rows, 4, figsize=(9, 2.35 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = ["Dataset prototype", "Classical baseline", "Quantum latent", "Quantum output"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title)

    for row, result in enumerate(results):
        panels = [
            result.target_prototype,
            result.classical_image,
            result.quantum_mask,
            result.quantum_image,
        ]
        for col, panel in enumerate(panels):
            axes[row, col].imshow(colorize_image(panel, result.label), interpolation="nearest")
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(
            f"{result.prompt}\n"
            f"C pred={result.classical_pred}, Q pred={result.quantum_pred}\n"
            f"C MAE={result.classical_mae:.3f}, Q MAE={result.quantum_mae:.3f}",
            rotation=0,
            labelpad=68,
            va="center",
        )

    fig.suptitle("Quantum-conditioned diffusion POC on sklearn Digits", y=0.995)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_metrics(results: list[GenerationResult], output_path: Path) -> None:
    """Persist metrics so the run is easy to inspect without rerunning plots."""
    payload = [
        {
            "prompt": result.prompt,
            "label": result.label,
            "classical_prediction": result.classical_pred,
            "quantum_prediction": result.quantum_pred,
            "classical_mae": result.classical_mae,
            "quantum_mae": result.quantum_mae,
            "classical_mse": result.classical_mse,
            "quantum_mse": result.quantum_mse,
        }
        for result in results
    ]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["generate digit 0", "generate digit 3", "generate digit 8", "generate digit 9"],
        help="Text prompts. Each prompt must include a digit from 0 to 9.",
    )
    parser.add_argument("--shots", type=int, default=512, help="Quantum simulation shots.")
    parser.add_argument("--steps", type=int, default=20, help="Denoising steps.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for generated plots and metrics.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    images, labels = load_digit_data()
    prototypes = build_digit_prototypes(images, labels)
    evaluator = train_evaluator(images, labels)

    results = [
        generate_for_prompt(
            prompt,
            prototypes,
            evaluator,
            shots=args.shots,
            steps=args.steps,
            seed=args.seed,
        )
        for prompt in args.prompts
    ]

    report_path = args.output_dir / "quantum_diffusion_report.png"
    metrics_path = args.output_dir / "metrics.json"
    save_visual_report(results, report_path)
    save_metrics(results, metrics_path)

    print("\nGeneration summary:")
    for result in results:
        print(
            f"- {result.prompt!r}: "
            f"classical pred={result.classical_pred}, quantum pred={result.quantum_pred}, "
            f"classical MAE={result.classical_mae:.3f}, quantum MAE={result.quantum_mae:.3f}"
        )
    print(f"\nSaved visual report to: {report_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
