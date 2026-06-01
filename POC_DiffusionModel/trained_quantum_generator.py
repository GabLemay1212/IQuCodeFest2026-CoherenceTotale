"""Trained 8x8 variational quantum image generator."""

from __future__ import annotations

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

from tiny_imagenet_adapter import CACHE_DIR, match_tiny_imagenet_prompt


MODEL_PATH = CACHE_DIR / "trained_vqg_8x8_rgb.npz"
MODEL_META_PATH = CACHE_DIR / "trained_vqg_8x8_rgb_metadata.json"
TRAINED_SHAPE = (8, 8, 3)


@dataclass(frozen=True)
class TrainedQuantumResult:
    prompt: str
    class_id: str
    class_name: str
    image: np.ndarray
    target: np.ndarray
    shots: int
    train_loss: float


def _resize_rgb(image: np.ndarray, size: tuple[int, int], resample: int) -> np.ndarray:
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
    pil = pil.resize(size, resample)
    return np.asarray(pil, dtype=np.float32) / 255.0


def model_available() -> bool:
    return MODEL_PATH.exists() and MODEL_META_PATH.exists()


def load_trained_model() -> tuple[np.ndarray, list[str], list[str], list[float]]:
    if not model_available():
        raise FileNotFoundError(
            "Trained quantum model not found. Run "
            "`python POC_DiffusionModel/train_quantum_generator.py` first."
        )
    data = np.load(MODEL_PATH)
    meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    return data["theta"], meta["label_ids"], meta["class_names"], meta["losses"]


def generate_trained_quantum_for_prompt(
    prompt: str,
    *,
    shots: int = 768,
    seed: int = 7,
) -> TrainedQuantumResult | None:
    """Generate from learned 8x8 quantum parameters if prompt matches a class."""
    match = match_tiny_imagenet_prompt(prompt)
    if match is None or not model_available():
        return None

    theta, label_ids, class_names, losses = load_trained_model()
    label = match.label
    if label >= theta.shape[0]:
        return None

    simulator = AerSimulator(seed_simulator=seed + label)
    image = np.zeros(TRAINED_SHAPE, dtype=np.float32)

    for channel in range(3):
        for row in range(TRAINED_SHAPE[0]):
            qreg = QuantumRegister(TRAINED_SHAPE[1], f"q{channel}_{row}")
            creg = ClassicalRegister(TRAINED_SHAPE[1], f"c{channel}_{row}")
            circuit = QuantumCircuit(qreg, creg)

            for col in range(TRAINED_SHAPE[1]):
                circuit.ry(float(theta[label, row, col, channel]), qreg[col])

            # A light learned-state readout circuit. The model's learned RY
            # angles control the probabilities; the final image comes from
            # quantum measurements.
            circuit.measure(qreg, creg)
            counts = simulator.run(circuit, shots=shots).result().get_counts()

            ones = np.zeros(TRAINED_SHAPE[1], dtype=np.float32)
            for bitstring, count in counts.items():
                for col, bit in enumerate(bitstring[::-1]):
                    if bit == "1":
                        ones[col] += count
            image[row, :, channel] = ones / shots

    target = _resize_rgb(match.prototype, TRAINED_SHAPE[:2][::-1], Image.Resampling.BICUBIC)
    return TrainedQuantumResult(
        prompt=prompt,
        class_id=label_ids[label],
        class_name=class_names[label],
        image=np.clip(image, 0.0, 1.0),
        target=target,
        shots=shots,
        train_loss=float(losses[label]),
    )


def save_trained_quantum_report(result: TrainedQuantumResult, output_path: Path) -> None:
    """Save target vs trained quantum output as a pixel-art report."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.7))
    panels = [
        ("8x8 target", result.target),
        ("trained quantum output", result.image),
    ]
    for axis, (title, panel) in zip(axes, panels):
        enlarged = _resize_rgb(panel, (256, 256), Image.Resampling.NEAREST)
        axis.imshow(enlarged, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(
        f"{result.prompt} -> {result.class_name}\n"
        f"{result.shots} shots, train loss={result.train_loss:.5f}"
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
