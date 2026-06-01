"""Runtime for the latent conditional quantum generator."""

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

from tiny_imagenet_adapter import match_tiny_imagenet_prompt
from train_latent_quantum_generator import IMAGE_SIZE, LATENT_DIM, MODEL_META_PATH, MODEL_PATH


@dataclass(frozen=True)
class LatentQuantumResult:
    prompt: str
    class_name: str
    class_id: str
    image: np.ndarray
    shots: int
    latent_seed: int


def latent_model_available() -> bool:
    return MODEL_PATH.exists() and MODEL_META_PATH.exists()


def _load_model():
    if not latent_model_available():
        raise FileNotFoundError(
            "Latent quantum model not found. Run "
            "`python POC_DiffusionModel/train_latent_quantum_generator.py` first."
        )
    data = np.load(MODEL_PATH)
    meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    return data["class_bias"], data["weights"], meta


def generate_latent_quantum_for_prompt(
    prompt: str,
    *,
    shots: int = 768,
    seed: int | None = None,
) -> LatentQuantumResult | None:
    """Generate a varied 8x8 grayscale image from class + random latent."""
    match = match_tiny_imagenet_prompt(prompt)
    if match is None or not latent_model_available():
        return None

    class_bias, weights, meta = _load_model()
    if match.label not in meta["labels"]:
        return None

    local_class_index = meta["labels"].index(match.label)
    latent_seed = int(seed if seed is not None else np.random.default_rng().integers(0, 2**31 - 1))
    rng = np.random.default_rng(latent_seed)
    latent = rng.normal(0.0, 1.0, size=(LATENT_DIM,)).astype(np.float32)

    phi = class_bias[local_class_index] + np.einsum("d,dhw->hw", latent, weights[local_class_index])
    phi = np.clip(phi, 0.02, np.pi - 0.02)

    simulator = AerSimulator(seed_simulator=latent_seed)
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for row in range(IMAGE_SIZE):
        qreg = QuantumRegister(IMAGE_SIZE, f"q{row}")
        creg = ClassicalRegister(IMAGE_SIZE, f"c{row}")
        circuit = QuantumCircuit(qreg, creg)
        for col in range(IMAGE_SIZE):
            circuit.ry(float(phi[row, col]), qreg[col])
        circuit.measure(qreg, creg)
        counts = simulator.run(circuit, shots=shots).result().get_counts()

        ones = np.zeros(IMAGE_SIZE, dtype=np.float32)
        for bitstring, count in counts.items():
            for col, bit in enumerate(bitstring[::-1]):
                if bit == "1":
                    ones[col] += count
        image[row] = ones / shots

    return LatentQuantumResult(
        prompt=prompt,
        class_name=match.class_name,
        class_id=match.class_id,
        image=np.clip(image, 0.0, 1.0),
        shots=shots,
        latent_seed=latent_seed,
    )


def save_latent_quantum_report(result: LatentQuantumResult, output_path: Path) -> None:
    """Save one generated sample enlarged as pixel art."""
    enlarged = Image.fromarray((result.image * 255).astype(np.uint8), mode="L")
    enlarged = enlarged.resize((320, 320), Image.Resampling.NEAREST)

    fig, axis = plt.subplots(1, 1, figsize=(4.8, 5.2))
    axis.imshow(np.asarray(enlarged), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axis.set_title(
        f"{result.prompt} -> {result.class_name}\n"
        f"latent trained quantum sample, {result.shots} shots, seed {result.latent_seed}"
    )
    axis.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
