"""Generate samples from the full Tiny ImageNet latent quantum checkpoint."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "POC_DiffusionModel"))

from tiny_imagenet_adapter import match_tiny_imagenet_prompt  # noqa: E402
from train_full_tiny_imagenet_quantum import CHECKPOINT_PATH, IMAGE_SIZE, META_PATH  # noqa: E402


@dataclass(frozen=True)
class FullModelSample:
    prompt: str
    class_id: str
    class_name: str
    image: np.ndarray
    shots: int
    latent_seed: int


def model_available() -> bool:
    return CHECKPOINT_PATH.exists() and META_PATH.exists()


def generate(prompt: str, *, shots: int = 768, seed: int | None = None) -> FullModelSample | None:
    if not model_available():
        return None

    match = match_tiny_imagenet_prompt(prompt)
    if match is None:
        return None

    data = np.load(CHECKPOINT_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    class_bias = data["class_bias"]
    weights = data["weights"]
    latent_dim = int(meta["latent_dim"])

    latent_seed = int(seed if seed is not None else np.random.default_rng().integers(0, 2**31 - 1))
    rng = np.random.default_rng(latent_seed)
    latent = rng.normal(0.0, 1.0, size=(latent_dim,)).astype(np.float32)

    phi = class_bias[match.label] + np.einsum("d,dhw->hw", latent, weights[match.label])
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

    return FullModelSample(
        prompt=prompt,
        class_id=match.class_id,
        class_name=match.class_name,
        image=np.clip(image, 0.0, 1.0),
        shots=shots,
        latent_seed=latent_seed,
    )


def save_report(sample: FullModelSample, output_path: Path) -> None:
    enlarged = Image.fromarray((sample.image * 255).astype(np.uint8), mode="L")
    enlarged = enlarged.resize((320, 320), Image.Resampling.NEAREST)

    fig, axis = plt.subplots(1, 1, figsize=(4.8, 5.2))
    axis.imshow(np.asarray(enlarged), cmap="gray", vmin=0, vmax=255)
    axis.set_title(
        f"{sample.prompt} -> {sample.class_name}\n"
        f"full Tiny ImageNet VQG, {sample.shots} shots, seed {sample.latent_seed}"
    )
    axis.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
