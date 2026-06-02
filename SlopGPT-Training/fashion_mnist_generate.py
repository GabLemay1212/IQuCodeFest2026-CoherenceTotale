"""Generate 16x16 grayscale Fashion-MNIST samples from the trained checkpoint."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

from train_fashion_mnist_quantum_16 import (
    CHECKPOINT_PATH,
    FASHION_CLASSES,
    IMAGE_SIZE,
    META_PATH,
    THRESHOLD,
)


PROMPT_ALIASES = {
    "t-shirt": 0,
    "tshirt": 0,
    "tee": 0,
    "top": 0,
    "trouser": 1,
    "trousers": 1,
    "pants": 1,
    "pant": 1,
    "pullover": 2,
    "sweater": 2,
    "dress": 3,
    "coat": 4,
    "jacket": 4,
    "sandal": 5,
    "sandals": 5,
    "shirt": 6,
    "sneaker": 7,
    "sneakers": 7,
    "shoe": 7,
    "shoes": 7,
    "bag": 8,
    "purse": 8,
    "ankle boot": 9,
    "boot": 9,
    "boots": 9,
}


@dataclass(frozen=True)
class FashionSample:
    prompt: str
    label: int
    class_name: str
    probability_image: np.ndarray
    binary_image: np.ndarray
    shots: int
    seed: int
    checkpoint_step: int
    latent_scale: float = 1.0
    candidates: int = 1
    candidate_score: float = 0.0
    quantum_backend: str = "aer_simulator"
    ibm_backend_name: str | None = None
    ibm_job_id: str | None = None


def model_available() -> bool:
    return CHECKPOINT_PATH.exists() and META_PATH.exists()


def match_prompt(prompt: str) -> tuple[int, str] | None:
    lowered = prompt.lower().replace("_", " ").replace("-", "-")
    for alias in sorted(PROMPT_ALIASES, key=len, reverse=True):
        if alias in lowered:
            label = PROMPT_ALIASES[alias]
            return label, FASHION_CLASSES[label]
    return None


def _score_candidate(image: np.ndarray) -> float:
    """Prefer recognizable, non-blank grayscale samples."""
    foreground = image > 0.22
    foreground_ratio = float(foreground.mean())
    contrast = float(image.std())
    center_mass = float(image[3:13, 3:13].mean())
    border_mass = float(
        np.concatenate(
            [
                image[0, :],
                image[-1, :],
                image[:, 0],
                image[:, -1],
            ]
        ).mean()
    )
    blank_penalty = abs(foreground_ratio - 0.34)
    return (1.8 * contrast) + (0.8 * center_mass) - (0.9 * border_mass) - blank_penalty


def _row_circuit(phi_row: np.ndarray, row: int) -> QuantumCircuit:
    qreg = QuantumRegister(IMAGE_SIZE, f"q{row}")
    creg = ClassicalRegister(IMAGE_SIZE, "c")
    circuit = QuantumCircuit(qreg, creg)
    for col in range(IMAGE_SIZE):
        circuit.ry(float(phi_row[col]), qreg[col])
    circuit.measure(qreg, creg)
    return circuit


def _counts_to_row(counts: dict[str, int], shots: int) -> np.ndarray:
    ones = np.zeros(IMAGE_SIZE, dtype=np.float32)
    for bitstring, count in counts.items():
        for col, bit in enumerate(bitstring[::-1]):
            if bit == "1":
                ones[col] += count
    return ones / shots


def _simulate_probability_image(phi: np.ndarray, *, shots: int, seed: int) -> np.ndarray:
    simulator = AerSimulator(seed_simulator=seed)
    probability_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for row in range(IMAGE_SIZE):
        circuit = _row_circuit(phi[row], row)
        counts = simulator.run(circuit, shots=shots).result().get_counts()
        probability_image[row] = _counts_to_row(counts, shots)

    return np.clip(probability_image, 0.0, 1.0)


def _ibm_probability_image(
    phi: np.ndarray,
    *,
    shots: int,
) -> tuple[np.ndarray, str, str]:
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "RealQuantumDemo requires qiskit-ibm-runtime. Install it with "
            "pip install qiskit-ibm-runtime."
        ) from exc

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE")
    backend_name = os.environ.get("IBM_QUANTUM_BACKEND")

    if not token:
        raise RuntimeError("Set IBM_QUANTUM_TOKEN before using RealQuantumDemo.")
    if not instance:
        raise RuntimeError(
            "Set IBM_QUANTUM_INSTANCE before using RealQuantumDemo. Use your IBM Quantum "
            "instance CRN from the Instances page."
        )

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
    )
    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=IMAGE_SIZE,
        )

    circuits = [_row_circuit(phi[row], row) for row in range(IMAGE_SIZE)]
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = pass_manager.run(circuits)

    sampler = Sampler(mode=backend)
    job = sampler.run(isa_circuits, shots=shots)
    result = job.result()

    probability_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for row, pub_result in enumerate(result):
        counts = pub_result.data.c.get_counts()
        probability_image[row] = _counts_to_row(counts, shots)

    return np.clip(probability_image, 0.0, 1.0), backend.name, job.job_id()


def generate(
    prompt: str,
    *,
    shots: int = 768,
    seed: int | None = None,
    latent_scale: float = 1.0,
    candidates: int = 1,
    backend: str = "simulator",
) -> FashionSample | None:
    match = match_prompt(prompt)
    if match is None or not model_available():
        return None

    label, class_name = match
    data = np.load(CHECKPOINT_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    class_bias = data["class_bias"]
    weights = data["weights"]
    latent_dim = int(meta["latent_dim"])
    checkpoint_step = int(data["step"])

    base_seed = int(seed if seed is not None else np.random.default_rng().integers(0, 2**31 - 1))
    candidates = max(1, int(candidates))
    latent_scale = max(0.0, float(latent_scale))
    seed_rng = np.random.default_rng(base_seed)

    best_image: np.ndarray | None = None
    best_seed = base_seed
    best_score = -float("inf")
    best_backend_name = None
    best_job_id = None
    backend = backend.lower().strip()

    for candidate_index in range(candidates):
        if candidate_index == 0:
            sample_seed = base_seed
        else:
            sample_seed = int(seed_rng.integers(0, 2**31 - 1))

        rng = np.random.default_rng(sample_seed)
        latent = rng.normal(0.0, latent_scale, size=(latent_dim,)).astype(np.float32)
        phi = class_bias[label] + np.einsum("d,dhw->hw", latent, weights[label])
        phi = np.clip(phi, 0.02, np.pi - 0.02)
        if backend in {"ibm", "real", "realquantum"}:
            candidate_image, backend_name, job_id = _ibm_probability_image(phi, shots=shots)
        else:
            candidate_image = _simulate_probability_image(phi, shots=shots, seed=sample_seed)
            backend_name = "aer_simulator"
            job_id = None
        candidate_score = _score_candidate(candidate_image)

        if candidate_score > best_score:
            best_image = candidate_image
            best_seed = sample_seed
            best_score = candidate_score
            best_backend_name = backend_name
            best_job_id = job_id

    if best_image is None:
        return None

    probability_image = np.clip(best_image, 0.0, 1.0)
    binary_image = (probability_image > THRESHOLD).astype(np.float32)
    return FashionSample(
        prompt=prompt,
        label=label,
        class_name=class_name,
        probability_image=probability_image,
        binary_image=binary_image,
        shots=shots,
        seed=best_seed,
        checkpoint_step=checkpoint_step,
        latent_scale=latent_scale,
        candidates=candidates,
        candidate_score=best_score,
        quantum_backend="ibm_quantum" if backend in {"ibm", "real", "realquantum"} else "aer_simulator",
        ibm_backend_name=best_backend_name,
        ibm_job_id=best_job_id,
    )


def _nearest_enlarge(image: np.ndarray, scale: int = 18) -> np.ndarray:
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    pil = pil.resize((image.shape[1] * scale, image.shape[0] * scale), Image.Resampling.NEAREST)
    return np.asarray(pil)


def save_report(sample: FashionSample, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2))
    panels = [
        ("grayscale probabilities", sample.probability_image),
        ("debug threshold", sample.binary_image),
    ]
    for axis, (title, panel) in zip(axes, panels):
        axis.imshow(_nearest_enlarge(panel), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(
        f"{sample.prompt} -> {sample.class_name}\n"
        f"Fashion-MNIST 16x16 grayscale, {sample.shots} shots, "
        f"{sample.candidates} candidates, latent scale {sample.latent_scale:g}, "
        f"seed {sample.seed}, step {sample.checkpoint_step}"
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_images(sample: FashionSample, output_prefix: Path) -> tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    grayscale_path = output_prefix.with_name(output_prefix.name + "_grayscale.png")
    binary_debug_path = output_prefix.with_name(output_prefix.name + "_binary_debug.png")
    Image.fromarray(_nearest_enlarge(sample.probability_image), mode="L").save(grayscale_path)
    Image.fromarray(_nearest_enlarge(sample.binary_image), mode="L").save(binary_debug_path)
    return grayscale_path, binary_debug_path
