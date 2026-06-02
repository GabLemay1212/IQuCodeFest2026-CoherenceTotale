"""Generate a 16x16 Fashion-MNIST sample from the quantum-sim-trained checkpoint."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from qiskit_aer import AerSimulator

from train_quantum_sim_fashion_mnist_16 import (
    CHECKPOINT_PATH,
    FASHION_CLASSES,
    IMAGE_SIZE,
    OUT_DIR,
    compose_angles,
    counts_to_row,
    row_circuit,
)


HARDCODED_IBM_QUANTUM_TOKEN = "DrKh1NpZA9Y_h-7rztu1DhGwIe6xbRIBxqw7eLuCQdl7"
HARDCODED_IBM_QUANTUM_INSTANCE = (
    "crn:v1:bluemix:public:quantum-computing:us-east:"
    "a/d2c50f33c43a44abb94280706332351d:"
    "fa6ae649-f03a-4eb4-9434-1c3f512203fe::"
)
HARDCODED_IBM_QUANTUM_BACKEND = ""


PROMPT_ALIASES = {
    "t-shirt": 0,
    "tshirt": 0,
    "tee": 0,
    "top": 0,
    "trouser": 1,
    "trousers": 1,
    "pants": 1,
    "pullover": 2,
    "dress": 3,
    "coat": 4,
    "jacket": 4,
    "sandal": 5,
    "shirt": 6,
    "sneaker": 7,
    "shoe": 7,
    "bag": 8,
    "ankle boot": 9,
    "boot": 9,
}

ATTRIBUTE_ALIASES = {
    "futuristic": {"futuristic", "future", "cyber", "cyberpunk", "tech", "sci-fi", "scifi"},
    "cool": {"cool", "stylish", "modern", "fresh"},
    "tall": {"tall", "high", "long"},
    "small": {"small", "tiny", "mini", "little", "compact"},
    "short": {"short", "low", "stubby"},
    "chunky": {"chunky", "wide", "thick", "bulky", "large"},
    "slim": {"slim", "thin", "narrow", "sleek"},
    "heavy": {"heavy", "combat", "rugged", "sturdy"},
    "simple": {"simple", "plain", "minimal", "clean"},
    "sharp": {"sharp", "pointy", "angular", "edgy"},
}


@dataclass(frozen=True)
class GenerationResult:
    image: np.ndarray
    seed: int
    quantum_backend: str
    ibm_backend_name: str | None = None
    ibm_job_id: str | None = None
    ibm_status: str | None = None
    ibm_fallback_reason: str | None = None


class IBMJobPendingError(RuntimeError):
    """Raised when an IBM job was submitted but did not finish in request time."""

    def __init__(self, metadata: dict[str, str | None]):
        self.metadata = metadata
        super().__init__(
            metadata.get("ibm_fallback_reason")
            or "IBM quantum job is still pending or running."
        )


def match_prompt(prompt: str) -> tuple[int, str]:
    lowered = prompt.lower().replace("_", " ")
    for alias in sorted(PROMPT_ALIASES, key=len, reverse=True):
        if alias in lowered:
            label = PROMPT_ALIASES[alias]
            return label, FASHION_CLASSES[label]
    raise ValueError(
        "Unsupported prompt. Try: "
        + ", ".join(sorted(PROMPT_ALIASES, key=lambda item: (PROMPT_ALIASES[item], item)))
    )


def safe_name(prompt: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", prompt.lower()).strip("_") or "sample"


def parse_prompt_attributes(prompt: str) -> dict[str, object]:
    """Map descriptive prompt words to simple visual-conditioning flags.

    This is not full natural-language understanding. It is a small keyword
    parser that keeps the Fashion-MNIST class match, then modifies the quantum
    angle matrix before simulator execution.
    """
    lowered = prompt.lower().replace("_", " ")
    detected: list[str] = []
    for attribute, aliases in ATTRIBUTE_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            detected.append(attribute)
    return {
        "detected": detected,
        "flags": {name: True for name in detected},
    }


def attributes_to_string(attributes: dict[str, object]) -> str:
    detected = attributes.get("detected", [])
    if not isinstance(detected, list) or not detected:
        return "none"
    return ", ".join(str(item) for item in detected)


def angles_to_probabilities(angles: np.ndarray) -> np.ndarray:
    return np.clip(np.sin(angles / 2.0) ** 2, 0.0, 1.0).astype(np.float32)


def probabilities_to_angles(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, 0.02, 0.98)
    return (2.0 * np.arcsin(np.sqrt(probabilities))).astype(np.float32)


def smooth_image(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1, mode="edge")
    result = np.zeros_like(image)
    for y_offset in range(3):
        for x_offset in range(3):
            result += padded[y_offset : y_offset + IMAGE_SIZE, x_offset : x_offset + IMAGE_SIZE]
    return result / 9.0


def resize_axis(image: np.ndarray, *, width: int | None = None, height: int | None = None) -> np.ndarray:
    target_width = int(width if width is not None else image.shape[1])
    target_height = int(height if height is not None else image.shape[0])
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    resized = pil.resize((target_width, target_height), Image.Resampling.BICUBIC)
    resized_arr = np.asarray(resized, dtype=np.float32) / 255.0
    output = np.zeros_like(image)
    y_start = max(0, (IMAGE_SIZE - target_height) // 2)
    x_start = max(0, (IMAGE_SIZE - target_width) // 2)
    crop_y = max(0, (target_height - IMAGE_SIZE) // 2)
    crop_x = max(0, (target_width - IMAGE_SIZE) // 2)
    paste_h = min(IMAGE_SIZE, target_height)
    paste_w = min(IMAGE_SIZE, target_width)
    output[y_start : y_start + paste_h, x_start : x_start + paste_w] = resized_arr[
        crop_y : crop_y + paste_h,
        crop_x : crop_x + paste_w,
    ]
    return np.clip(output, 0.0, 1.0)


def center_scale_matrix(matrix: np.ndarray, scale_y: float, scale_x: float) -> np.ndarray:
    target_h = max(3, int(round(IMAGE_SIZE * scale_y)))
    target_w = max(3, int(round(IMAGE_SIZE * scale_x)))
    return resize_axis(matrix, width=target_w, height=target_h)


def vertical_stretch_matrix(matrix: np.ndarray, factor: float) -> np.ndarray:
    return center_scale_matrix(matrix, scale_y=factor, scale_x=1.0)


def horizontal_stretch_matrix(matrix: np.ndarray, factor: float) -> np.ndarray:
    return center_scale_matrix(matrix, scale_y=1.0, scale_x=factor)


def shift_mass_up(matrix: np.ndarray, amount: int) -> np.ndarray:
    shifted = np.roll(matrix, -amount, axis=0)
    shifted[-amount:, :] = 0.0
    return shifted


def shift_mass_down(matrix: np.ndarray, amount: int) -> np.ndarray:
    shifted = np.roll(matrix, amount, axis=0)
    shifted[:amount, :] = 0.0
    return shifted


def thicken_matrix(matrix: np.ndarray, strength: float = 1.0) -> np.ndarray:
    thick = np.maximum.reduce(
        [
            matrix,
            np.roll(matrix, 1, axis=1),
            np.roll(matrix, -1, axis=1),
            np.roll(matrix, 1, axis=0),
            np.roll(matrix, -1, axis=0),
        ]
    )
    return np.clip((1.0 - strength) * matrix + strength * thick, 0.0, 1.0)


def thin_matrix(matrix: np.ndarray, strength: float = 0.75) -> np.ndarray:
    narrow = center_scale_matrix(matrix, scale_y=1.0, scale_x=0.62)
    side_mask = np.ones_like(matrix)
    side_mask[:, :3] = 0.45
    side_mask[:, -3:] = 0.45
    narrow *= side_mask
    return np.clip((1.0 - strength) * matrix + strength * narrow, 0.0, 1.0)


def contrast_matrix(matrix: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(0.5 + factor * (matrix - 0.5), 0.0, 1.0)


def add_futuristic_panels(matrix: np.ndarray, strength: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.indices(matrix.shape)
    vertical = ((x % 4) == 0).astype(np.float32)
    diagonal = (((x + y) % 5) == 0).astype(np.float32)
    horizontal = ((y % 5) == 1).astype(np.float32)
    panels = (0.55 * vertical + 0.35 * diagonal + 0.25 * horizontal)
    panels *= np.clip(matrix + 0.15, 0.0, 1.0)
    jitter = rng.normal(0.0, 0.08 * strength, size=matrix.shape).astype(np.float32)
    sharpened = matrix + 1.15 * (matrix - smooth_image(matrix))
    return np.clip(sharpened + strength * 0.28 * panels + jitter, 0.0, 1.0)


def apply_prompt_attributes(
    angles: np.ndarray,
    attributes: dict[str, object],
    *,
    seed: int,
    variation: float = 0.05,
) -> np.ndarray:
    """Return a conditioned angle matrix without mutating checkpoint data."""
    flags = attributes.get("flags", {})
    if not isinstance(flags, dict):
        flags = {}

    rng = np.random.default_rng(seed)
    variation = max(0.0, float(variation))
    probs = angles_to_probabilities(angles)

    if flags.get("tall"):
        probs = vertical_stretch_matrix(probs, 1.38)
        probs = shift_mass_up(probs, 1)
        row_ramp = np.linspace(0.16, -0.04, IMAGE_SIZE, dtype=np.float32)[:, None]
        probs = contrast_matrix(probs + row_ramp, 1.08)

    if flags.get("small"):
        probs = center_scale_matrix(probs, scale_y=0.68, scale_x=0.68)
        probs = probs * 0.86

    if flags.get("short"):
        probs = center_scale_matrix(probs, scale_y=0.62, scale_x=1.0)
        probs = shift_mass_down(probs, 1)
        probs = probs * 0.92

    if flags.get("chunky"):
        probs = horizontal_stretch_matrix(probs, 1.28)
        probs = thicken_matrix(probs, 0.95)
        probs = contrast_matrix(probs * 1.12, 1.18)

    if flags.get("slim"):
        probs = thin_matrix(probs, 0.92)
        probs = contrast_matrix(probs * 0.94, 1.06)

    if flags.get("heavy"):
        probs = thicken_matrix(probs, 0.75)
        lower_ramp = np.linspace(-0.08, 0.28, IMAGE_SIZE, dtype=np.float32)[:, None]
        probs = shift_mass_down(probs, 1) + lower_ramp
        probs = contrast_matrix(probs, 1.12)

    if flags.get("simple"):
        probs = smooth_image(smooth_image(probs))
        probs = 0.5 + 0.72 * (probs - 0.5)

    if flags.get("cool"):
        probs = contrast_matrix(probs, 1.22)
        probs += rng.normal(0.0, variation * 0.35, size=probs.shape).astype(np.float32)

    if flags.get("futuristic"):
        probs = contrast_matrix(probs, 1.35)
        probs = add_futuristic_panels(probs, strength=1.0 + variation, seed=seed)

    if flags.get("sharp"):
        probs = contrast_matrix(probs + 1.1 * (probs - smooth_image(probs)), 1.12)

    if variation > 0 and not flags.get("simple"):
        probs += rng.normal(0.0, variation * 0.12, size=probs.shape).astype(np.float32)

    return np.clip(probabilities_to_angles(probs), 0.02, np.pi - 0.02)


def score_image(image: np.ndarray) -> float:
    foreground = image > 0.2
    foreground_ratio = float(foreground.mean())
    contrast = float(image.std())
    center_mass = float(image[3:13, 3:13].mean())
    border_mass = float(
        np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]]).mean()
    )
    return (1.7 * contrast) + (0.8 * center_mass) - (0.8 * border_mass) - abs(foreground_ratio - 0.34)


def simulate_phi(phi: np.ndarray, *, shots: int, seed: int) -> np.ndarray:
    simulator = AerSimulator(seed_simulator=seed)
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for row in range(IMAGE_SIZE):
        circuit = row_circuit(phi[row], row)
        counts = simulator.run(circuit, shots=shots).result().get_counts()
        image[row] = counts_to_row(counts, shots)
    return np.clip(image, 0.0, 1.0)


def status_to_string(job: object) -> str:
    try:
        status = job.status()
    except Exception:  # noqa: BLE001
        return "unknown"
    return getattr(status, "name", str(status))


def ibm_probability_image(
    phi: np.ndarray,
    *,
    shots: int,
    timeout_seconds: float,
) -> tuple[np.ndarray | None, dict[str, str | None]]:
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "RealQuantumDemo requires qiskit-ibm-runtime. Install it with "
            "pip install qiskit-ibm-runtime."
        ) from exc

    token = os.environ.get("IBM_QUANTUM_TOKEN") or HARDCODED_IBM_QUANTUM_TOKEN
    instance = os.environ.get("IBM_QUANTUM_INSTANCE") or HARDCODED_IBM_QUANTUM_INSTANCE
    backend_name = os.environ.get("IBM_QUANTUM_BACKEND") or HARDCODED_IBM_QUANTUM_BACKEND

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
    )
    backend = (
        service.backend(backend_name)
        if backend_name
        else service.least_busy(operational=True, simulator=False, min_num_qubits=IMAGE_SIZE)
    )

    circuits = [row_circuit(phi[row], row) for row in range(IMAGE_SIZE)]
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = pass_manager.run(circuits)

    sampler = Sampler(mode=backend)
    job = sampler.run(isa_circuits, shots=shots)
    metadata = {
        "ibm_backend_name": backend.name,
        "ibm_job_id": job.job_id(),
        "ibm_status": status_to_string(job),
        "ibm_fallback_reason": None,
    }

    try:
        result = job.result(timeout=timeout_seconds)
    except Exception as exc:  # noqa: BLE001
        metadata["ibm_status"] = status_to_string(job)
        metadata["ibm_fallback_reason"] = (
            f"IBM job did not finish within {timeout_seconds:g}s; "
            "no image was returned because the hardware result is not ready yet."
        )
        return None, metadata

    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for row, pub_result in enumerate(result):
        counts = pub_result.data.c.get_counts()
        image[row] = counts_to_row(counts, shots)
    metadata["ibm_status"] = status_to_string(job)
    return np.clip(image, 0.0, 1.0), metadata


def generate_result(
    label: int,
    *,
    shots: int,
    seed: int | None,
    prompt: str = "",
    variation: float = 0.05,
    latent_scale: float = 1.0,
    candidates: int = 1,
    backend: str = "simulator",
    ibm_timeout_seconds: float = 45.0,
) -> GenerationResult:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}. Train first.")

    data = np.load(CHECKPOINT_PATH)
    base_angles = data["angles"].astype(np.float32)
    if "basis" in data:
        basis = data["basis"].astype(np.float32)
    else:
        basis = np.zeros((base_angles.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    base_seed = int(seed if seed is not None else np.random.default_rng().integers(0, 2**31 - 1))
    seed_rng = np.random.default_rng(base_seed)
    candidates = max(1, int(candidates))
    latent_scale = max(0.0, float(latent_scale))
    attributes = parse_prompt_attributes(prompt)
    backend = backend.lower().strip()
    best_image: np.ndarray | None = None
    best_score = -float("inf")
    best_seed = base_seed
    best_metadata: dict[str, str | None] = {}

    if backend in {"ibm", "real", "realquantum"}:
        candidates = 1

    for candidate_index in range(candidates):
        sample_seed = base_seed if candidate_index == 0 else int(seed_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sample_seed)
        latent = rng.normal(0.0, latent_scale, size=(basis.shape[1],)).astype(np.float32)
        phi = compose_angles(base_angles, basis, label, latent)
        phi = apply_prompt_attributes(phi, attributes, seed=sample_seed, variation=variation)
        if backend in {"ibm", "real", "realquantum"}:
            image, best_metadata = ibm_probability_image(
                phi,
                shots=shots,
                timeout_seconds=ibm_timeout_seconds,
            )
            if image is None:
                raise IBMJobPendingError(best_metadata)
            score = score_image(image)
        else:
            image = simulate_phi(phi, shots=shots, seed=sample_seed)
            score = score_image(image)
        if score > best_score:
            best_score = score
            best_image = image
            best_seed = sample_seed

    if best_image is None:
        raise RuntimeError("No candidate image was generated.")

    if backend in {"ibm", "real", "realquantum"}:
        quantum_backend = "ibm_quantum" if not best_metadata.get("ibm_fallback_reason") else "aer_simulator_fallback"
    else:
        quantum_backend = "aer_simulator"

    return GenerationResult(
        image=best_image,
        seed=best_seed,
        quantum_backend=quantum_backend,
        ibm_backend_name=best_metadata.get("ibm_backend_name"),
        ibm_job_id=best_metadata.get("ibm_job_id"),
        ibm_status=best_metadata.get("ibm_status"),
        ibm_fallback_reason=best_metadata.get("ibm_fallback_reason"),
    )


def generate_image(
    label: int,
    *,
    shots: int,
    seed: int | None,
    prompt: str = "",
    variation: float = 0.05,
    latent_scale: float = 1.0,
    candidates: int = 1,
) -> np.ndarray:
    return generate_result(
        label,
        shots=shots,
        seed=seed,
        prompt=prompt,
        variation=variation,
        latent_scale=latent_scale,
        candidates=candidates,
    ).image


def save_image(image: np.ndarray, path: Path, *, scale: int = 18) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray((image * 255).astype(np.uint8), mode="L")
    pil = pil.resize((image.shape[1] * scale, image.shape[0] * scale), Image.Resampling.NEAREST)
    pil.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="sneaker")
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--variation", type=float, default=0.05)
    parser.add_argument("--latent-scale", type=float, default=1.0)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--backend", choices=("simulator", "ibm"), default="simulator")
    parser.add_argument("--ibm-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label, class_name = match_prompt(args.prompt)
    seed = int(args.seed if args.seed is not None else np.random.default_rng().integers(0, 2**31 - 1))
    attributes = parse_prompt_attributes(args.prompt)
    try:
        result = generate_result(
            label,
            shots=args.shots,
            seed=seed,
            prompt=args.prompt,
            variation=args.variation,
            latent_scale=args.latent_scale,
            candidates=args.candidates,
            backend=args.backend,
            ibm_timeout_seconds=args.ibm_timeout_seconds,
        )
    except IBMJobPendingError as exc:
        metadata = exc.metadata
        print(f"matched class: {class_name} ({label})")
        print(f"detected attributes: {attributes_to_string(attributes)}")
        print(f"seed: {seed}")
        print(f"variation: {args.variation}")
        print(f"shots: {args.shots}")
        print("ibm job pending")
        print(f"ibm backend: {metadata.get('ibm_backend_name')}")
        print(f"ibm job id: {metadata.get('ibm_job_id')}")
        print(f"ibm status: {metadata.get('ibm_status')}")
        print(str(exc))
        return
    path = args.output_dir / f"{safe_name(args.prompt)}_quantum_sim_trained_{args.shots}shots.png"
    save_image(result.image, path)
    print(f"matched class: {class_name} ({label})")
    print(f"detected attributes: {attributes_to_string(attributes)}")
    print(f"seed: {result.seed}")
    print(f"variation: {args.variation}")
    print(f"shots: {args.shots}")
    print(f"quantum backend: {result.quantum_backend}")
    if result.ibm_job_id:
        print(f"ibm backend: {result.ibm_backend_name}")
        print(f"ibm job id: {result.ibm_job_id}")
        print(f"ibm status: {result.ibm_status}")
    if result.ibm_fallback_reason:
        print(f"fallback: {result.ibm_fallback_reason}")
    print(f"generated: {path}")


if __name__ == "__main__":
    main()
