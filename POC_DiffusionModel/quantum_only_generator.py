"""Quantum-only prompt image generator.

This module removes the classical denoising/baseline path. The prompt is used
only to choose circuit parameters. Pixel intensities come from measurement
probabilities produced by Qiskit simulations.

For practicality, the 32x32 image is generated as 32 row circuits of 32 qubits
each. A single 1024-qubit image circuit is not realistic on a local simulator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

from quantum_diffusion_poc import COLOR_PALETTES, IMAGE_SHAPE, colorize_image, parse_color
from shape_diffusion import SHAPE_TO_ID, parse_shape_prompt


@dataclass(frozen=True)
class QuantumOnlyResult:
    prompt: str
    target: str
    color: str
    image: np.ndarray
    shots: int
    depth: int
    circuits: int
    qubits_per_circuit: int


def _prompt_seed(prompt: str) -> int:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _target_from_prompt(prompt: str) -> tuple[str, int]:
    try:
        shape = parse_shape_prompt(prompt)
        return shape, SHAPE_TO_ID[shape]
    except ValueError:
        pass

    for token in prompt.lower().replace(",", " ").split():
        if token.isdigit():
            digit = int(token)
            if 0 <= digit <= 9:
                return f"digit_{digit}", digit

    return "abstract", _prompt_seed(prompt) % 10


def _shape_probability(target: str, x: float, y: float, rng_value: float) -> float:
    """Map a normalized coordinate to a target probability."""
    cx = cy = 0.5
    dx = x - cx
    dy = y - cy

    if target == "circle":
        distance = np.sqrt(dx * dx + dy * dy)
        return 0.88 if distance < 0.30 else 0.10
    if target == "square":
        return 0.88 if abs(dx) < 0.30 and abs(dy) < 0.30 else 0.10
    if target == "triangle":
        inside = y > 0.20 and y < 0.78 and abs(dx) < (y - 0.15) * 0.62
        return 0.88 if inside else 0.10
    if target == "diamond":
        return 0.88 if abs(dx) + abs(dy) < 0.38 else 0.10
    if target == "star":
        angle = np.arctan2(dy, dx)
        radius = np.sqrt(dx * dx + dy * dy)
        boundary = 0.23 + 0.12 * (0.5 + 0.5 * np.cos(5.0 * angle))
        return 0.90 if radius < boundary else 0.08
    if target in {"plus", "cross"}:
        return 0.88 if abs(dx) < 0.11 or abs(dy) < 0.11 else 0.10
    if target == "x":
        return 0.88 if abs(abs(dx) - abs(dy)) < 0.08 else 0.10
    if target.startswith("digit_"):
        digit = int(target.split("_", 1)[1])
        wave = 0.5 + 0.5 * np.sin((digit + 2) * np.pi * x + (digit + 1) * np.pi * y)
        ring = np.exp(-((np.sqrt(dx * dx + dy * dy) - 0.28) ** 2) / 0.018)
        return float(np.clip(0.12 + 0.48 * wave + 0.36 * ring, 0.04, 0.94))

    wave_a = 0.5 + 0.5 * np.sin((2.0 + 4.0 * rng_value) * np.pi * x)
    wave_b = 0.5 + 0.5 * np.cos((3.0 + 3.0 * rng_value) * np.pi * y)
    blob = np.exp(-((dx * dx + dy * dy) / (0.05 + 0.10 * rng_value)))
    return float(np.clip(0.12 + 0.28 * wave_a + 0.24 * wave_b + 0.30 * blob, 0.04, 0.94))


def generate_quantum_only_for_prompt(
    prompt: str,
    *,
    shots: int = 512,
    depth: int = 3,
    seed: int = 7,
) -> QuantumOnlyResult:
    """Generate a 32x32 image using only quantum measurement probabilities."""
    target, target_index = _target_from_prompt(prompt)
    color = parse_color(prompt, default_index=target_index)
    prompt_seed = _prompt_seed(prompt)
    rng = np.random.default_rng(seed + prompt_seed)
    simulator = AerSimulator(method="matrix_product_state", seed_simulator=seed + prompt_seed)

    rows, cols = IMAGE_SHAPE
    image = np.zeros(IMAGE_SHAPE, dtype=float)

    for row in range(rows):
        qreg = QuantumRegister(cols, f"qrow_{row}")
        creg = ClassicalRegister(cols, f"crow_{row}")
        circuit = QuantumCircuit(qreg, creg)

        y = row / max(rows - 1, 1)
        row_jitter = rng.uniform(0.0, 1.0)
        for col in range(cols):
            x = col / max(cols - 1, 1)
            probability = _shape_probability(target, x, y, row_jitter)
            probability = float(np.clip(probability + rng.uniform(-0.035, 0.035), 0.02, 0.98))
            theta = 2.0 * np.arcsin(np.sqrt(probability))
            circuit.ry(theta, qreg[col])
            circuit.rz((target_index + 1) * (x + y + 0.1) * np.pi / 3.0, qreg[col])

        for layer in range(depth):
            offset = layer % 2
            for col in range(offset, cols - 1, 2):
                circuit.cx(qreg[col], qreg[col + 1])
            for col in range(cols):
                phase = (layer + 1) * (target_index + 1) * (col + 1) / cols
                circuit.rx(0.08 * np.pi * np.sin(phase + row_jitter), qreg[col])

        circuit.measure(qreg, creg)
        counts = simulator.run(circuit, shots=shots).result().get_counts()

        ones = np.zeros(cols, dtype=float)
        for bitstring, count in counts.items():
            for col, bit in enumerate(bitstring[::-1]):
                if bit == "1":
                    ones[col] += count
        image[row] = ones / shots

    return QuantumOnlyResult(
        prompt=prompt,
        target=target,
        color=color,
        image=np.clip(image, 0.0, 1.0),
        shots=shots,
        depth=depth,
        circuits=rows,
        qubits_per_circuit=cols,
    )


def save_quantum_only_report(result: QuantumOnlyResult, output_path: Path) -> None:
    """Save a chat-friendly PNG containing only the quantum-generated image."""
    fig, axis = plt.subplots(1, 1, figsize=(5.2, 5.2))
    axis.imshow(colorize_image(result.image, color_name=result.color), interpolation="nearest")
    axis.set_title(f"{result.prompt}\nquantum-only: {result.target}, {result.shots} shots, depth {result.depth}")
    axis.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_quantum_only_metrics(result: QuantumOnlyResult, output_path: Path) -> None:
    payload = {
        "prompt": result.prompt,
        "target": result.target,
        "color": result.color,
        "shots": result.shots,
        "depth": result.depth,
        "circuits": result.circuits,
        "qubits_per_circuit": result.qubits_per_circuit,
        "mean_intensity": float(np.mean(result.image)),
        "std_intensity": float(np.std(result.image)),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
