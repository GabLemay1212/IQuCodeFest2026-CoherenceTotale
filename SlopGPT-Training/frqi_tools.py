"""FRQI utilities for 64x64 image experiments.

FRQI stores pixel location in position qubits and pixel intensity in a color
qubit rotation:

    |I> = 1/sqrt(N) sum_i |i> (cos(theta_i)|0> + sin(theta_i)|1>)

For a 64x64 image:

    N = 4096 pixels
    position qubits = log2(4096) = 12
    color qubit = 1
    total per grayscale channel = 13 qubits

For RGB we use three FRQI circuits, one per color channel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


FRQI_SIZE = 64
FRQI_PIXELS = FRQI_SIZE * FRQI_SIZE
FRQI_POSITION_QUBITS = int(math.log2(FRQI_PIXELS))
FRQI_TOTAL_QUBITS = FRQI_POSITION_QUBITS + 1


@dataclass(frozen=True)
class FRQIReconstruction:
    image: np.ndarray
    shots: int
    observed_pixels: int
    position_qubits: int = FRQI_POSITION_QUBITS
    total_qubits: int = FRQI_TOTAL_QUBITS


def intensity_to_theta(intensity: np.ndarray) -> np.ndarray:
    """Convert normalized intensity [0, 1] to FRQI theta."""
    clipped = np.clip(np.asarray(intensity, dtype=np.float32), 0.0, 1.0)
    return np.arcsin(np.sqrt(clipped)).astype(np.float32)


def theta_to_intensity(theta: np.ndarray) -> np.ndarray:
    """Convert FRQI theta to normalized intensity [0, 1]."""
    return (np.sin(theta) ** 2).astype(np.float32)


def _apply_x_mask(qc: QuantumCircuit, address_bitstring: str, position_qubits: list[int]) -> None:
    for bit_idx, bit in enumerate(reversed(address_bitstring)):
        if bit == "0":
            qc.x(position_qubits[bit_idx])


def build_frqi_channel_circuit(channel: np.ndarray, *, skip_near_zero: bool = True) -> QuantumCircuit:
    """Build an FRQI circuit for one 64x64 normalized channel.

    This is exact but deep: one controlled rotation per nonzero pixel.
    Use for demos or selected generated samples, not inside every training step.
    """
    flat = np.asarray(channel, dtype=np.float32).reshape(-1)
    if flat.size != FRQI_PIXELS:
        raise ValueError(f"Expected {FRQI_PIXELS} pixels, got {flat.size}.")

    position_qubits = list(range(FRQI_POSITION_QUBITS))
    color_qubit = FRQI_POSITION_QUBITS
    qc = QuantumCircuit(FRQI_TOTAL_QUBITS, FRQI_TOTAL_QUBITS)
    qc.h(position_qubits)

    theta = intensity_to_theta(flat)
    for idx, angle in enumerate(theta):
        if skip_near_zero and angle < 1e-4:
            continue
        phi = 2.0 * float(angle)
        address = format(idx, f"0{FRQI_POSITION_QUBITS}b")
        _apply_x_mask(qc, address, position_qubits)
        qc.mcry(phi, position_qubits, color_qubit)
        _apply_x_mask(qc, address, position_qubits)

    qc.measure(range(FRQI_TOTAL_QUBITS), range(FRQI_TOTAL_QUBITS))
    return qc


def reconstruct_channel_from_counts(counts: dict[str, int], shots: int) -> FRQIReconstruction:
    """Reconstruct channel intensities with the per-address ratio method."""
    total = np.zeros(FRQI_PIXELS, dtype=np.int64)
    ones = np.zeros(FRQI_PIXELS, dtype=np.int64)

    for outcome, count in counts.items():
        bits = outcome.replace(" ", "")
        # Qiskit classical bit order is displayed high-to-low. Since we measured
        # qubit i into classical bit i, the color bit is the leftmost bit.
        color_bit = bits[0]
        position_bits = bits[1:]
        idx = int(position_bits, 2)
        if idx < FRQI_PIXELS:
            total[idx] += count
            if color_bit == "1":
                ones[idx] += count

    image = np.zeros(FRQI_PIXELS, dtype=np.float32)
    observed = total > 0
    image[observed] = ones[observed] / total[observed]
    return FRQIReconstruction(
        image=image.reshape(FRQI_SIZE, FRQI_SIZE),
        shots=shots,
        observed_pixels=int(np.sum(observed)),
    )


def simulate_frqi_channel(channel: np.ndarray, *, shots: int = 262_144) -> FRQIReconstruction:
    """Build and simulate a single FRQI channel circuit."""
    qc = build_frqi_channel_circuit(channel)
    simulator = AerSimulator(method="matrix_product_state")
    counts = simulator.run(qc, shots=shots).result().get_counts()
    return reconstruct_channel_from_counts(counts, shots)


def recommended_shots(samples_per_pixel: int = 64) -> int:
    """FRQI needs address coverage; shots scale with number of pixels."""
    return FRQI_PIXELS * samples_per_pixel
