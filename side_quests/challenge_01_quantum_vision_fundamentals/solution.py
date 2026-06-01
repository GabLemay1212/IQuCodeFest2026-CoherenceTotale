"""
Challenge 1 — Quantum Vision Fundamentals (FRQI)
Solutions to Problems 1, 2 and 3.

Run from the challenge folder so `utils_images` and `utils_quantum` resolve:
    cd challenge_01_quantum_vision_fundamentals
    python challenge_01_frqi.py
"""

import math
import numpy as np
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

import utils_images as img_utils
import utils_quantum as q_utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _apply_x_mask(qc: QuantumCircuit, address_bitstring: str, position_qubits: list[int]) -> None:
    """X-mask so that an MCX/MCRY triggers on an address with 0-bits."""
    for bit_idx, bit in enumerate(reversed(address_bitstring)):
        if bit == "0":
            qc.x(position_qubits[bit_idx])


# ---------------------------------------------------------------------------
# Problem 1 — FRQI encoding/decoding for a grayscale image
# ---------------------------------------------------------------------------
def encode_frqi_grayscale(grayscale_image: np.ndarray) -> tuple[QuantumCircuit, int]:
    """
    For each pixel of intensity I in [0, 1] we want P(|1>) = I on the color qubit.
        sin^2(theta) = I  =>  theta = arcsin(sqrt(I))
    Qiskit's RY uses the half-angle convention, so we pass phi = 2*theta.
    """
    flat = np.asarray(grayscale_image, dtype=float).flatten() / 255.0
    n_pixels = flat.size
    if n_pixels == 0:
        raise ValueError("Empty image.")

    n_pos = math.ceil(math.log2(n_pixels))
    pos_qubits = list(range(n_pos))
    color_qubit = n_pos

    qc = QuantumCircuit(n_pos + 1)
    qc.h(pos_qubits)  # uniform superposition over addresses

    for i, intensity in enumerate(flat):
        if intensity <= 0.0:
            continue  # theta = 0 -> identity
        theta = math.asin(math.sqrt(min(max(intensity, 0.0), 1.0)))
        phi = 2.0 * theta

        addr = format(i, f"0{n_pos}b")  # MSB..LSB
        _apply_x_mask(qc, addr, pos_qubits)
        qc.mcry(phi, pos_qubits, color_qubit)
        _apply_x_mask(qc, addr, pos_qubits)
        qc.barrier()

    print(f"FRQI grayscale circuit — depth {qc.depth()}, size {qc.size()}")
    return qc, n_pos


def reconstruct_grayscale_from_frqi(
    counts: dict, n_position_qubits: int, image_shape: tuple[int, int]
) -> np.ndarray:
    """Ratio method: per-address P(|1>) ~= intensity, rescaled to 0..255."""
    n_pixels = image_shape[0] * image_shape[1]
    total = np.zeros(n_pixels, dtype=int)
    ones = np.zeros(n_pixels, dtype=int)

    for outcome, count in counts.items():
        outcome = outcome.replace(" ", "")
        color_char = outcome[0]           # leftmost = color qubit
        position_string = outcome[1:]     # remaining = MSB..LSB of position
        idx = int(position_string, 2)
        if idx < n_pixels:
            total[idx] += count
            if color_char == "1":
                ones[idx] += count

    out = np.zeros(n_pixels, dtype=np.uint8)
    nz = total > 0
    out[nz] = np.rint(ones[nz] / total[nz] * 255).astype(np.uint8)
    return out.reshape(image_shape)


# ---------------------------------------------------------------------------
# Problem 2 — Negative shot (single X on the color qubit)
# ---------------------------------------------------------------------------
def negative_shot(image: np.ndarray) -> np.ndarray:
    qc, n_pos = encode_frqi_grayscale(image)
    qc.x(n_pos)  # P(|1>) = sin^2 -> cos^2 = 1 - I
    qc.measure_all()
    counts = q_utils.run_simulation(qc, shots=200 * (2 ** n_pos))
    return reconstruct_grayscale_from_frqi(counts, n_pos, image.shape)


# ---------------------------------------------------------------------------
# Problem 3 — Block swap (left<->right) and full horizontal flip
# ---------------------------------------------------------------------------
def block_swap_lr(image: np.ndarray) -> np.ndarray:
    """Swap left & right halves: flip the MSB of the x-register."""
    qc, n_pos = encode_frqi_grayscale(image)
    n_per_axis = n_pos // 2
    x_msb = n_per_axis - 1
    qc.x(x_msb)
    qc.measure_all()
    counts = q_utils.run_simulation(qc, shots=200 * (2 ** n_pos))
    return reconstruct_grayscale_from_frqi(counts, n_pos, image.shape)


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """Mirror x -> (W-1)-x: flip EVERY x qubit."""
    qc, n_pos = encode_frqi_grayscale(image)
    n_per_axis = n_pos // 2
    for q in range(n_per_axis):  # q0..q(n_per_axis-1) encode x
        qc.x(q)
    qc.measure_all()
    counts = q_utils.run_simulation(qc, shots=200 * (2 ** n_pos))
    return reconstruct_grayscale_from_frqi(counts, n_pos, image.shape)


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------
def show_pair(original: np.ndarray, processed: np.ndarray, title1: str, title2: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(title1)
    axes[0].axis("off")
    axes[1].imshow(processed, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(title2)
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- Problem 1: 4x4 grayscale ----
    grayscale_4x4 = np.array(
        [[0, 64, 128, 255],
         [64, 128, 255, 128],
         [128, 255, 128, 64],
         [255, 128, 64, 0]],
        dtype=np.uint8,
    )

    qc, n_pos = encode_frqi_grayscale(grayscale_4x4)
    qc.measure_all()
    counts = q_utils.run_simulation(qc, shots=200 * (2 ** n_pos))
    recon = reconstruct_grayscale_from_frqi(counts, n_pos, grayscale_4x4.shape)
    print("\nProblem 1 — reconstructed:\n", recon)
    print("MAE:", np.mean(np.abs(grayscale_4x4.astype(int) - recon.astype(int))))
    show_pair(grayscale_4x4, recon, "Original 4×4", "Reconstructed (FRQI)")

    # ---- Problem 2: negative ----
    neg = negative_shot(grayscale_4x4)
    print("\nProblem 2 — negative:\n", neg)
    show_pair(grayscale_4x4, neg, "Original 4×4", "Negative (X on color qubit)")

    # ---- Problem 3: 8x8 image with bar + diagonal + gradient ----
    image_8x8 = np.zeros((8, 8), dtype=np.uint8)
    image_8x8[1:7, 1] = 200
    for i in range(3):
        image_8x8[2 + i, 4 + i] = 255
    for y in range(8):
        for x in range(8):
            image_8x8[y, x] = max(image_8x8[y, x], x * 20)

    swapped = block_swap_lr(image_8x8)
    flipped = horizontal_flip(image_8x8)
    print("\nProblem 3a — block-swapped:\n", swapped)
    print("\nProblem 3b — horizontally flipped:\n", flipped)
    print("flip MAE vs np.fliplr:",
          np.mean(np.abs(flipped.astype(int) - np.fliplr(image_8x8).astype(int))))
    show_pair(image_8x8, swapped, "Original 8×8", "Block swap L↔R")
    show_pair(image_8x8, flipped, "Original 8×8", "Horizontal flip")
