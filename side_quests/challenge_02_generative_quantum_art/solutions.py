"""
Challenge 2 - Generative Quantum Art (FRQI)
Solutions for patterns 1..7
"""
import math, os, sys
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

sys.path.insert(0, "./tmp/consignes/challenge_02_generative_quantum_art")
from utils_quantum import run_simulation

OUT = "./tmp/out"
os.makedirs(OUT, exist_ok=True)


# ---------- FRQI reconstruction ----------
def reconstruct_from_frqi(counts, n_pos, shape, grayscale=False):
    """Ratio method. Bitstring layout: 'C P_{k-1} ... P_0' (Qiskit little-endian on a single register)."""
    n_pixels = shape[0] * shape[1]
    total = np.zeros(n_pixels)
    ones = np.zeros(n_pixels)
    for outcome, c in counts.items():
        outcome = outcome.replace(" ", "")
        color = outcome[0]
        pos = outcome[1:]            # MSB..LSB string
        idx = int(pos, 2)
        if idx < n_pixels:
            total[idx] += c
            if color == '1':
                ones[idx] += c
    img = np.zeros(n_pixels)
    for i in range(n_pixels):
        if total[i] > 0:
            r = ones[i] / total[i]
            img[i] = r * 255.0 if grayscale else (255.0 if r > 0.5 else 0.0)
    return img.reshape(shape)


def show(images_titles, fname, suptitle=None, cmap='gray'):
    n = len(images_titles)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (im, t) in zip(axes, images_titles):
        ax.imshow(im, cmap=cmap, vmin=0, vmax=255, interpolation='nearest')
        ax.set_title(t); ax.axis('off')
    if suptitle:
        plt.suptitle(suptitle, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT, fname)
    plt.savefig(p, dpi=110, bbox_inches='tight')
    plt.close()
    return p


def n_pos_for(shape):
    n_pix = shape[0] * shape[1]
    return max(1, math.ceil(math.log2(n_pix)))


# ===================================================================
# Pattern 1 — Random noise (each pixel 50/50, independent across shots)
# ===================================================================
# Trick: encode color qubit in superposition independently of address.
# Simply put ALL qubits (positions + color) in |+>. Each shot picks a random
# basis state -> for each address the color is a fresh fair coin.
def pattern1_random(shape=(8, 8)):
    n_pos = n_pos_for(shape)
    qc = QuantumCircuit(n_pos + 1)
    qc.h(range(n_pos))          # address superposition
    qc.h(n_pos)                 # color: |+> -> P(1)=0.5 on every address
    qc.measure_all()
    counts = run_simulation(qc, shots=200 * 2**n_pos)
    # Show 4 *individual* outcomes (random snapshots), not the averaged image
    imgs = []
    for i, (bs, c) in enumerate(list(counts.items())[:4]):
        # build a single-shot image: take the 4 most-likely bitstrings and
        # render the address->color mapping they encode
        img = np.full(shape[0] * shape[1], 128, dtype=float)
        # collect one color per address from the very first shots
        addr_color = {}
        for bs2, _ in counts.items():
            color = int(bs2[0]); idx = int(bs2[1:], 2)
            if idx < img.size and idx not in addr_color:
                addr_color[idx] = color * 255
        for k, v in addr_color.items():
            img[k] = v
        imgs.append((img.reshape(shape), f"Sample"))
        if i >= 3: break
    # better visual: 4 independent runs
    samples = []
    for s in range(4):
        c = run_simulation(qc, shots=200 * 2**n_pos)
        samples.append((reconstruct_from_frqi(c, n_pos, shape), f"Run {s+1}"))
    return show(samples, "pattern1_random.png", "Pattern 1 — Random Noise")


# ===================================================================
# Pattern 2 — Checkerboard (deterministic)
# ===================================================================
# Position qubits encode pixel index i = r*C + c (row-major).
# Bit 0 of i = LSB(c) (since C is a power of 2).
# Bit log2(C) of i = LSB(r).
# Checkerboard color = LSB(r) XOR LSB(c) -> CNOT both qubits onto color qubit.
def pattern2_checkerboard(shape=(8, 8), negative=False):
    R, C = shape
    b = int(math.log2(C))
    a = int(math.log2(R))
    n_pos = a + b
    qc = QuantumCircuit(n_pos + 1)
    qc.h(range(n_pos))
    qc.cx(0, n_pos)        # LSB(c)
    qc.cx(b, n_pos)        # LSB(r)
    if negative:
        qc.x(n_pos)
    qc.measure_all()
    counts = run_simulation(qc, shots=100 * 2**n_pos)
    return reconstruct_from_frqi(counts, n_pos, shape)


# ===================================================================
# Pattern 3 — Horizontal lines (alternate per row)
# ===================================================================
def pattern3_horizontal(shape=(8, 8)):
    R, C = shape
    b = int(math.log2(C))
    a = int(math.log2(R))
    n_pos = a + b
    qc = QuantumCircuit(n_pos + 1)
    qc.h(range(n_pos))
    qc.cx(b, n_pos)        # depend only on LSB(r)
    qc.measure_all()
    counts = run_simulation(qc, shots=100 * 2**n_pos)
    return reconstruct_from_frqi(counts, n_pos, shape)


# ===================================================================
# Pattern 4 — Vertical lines (alternate per column)
# ===================================================================
def pattern4_vertical(shape=(8, 8)):
    R, C = shape
    b = int(math.log2(C))
    a = int(math.log2(R))
    n_pos = a + b
    qc = QuantumCircuit(n_pos + 1)
    qc.h(range(n_pos))
    qc.cx(0, n_pos)        # depend only on LSB(c)
    qc.measure_all()
    counts = run_simulation(qc, shots=100 * 2**n_pos)
    return reconstruct_from_frqi(counts, n_pos, shape)


# ===================================================================
# Pattern 5 — Nested (concentric) squares — per-pixel MCX
# ===================================================================
def _apply_x_mask(qc, bits_msb, pos_qubits):
    # bits_msb[0] is MSB; pos_qubits[0] is qubit holding bit 0 (LSB)
    k = len(pos_qubits)
    for j, bit in enumerate(bits_msb):
        if bit == '0':
            qc.x(pos_qubits[k - 1 - j])

def pattern5_nested(shape=(8, 8), bg_white=False):
    R, C = shape
    n_pos = n_pos_for(shape)
    qc = QuantumCircuit(n_pos + 1)
    qc.h(range(n_pos))
    pos = list(range(n_pos))
    color = n_pos
    for r in range(R):
        for c in range(C):
            L = min(r, c, R - 1 - r, C - 1 - c)
            # layer 0 black, 1 white, 2 black, ...  (white when L odd)
            white = (L % 2 == 1)
            if bg_white:
                white = not white
            if not white:
                continue
            i = r * C + c
            addr = format(i, f"0{n_pos}b")  # MSB..LSB
            _apply_x_mask(qc, addr, pos)
            qc.mcx(pos, color)
            _apply_x_mask(qc, addr, pos)
    qc.measure_all()
    counts = run_simulation(qc, shots=200 * 2**n_pos)
    return reconstruct_from_frqi(counts, n_pos, shape)


# ===================================================================
# Pattern 6 — Grayscale nested squares (one intensity per layer)
# ===================================================================
def pattern6_nested_gray(shape=(8, 8)):
    R, C = shape
    n_pos = n_pos_for(shape)
    qc = QuantumCircuit(n_pos + 1)
    qc.h(range(n_pos))
    pos = list(range(n_pos))
    color = n_pos
    max_layer = min(R, C) // 2 - 1  # deepest layer
    for r in range(R):
        for c in range(C):
            L = min(r, c, R - 1 - r, C - 1 - c)
            # intensity grows from outside (dark) to inside (bright)
            intensity = L / max(1, max_layer)        # 0..1
            theta = math.asin(math.sqrt(intensity))  # P(1)=intensity
            if theta == 0:
                continue
            i = r * C + c
            addr = format(i, f"0{n_pos}b")
            _apply_x_mask(qc, addr, pos)
            qc.mcry(2 * theta, pos, color)
            _apply_x_mask(qc, addr, pos)
    qc.measure_all()
    counts = run_simulation(qc, shots=400 * 2**n_pos)
    return reconstruct_from_frqi(counts, n_pos, shape, grayscale=True)


# ===================================================================
# Pattern 7 (Bonus) — Sierpinski triangle + inverse via selection qubit
# ===================================================================
def sierpinski_mask(size=8):
    m = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            if c <= r and (r & c) == c:
                m[r, c] = 1
    return m

def pattern7_fractal(size=8):
    """
    One circuit, one extra 'selection' qubit s.
    Group A (s=0): reconstruct the mask. Group B (s=1): reconstruct its inverse.
    For every pixel we apply an MCX controlled on (positions=address) AND on s,
    where s control state is 0 if mask[r,c]=1 (we want white when s=0),
    and 1 if mask[r,c]=0 (we want white when s=1 -> inverted picture).
    """
    shape = (size, size)
    n_pos = n_pos_for(shape)
    mask = sierpinski_mask(size)

    # layout: qubits 0..n_pos-1 = positions, n_pos = selection s, n_pos+1 = color
    qc = QuantumCircuit(n_pos + 2)
    pos = list(range(n_pos))
    s = n_pos
    color = n_pos + 1

    qc.h(pos)
    qc.h(s)

    for r in range(size):
        for c in range(size):
            i = r * size + c
            addr = format(i, f"0{n_pos}b")
            _apply_x_mask(qc, addr, pos)
            # For mask=1 -> want color=1 when s=0  => sandwich X on s
            if mask[r, c] == 1:
                qc.x(s)
            qc.mcx(pos + [s], color)
            if mask[r, c] == 1:
                qc.x(s)
            _apply_x_mask(qc, addr, pos)

    qc.measure_all()
    shots = 400 * (2 ** (n_pos + 1))
    counts = run_simulation(qc, shots=shots)

    # Split by selection bit s. Bitstring order: 'C S P...'
    # n_pos+2 total qubits, color is highest, s is next.
    n_pix = size * size
    grp = {0: {}, 1: {}}
    for bs, cnt in counts.items():
        bs = bs.replace(" ", "")
        color_b = bs[0]
        s_b = int(bs[1])
        pos_b = bs[2:]
        key = color_b + pos_b
        grp[s_b][key] = grp[s_b].get(key, 0) + cnt

    img_a = reconstruct_from_frqi(grp[0], n_pos, shape)
    img_b = reconstruct_from_frqi(grp[1], n_pos, shape)
    return img_a, img_b, mask * 255


# ===================================================================
# Run everything and save figures
# ===================================================================
if __name__ == "__main__":
    print("Pattern 1 — random noise")
    pattern1_random((8, 8))

    print("Pattern 2 — checkerboard + negative")
    a = pattern2_checkerboard((8, 8), negative=False)
    b = pattern2_checkerboard((8, 8), negative=True)
    show([(a, "Checkerboard"), (b, "Negative")], "pattern2_checkerboard.png",
         "Pattern 2 — Checkerboard")

    print("Pattern 3 — horizontal lines")
    show([(pattern3_horizontal((8, 8)), "Horizontal lines")],
         "pattern3_horizontal.png", "Pattern 3")

    print("Pattern 4 — vertical lines")
    show([(pattern4_vertical((8, 8)), "Vertical lines")],
         "pattern4_vertical.png", "Pattern 4")

    print("Pattern 5 — nested squares 8x8")
    show([(pattern5_nested((8, 8)), "Nested squares")],
         "pattern5_nested.png", "Pattern 5")

    print("Pattern 6 — grayscale nested squares 8x8")
    show([(pattern6_nested_gray((8, 8)), "Grayscale nested")],
         "pattern6_nested_gray.png", "Pattern 6")

    print("Pattern 7 — Sierpinski + inverse 8x8")
    a, b, mref = pattern7_fractal(8)
    show([(mref, "Classical mask"), (a, "s=0  (mask)"), (b, "s=1  (inverse)")],
         "pattern7_fractal.png", "Pattern 7 (bonus)")

    print("\nDone. Outputs in", OUT)
