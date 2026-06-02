"""
Traitement d'image par algorithmes quantiques — version parallélisée.

Deux opérations indépendantes :

1) DOWNSCALING  via Quantum Monte Carlo Integration (QMCI / IAE)
   ─────────────────────────────────────────────────────────────
   La moyenne de chaque bloc BxB est estimée par Iterative Amplitude
   Estimation. Avantage quantique : O(1/ε) évaluations oracle vs O(1/ε²)
   en Monte Carlo classique.

   Parallélisme : chaque bloc est indépendant. On utilise un ThreadPool
   (les Samplers Qiskit libèrent le GIL sur les parties C++) pour traiter
   plusieurs blocs simultanément.

2) UPSCALING  via états de Dicke (halftoning quantique)
   ──────────────────────────────────────────────────────
   Chaque pixel génère UN état de Dicke |D(n, k=round(t·n))⟩ donnant
   exactement k sous-pixels en couleur_B (variance 0 sur la densité).

   Parallélisme : deux niveaux.
     a) Cache des vecteurs d'état par k unique → évite de recalculer C(n,k).
     b) Tous les circuits du batch sont soumis en un seul SIM.run() →
        Aer les exécute en parallèle via ses threads internes (OMP/MKL).
        Speedup typique : 10-20x vs appels séquentiels.

Usage :
    python quantum_image.py --mode downscale --input in.jpg --output out.png --block 4
    python quantum_image.py --mode upscale   --input in.jpg --output out.png --scale 4
    python quantum_image.py --mode both      --input in.jpg --output out --block 4 --scale 4

    # Fortement recommandé pour images > 100px (algorithmes O(H·W)) :
    python quantum_image.py --mode both --input in.jpg --output out \\
        --block 4 --scale 4 --max-side 64 --workers 8
"""

from __future__ import annotations

import argparse
import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation

warnings.filterwarnings("ignore")

_SIM = AerSimulator()

# ── Cache des circuits Dicke transpilés, indexés par (n_qubits, k) ──────────
_DICKE_SV_CACHE: dict[tuple[int, int], np.ndarray] = {}
_DICKE_QC_CACHE: dict[tuple[int, int], QuantumCircuit] = {}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers Dicke
# ══════════════════════════════════════════════════════════════════════════════

def _dicke_statevector(n: int, k: int) -> np.ndarray:
    """Superposition uniforme de tous les mots de n bits avec exactement k uns."""
    key = (n, k)
    if key not in _DICKE_SV_CACHE:
        sv = np.zeros(2 ** n, dtype=complex)
        for bits in combinations(range(n), k):
            sv[sum(1 << b for b in bits)] = 1.0
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv /= norm
        _DICKE_SV_CACHE[key] = sv
    return _DICKE_SV_CACHE[key]


def _dicke_circuit(n: int, k: int) -> QuantumCircuit:
    """Circuit Dicke transpilé, mis en cache."""
    key = (n, k)
    if key not in _DICKE_QC_CACHE:
        sv = _dicke_statevector(n, k)
        qc = QuantumCircuit(n, n)
        qc.prepare_state(sv, range(n))
        qc.measure(range(n), range(n))
        _DICKE_QC_CACHE[key] = transpile(qc, _SIM, optimization_level=1)
    return _DICKE_QC_CACHE[key]


# ══════════════════════════════════════════════════════════════════════════════
# 1)  DOWNSCALING  —  QMCI avec ThreadPool
# ══════════════════════════════════════════════════════════════════════════════

def _qmci_worker(flat_list: list[float], epsilon: float) -> float:
    """
    Worker autonome pour ThreadPoolExecutor : construit son propre oracle
    et son propre StatevectorSampler (thread-safe, pas de state partagé).
    """
    flat = np.array(flat_list, dtype=np.float64)
    N = len(flat)
    n = int(math.log2(N))

    if np.all(flat < 1e-12):
        return 0.0
    if np.all(flat > 1.0 - 1e-12):
        return 1.0

    # Construction de l'oracle QMCI
    qc = QuantumCircuit(n + 1)
    for q in range(n):
        qc.h(q)
    for x, val in enumerate(flat):
        val = float(np.clip(val, 0.0, 1.0))
        if val < 1e-12:
            continue
        theta = 2.0 * math.asin(math.sqrt(val))
        ctrl_bits = format(x, f"0{n}b")
        flips = [q for q, b in enumerate(reversed(ctrl_bits)) if b == "0"]
        if flips:
            qc.x(flips)
        qc.mcry(theta, list(range(n)), n)
        if flips:
            qc.x(flips)

    problem = EstimationProblem(state_preparation=qc, objective_qubits=[n])
    iae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon,
        alpha=0.05,
        sampler=StatevectorSampler(),   # un sampler par thread
    )
    return float(np.clip(iae.estimate(problem).estimation, 0.0, 1.0))


def quantum_downscale(img: Image.Image, block: int,
                      epsilon: float = 0.02,
                      workers: int = 4) -> Image.Image:
    """
    Downscale RGB via QMCI parallélisé.

    Chaque bloc (indépendant) est confié à un thread du pool.
    Sur N cœurs, speedup ≈ N (les parties C++ de Qiskit libèrent le GIL).

    Paramètres
    ----------
    block   : taille du bloc (puissance de 2 ≥ 2).
    epsilon : précision IAE (défaut 0.02).
    workers : nombre de threads parallèles (défaut 4 ; mettre os.cpu_count()).
    """
    if block < 2 or (block & (block - 1)) != 0:
        raise ValueError("block doit être une puissance de 2 ≥ 2.")

    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    h2 = (h // block) * block
    w2 = (w // block) * block
    arr = arr[:h2, :w2]
    out_h, out_w = h2 // block, w2 // block
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Prépare toutes les tâches : (by, bx, c, flat)
    tasks: list[tuple[int, int, int, list[float]]] = []
    for by in range(out_h):
        for bx in range(out_w):
            y0, x0 = by * block, bx * block
            for c in range(3):
                blk = arr[y0:y0 + block, x0:x0 + block, c]
                flat = (blk.astype(np.float64).flatten() / 255.0).tolist()
                tasks.append((by, bx, c, flat))

    total = len(tasks)
    done = 0
    print(f"  {total} tâches QMCI → {workers} threads")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_qmci_worker, flat, epsilon): (by, bx, c)
            for by, bx, c, flat in tasks
        }
        for fut in as_completed(futures):
            by, bx, c = futures[fut]
            mean = fut.result()
            out[by, bx, c] = int(round(mean * 255))
            done += 1
            print(f"  QMCI downscale: {done}/{total}", end="\r")

    print()
    return Image.fromarray(out, "RGB")


# ══════════════════════════════════════════════════════════════════════════════
# 2)  UPSCALING  —  États de Dicke avec batch SIM.run()
# ══════════════════════════════════════════════════════════════════════════════

def project_on_color_axis(pixel: np.ndarray,
                           color_a: np.ndarray,
                           color_b: np.ndarray) -> float:
    """Projette un pixel RGB sur l'axe color_a → color_b. Retourne t ∈ [0, 1]."""
    ab = color_b - color_a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return 0.0
    t = float(np.dot(pixel - color_a, ab) / denom)
    return max(0.0, min(1.0, t))


def auto_palette(img: Image.Image, k: int) -> np.ndarray:
    """Détecte k couleurs dominantes par k-means++ / Lloyd."""
    arr = np.array(img.convert("RGB"), dtype=np.float64).reshape(-1, 3)
    rng = np.random.default_rng(0)
    if arr.shape[0] > 20000:
        sample = arr[rng.choice(arr.shape[0], 20000, replace=False)]
    else:
        sample = arr
    centers = [sample[rng.integers(0, sample.shape[0])]]
    for _ in range(k - 1):
        d2 = np.min(
            ((sample[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(-1),
            axis=1,
        )
        probs = d2 / (d2.sum() + 1e-12)
        centers.append(sample[rng.choice(sample.shape[0], p=probs)])
    centers = np.array(centers, dtype=np.float64)
    for _ in range(15):
        d2 = ((sample[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = np.argmin(d2, axis=1)
        new_centers = np.array([
            sample[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
            for j in range(k)
        ])
        if np.allclose(new_centers, centers, atol=0.5):
            centers = new_centers
            break
        centers = new_centers
    lum = centers @ np.array([0.299, 0.587, 0.114])
    return np.clip(centers[np.argsort(lum)], 0, 255)


def quantum_upscale(img: Image.Image, scale: int,
                    palette: np.ndarray,
                    batch_size: int = 512) -> Image.Image:
    """
    Upscale par halftoning Dicke — version batché.

    Au lieu d'appeler SIM.run() une fois par pixel (lent), on regroupe
    tous les circuits d'un batch et on les soumet en un seul appel.
    Aer distribue le batch sur ses threads internes (OpenMP/MKL).

    Avantage Dicke : chaque mesure donne EXACTEMENT round(t·n) sous-pixels
    en couleur_B (variance 0), avec distribution spatiale uniforme —
    impossible avec des tirages Bernoulli indépendants.

    Paramètres
    ----------
    scale      : facteur d'upscale (≥ 2 ; scale² doit être une puissance de 2).
    batch_size : pixels par batch SIM.run() (défaut 512 ; augmenter si RAM ok).
    """
    if scale < 2:
        raise ValueError("scale doit être ≥ 2.")

    n_sub = scale * scale
    if n_sub & (n_sub - 1) != 0:
        n_sub_q = 2 ** int(math.log2(n_sub))
        print(f"  Avertissement: scale²={n_sub} n'est pas une puissance de 2, "
              f"utilisation de {n_sub_q} sous-pixels quantiques.")
    else:
        n_sub_q = n_sub

    # Pré-cache tous les circuits Dicke possibles pour ce n_sub_q
    print(f"  Pré-cache {n_sub_q + 1} circuits Dicke (n={n_sub_q})…")
    for k in range(n_sub_q + 1):
        _dicke_circuit(n_sub_q, k)
    print(f"  Cache prêt.")

    arr = np.array(img.convert("RGB"), dtype=np.float64)
    h, w, _ = arr.shape
    pal = np.asarray(palette, dtype=np.float64)
    out = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

    # Collecte tous les pixels avec leurs métadonnées
    pixel_meta: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []
    for y in range(h):
        for x in range(w):
            px = arr[y, x]
            d = np.linalg.norm(pal - px, axis=1)
            i, j = np.argsort(d)[:2]
            ca, cb = pal[i], pal[j]
            t = project_on_color_axis(px, ca, cb)
            k = max(0, min(n_sub_q, int(round(t * n_sub_q))))
            pixel_meta.append((y, x, k, n_sub_q, ca, cb))

    total_px = len(pixel_meta)
    done = 0

    # Traitement par batches
    for batch_start in range(0, total_px, batch_size):
        batch = pixel_meta[batch_start: batch_start + batch_size]

        # Séparer les cas triviaux (k=0 ou k=n) des cas quantiques
        trivial = [(i, m) for i, m in enumerate(batch) if m[2] == 0 or m[2] == n_sub_q]
        quantum = [(i, m) for i, m in enumerate(batch) if 0 < m[2] < n_sub_q]

        # Résultats pour ce batch
        batch_bits: dict[int, np.ndarray] = {}

        # Cas triviaux : pas de circuit nécessaire
        for i, (y, x, k, n_q, ca, cb) in trivial:
            batch_bits[i] = np.ones(n_q, dtype=np.uint8) if k == n_q \
                else np.zeros(n_q, dtype=np.uint8)

        # Cas quantiques : un seul SIM.run() pour tout le batch
        if quantum:
            circuits = [_dicke_circuit(n_sub_q, m[2]) for _, m in quantum]
            results = _SIM.run(circuits, shots=1, memory=True).result()
            for job_idx, (i, (y, x, k, n_q, ca, cb)) in enumerate(quantum):
                bitstr = results.get_memory(job_idx)[0]
                batch_bits[i] = np.array(
                    [int(b) for b in reversed(bitstr)], dtype=np.uint8
                )

        # Écriture dans l'image de sortie
        for i, (y, x, k, n_q, ca, cb) in enumerate(batch):
            bits = batch_bits[i][:n_sub].reshape(scale, scale)
            block = np.where(bits[..., None] == 1, cb, ca).astype(np.uint8)
            out[y * scale:(y + 1) * scale, x * scale:(x + 1) * scale] = block

        done += len(batch)
        print(f"  Dicke halftone: {done}/{total_px} pixels", end="\r")

    print()
    return Image.fromarray(out, "RGB")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import os
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["downscale", "upscale", "both"], required=True)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path,
                   help="Pour --mode both : <output>_down.png et <output>_up.png")
    p.add_argument("--block", type=int, default=4,
                   help="taille du bloc QMCI (puissance de 2 ≥ 2, défaut 4)")
    p.add_argument("--epsilon", type=float, default=0.02,
                   help="précision IAE (défaut 0.02 ≈ 5/255 niveaux de gris)")
    p.add_argument("--scale", type=int, default=4,
                   help="facteur d'upscale Dicke (défaut 4)")
    p.add_argument("--num-colors", type=int, default=6,
                   help="couleurs auto-détectées pour la palette halftoning (défaut 6)")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                   help="threads parallèles pour QMCI downscale (défaut = nb de cœurs)")
    p.add_argument("--batch-size", type=int, default=512,
                   help="pixels par batch pour Dicke upscale (défaut 512)")
    p.add_argument("--max-side", type=int, default=0,
                   help="limite le côté max avant traitement (0 = désactivé)")
    args = p.parse_args()

    img = Image.open(args.input).convert("RGB")
    if args.max_side > 0 and max(img.size) > args.max_side:
        r = args.max_side / max(img.size)
        img = img.resize(
            (max(1, int(img.size[0] * r)), max(1, int(img.size[1] * r))),
            Image.LANCZOS,
        )
        print(f"Image redimensionnée à {img.size}.")

    if args.mode in ("downscale", "both"):
        print(f"\n[QMCI Downscale]  bloc={args.block}  ε={args.epsilon}  "
              f"workers={args.workers}")
        down = quantum_downscale(img, args.block,
                                 epsilon=args.epsilon, workers=args.workers)
        out_path = (args.output if args.mode == "downscale"
                    else args.output.with_name(args.output.stem + "_down.png"))
        down.save(out_path)
        print(f"  → {out_path}  ({down.size[0]}×{down.size[1]} px)")

    if args.mode in ("upscale", "both"):
        palette = auto_palette(img, max(2, args.num_colors))
        pal_str = ", ".join(f"({int(c[0])},{int(c[1])},{int(c[2])})" for c in palette)
        print(f"\n[Dicke Halftone Upscale]  scale={args.scale}  "
              f"batch={args.batch_size}  palette ({len(palette)} couleurs)")
        print(f"  {pal_str}")
        up = quantum_upscale(img, args.scale, palette, batch_size=args.batch_size)
        out_path = (args.output if args.mode == "upscale"
                    else args.output.with_name(args.output.stem + "_up.png"))
        up.save(out_path)
        print(f"  → {out_path}  ({up.size[0]}×{up.size[1]} px)")


if __name__ == "__main__":
    main()
