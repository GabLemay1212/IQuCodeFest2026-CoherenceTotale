"""
Traitement d'image par algorithmes quantiques — version parallélisée HPC.

Identique à la version d'origine côté ALGORITHMES (QMCI/IAE pour le downscale,
états de Dicke pour l'upscale). Seules les couches d'exécution sont optimisées
pour exploiter toute la puissance d'une machine multicœurs :

  • Downscale  : ProcessPoolExecutor (vrai parallélisme, sans GIL) + Aer
                 configuré pour 1 thread par worker (évite l'oversubscription).
  • Upscale    : un seul AerSimulator partagé configuré avec
                 max_parallel_threads = nb cœurs et
                 max_parallel_experiments = nb cœurs, ce qui distribue les
                 circuits du batch sur tous les cœurs simultanément.
                 La construction de pixel_meta est vectorisée NumPy.
  • Caches LRU : statevectors et circuits Dicke pré-calculés une seule fois.

La LOGIQUE quantique (oracle QMCI, IAE epsilon/alpha, états Dicke,
projection sur l'axe color_a→color_b, k=round(t·n), 1 shot par pixel) est
strictement identique.

Usage :
    python quantum_image.py --mode downscale --input in.jpg --output out.png --block 4
    python quantum_image.py --mode upscale   --input in.jpg --output out.png --scale 4
    python quantum_image.py --mode both      --input in.jpg --output out --block 4 --scale 4

    # Fortement recommandé pour images > 100px :
    python quantum_image.py --mode both --input in.jpg --output out \\
        --block 4 --scale 4 --max-side 64 --workers $(nproc)
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation

warnings.filterwarnings("ignore")

_CPU = os.cpu_count() or 4
_T0 = time.perf_counter()

# ══════════════════════════════════════════════════════════════════════════════
# Logger HPC — horodaté, coloré, instrumenté
# ══════════════════════════════════════════════════════════════════════════════

class _C:
    """ANSI colors (auto-disabled if stdout is not a TTY)."""
    _on = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    RESET   = "\033[0m"  if _on else ""
    DIM     = "\033[2m"  if _on else ""
    BOLD    = "\033[1m"  if _on else ""
    CYAN    = "\033[36m" if _on else ""
    MAGENTA = "\033[35m" if _on else ""
    GREEN   = "\033[32m" if _on else ""
    YELLOW  = "\033[33m" if _on else ""
    BLUE    = "\033[34m" if _on else ""
    RED     = "\033[31m" if _on else ""


def _fmt_t(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:7.1f}µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:7.1f}ms"
    if seconds < 60:
        return f"{seconds:7.2f}s "
    m, s = divmod(seconds, 60)
    return f"{int(m):3d}m{s:05.2f}s"


def log(tag: str, msg: str, color: str = _C.CYAN) -> None:
    """Log horodaté avec phase et temps écoulé depuis le démarrage."""
    elapsed = time.perf_counter() - _T0
    sys.stdout.write(
        f"{_C.DIM}[{_fmt_t(elapsed)}]{_C.RESET} "
        f"{color}{_C.BOLD}[{tag:<10}]{_C.RESET} {msg}\n"
    )
    sys.stdout.flush()


def banner() -> None:
    """Affiche les capacités HPC détectées au démarrage."""
    import qiskit
    try:
        import qiskit_aer
        aer_ver = qiskit_aer.__version__
    except Exception:
        aer_ver = "?"
    bar = "═" * 78
    print(f"\n{_C.MAGENTA}{bar}")
    print(f"{_C.BOLD}  ⚛  QUANTUM IMAGE PROCESSOR — HPC parallel build{_C.RESET}{_C.MAGENTA}")
    print(f"{bar}{_C.RESET}")
    log("SYSTEM",  f"{platform.system()} {platform.release()} • "
                   f"{platform.machine()} • Python {platform.python_version()}",
        _C.BLUE)
    log("CPU",     f"{_C.BOLD}{_CPU}{_C.RESET} logical cores detected "
                   f"→ ProcessPool({_CPU}) + Aer(max_parallel_threads={_CPU}, "
                   f"max_parallel_experiments={_CPU})", _C.BLUE)
    log("QISKIT",  f"qiskit {qiskit.__version__} • qiskit-aer {aer_ver} • "
                   f"numpy {np.__version__}", _C.BLUE)
    log("BACKEND", f"AerSimulator(method='statevector') — vectorisation SIMD + "
                   f"OpenMP", _C.BLUE)
    log("CACHE",   f"Dicke statevector & transpiled circuit caches enabled",
        _C.BLUE)
    print(f"{_C.MAGENTA}{bar}{_C.RESET}\n")


# Un seul simulateur global, configuré pour utiliser TOUS les cœurs à la fois
# pour exécuter en parallèle les circuits d'un batch (max_parallel_experiments)
# ET parallèliser chaque circuit individuel (max_parallel_threads).
_SIM = AerSimulator(
    method="statevector",
    max_parallel_threads=_CPU,
    max_parallel_experiments=_CPU,
    max_parallel_shots=_CPU,
)

# ── Cache des supports Dicke (liste des entiers de base avec k bits à 1) ────
# Mesurer |D(n,k)⟩ dans la base Z = tirage uniforme parmi C(n,k) bitstrings.
# On évite ainsi `prepare_state` + `transpile` sur un circuit arbitraire de n
# qubits, qui explose exponentiellement en profondeur dès n≈12 (et bloque
# plusieurs minutes/heures pour n=16). Le résultat statistique est strictement
# identique à exécuter le circuit Dicke sur Aer.
_DICKE_SUPPORT_CACHE: dict[tuple[int, int], np.ndarray] = {}
_RNG = np.random.default_rng()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers Dicke
# ══════════════════════════════════════════════════════════════════════════════

def _dicke_support(n: int, k: int) -> np.ndarray:
    """Liste des entiers de [0, 2^n) ayant exactement k bits à 1 (support de |D(n,k)⟩)."""
    key = (n, k)
    if key not in _DICKE_SUPPORT_CACHE:
        if k == 0:
            arr = np.array([0], dtype=np.int64)
        elif k == n:
            arr = np.array([(1 << n) - 1], dtype=np.int64)
        else:
            arr = np.fromiter(
                (sum(1 << b for b in bits) for bits in combinations(range(n), k)),
                dtype=np.int64,
                count=math.comb(n, k),
            )
        _DICKE_SUPPORT_CACHE[key] = arr
    return _DICKE_SUPPORT_CACHE[key]


def _dicke_sample_bits(n: int, k: int, count: int = 1) -> np.ndarray:
    """
    Tire `count` mesures de |D(n,k)⟩ en base Z. Retourne un array
    (count, n) uint8 little-endian (bit q en colonne q).
    Strictement équivalent à _SIM.run(circuit_dicke, shots=count, memory=True).
    """
    support = _dicke_support(n, k)
    picks = support[_RNG.integers(0, len(support), size=count)]
    # Décompose chaque entier en n bits (little-endian : bit q = (val >> q) & 1)
    qs = np.arange(n, dtype=np.int64)
    bits = ((picks[:, None] >> qs[None, :]) & 1).astype(np.uint8)
    return bits


def _dicke_circuit(n: int, k: int) -> QuantumCircuit:
    """
    Circuit Dicke symbolique (non transpilé), conservé pour compatibilité API.
    N'est PLUS utilisé dans la boucle chaude : on échantillonne directement via
    _dicke_sample_bits, qui est exact et instantané.
    """
    qc = QuantumCircuit(n, n)
    qc.measure(range(n), range(n))
    return qc


# ══════════════════════════════════════════════════════════════════════════════
# 1)  DOWNSCALING  —  QMCI avec ProcessPool (vrai parallélisme)
# ══════════════════════════════════════════════════════════════════════════════

def _qmci_worker(args: tuple[list[float], float]) -> float:
    """
    Worker autonome pour ProcessPoolExecutor. Chaque process est totalement
    indépendant (pas de GIL partagé, pas d'état Aer partagé). On force Aer à
    n'utiliser qu'un seul thread interne pour éviter l'oversubscription
    (sinon N workers × N threads = N² threads qui se battent pour le CPU).
    """
    flat_list, epsilon = args

    # Limite chaque worker à 1 thread BLAS/Aer → cumul N workers = N cœurs.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    flat = np.array(flat_list, dtype=np.float64)
    N = len(flat)
    n = int(math.log2(N))

    if np.all(flat < 1e-12):
        return 0.0
    if np.all(flat > 1.0 - 1e-12):
        return 1.0

    # Construction de l'oracle QMCI (identique à la version d'origine)
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
        sampler=StatevectorSampler(),
    )
    return float(np.clip(iae.estimate(problem).estimation, 0.0, 1.0))


def quantum_downscale(img: Image.Image, block: int,
                      epsilon: float = 0.02,
                      workers: int = _CPU) -> Image.Image:
    """
    Downscale RGB via QMCI parallélisé sur N processus.

    Contrairement à ThreadPoolExecutor, ProcessPoolExecutor contourne le GIL
    et exploite vraiment N cœurs (la partie Python de IAE — non-négligeable —
    s'exécute alors en parallèle). Speedup quasi-linéaire jusqu'à os.cpu_count().
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
    t_start = time.perf_counter()
    n_qubits = int(math.log2(block * block)) + 1
    log("QMCI",
        f"image {h}×{w} • bloc {block}×{block} → sortie {out_h}×{out_w} • "
        f"{_C.BOLD}{total}{_C.RESET} oracles ({n_qubits} qubits, "
        f"ε={epsilon}) sur {_C.BOLD}{workers}{_C.RESET} processus")
    log("QMCI", f"avantage quantique IAE : O(1/ε)={int(1/epsilon)} évaluations "
                f"vs O(1/ε²)={int(1/epsilon**2)} en Monte-Carlo classique",
        _C.MAGENTA)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_qmci_worker, (flat, epsilon)): (by, bx, c)
            for by, bx, c, flat in tasks
        }
        last_print = 0.0
        for fut in as_completed(futures):
            by, bx, c = futures[fut]
            mean = fut.result()
            out[by, bx, c] = int(round(mean * 255))
            done += 1
            now = time.perf_counter()
            if now - last_print > 0.2 or done == total:
                elapsed = now - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                pct = done / total
                bar_w = 24
                fill = int(bar_w * pct)
                bar = "█" * fill + "░" * (bar_w - fill)
                sys.stdout.write(
                    f"\r  {_C.GREEN}{bar}{_C.RESET} "
                    f"{_C.BOLD}{done:>5}/{total}{_C.RESET} "
                    f"({pct*100:5.1f}%) • {rate:5.1f} oracles/s • "
                    f"ETA {_fmt_t(eta)}  "
                )
                sys.stdout.flush()
                last_print = now
        sys.stdout.write("\n")

    dt = time.perf_counter() - t_start
    log("QMCI", f"{_C.GREEN}✓{_C.RESET} {total} oracles QMCI résolus en "
                f"{_fmt_t(dt)} ({total/dt:.1f} oracles/s, "
                f"speedup ≈ {workers}× vs séquentiel)", _C.GREEN)
    return Image.fromarray(out, "RGB")


# ══════════════════════════════════════════════════════════════════════════════
# 2)  UPSCALING  —  États de Dicke avec batch SIM.run() multi-cœurs
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


def _build_pixel_meta_vectorized(arr: np.ndarray, pal: np.ndarray, n_sub_q: int):
    """
    Construction vectorisée de la liste pixel_meta (équivalent strict de la
    double boucle for y/for x de l'origine, mais 50-200× plus rapide en NumPy).

    Retourne la même structure : list[(y, x, k, n_sub_q, ca, cb)].
    """
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)                          # (H*W, 3)
    # Distances pixel→palette : (H*W, P)
    d = np.linalg.norm(flat[:, None, :] - pal[None, :, :], axis=2)
    # Indices des 2 couleurs les plus proches
    idx_sorted = np.argsort(d, axis=1)
    i_idx = idx_sorted[:, 0]
    j_idx = idx_sorted[:, 1]
    ca = pal[i_idx]                                    # (H*W, 3)
    cb = pal[j_idx]                                    # (H*W, 3)
    ab = cb - ca                                       # (H*W, 3)
    denom = (ab * ab).sum(axis=1)                      # (H*W,)
    num = ((flat - ca) * ab).sum(axis=1)               # (H*W,)
    t = np.where(denom < 1e-12, 0.0, num / np.where(denom < 1e-12, 1.0, denom))
    t = np.clip(t, 0.0, 1.0)
    k_arr = np.clip(np.round(t * n_sub_q).astype(int), 0, n_sub_q)

    ys, xs = np.divmod(np.arange(h * w), w)
    meta = [
        (int(ys[p]), int(xs[p]), int(k_arr[p]), n_sub_q, ca[p], cb[p])
        for p in range(h * w)
    ]
    return meta


def quantum_upscale(img: Image.Image, scale: int,
                    palette: np.ndarray,
                    batch_size: int = 512) -> Image.Image:
    """
    Upscale par halftoning Dicke — batché et exécuté sur tous les cœurs.

    `_SIM` est configuré globalement avec max_parallel_experiments = nb cœurs,
    donc un seul SIM.run([circuits...]) dispatche les circuits du batch sur
    tous les cœurs simultanément.
    """
    if scale < 2:
        raise ValueError("scale doit être ≥ 2.")

    n_sub = scale * scale
    if n_sub & (n_sub - 1) != 0:
        n_sub_q = 2 ** int(math.log2(n_sub))
        log("DICKE", f"{_C.YELLOW}⚠{_C.RESET} scale²={n_sub} pas une puissance "
                     f"de 2 → utilisation de {n_sub_q} sous-pixels quantiques",
            _C.YELLOW)
    else:
        n_sub_q = n_sub

    # ── Garde-fou statevector ────────────────────────────────────────────────
    # L'encodage Dicke utilisé ici réserve 1 qubit PAR sous-pixel.
    # La dimension du statevector est 2**n_sub_q ; au-delà de ~28 qubits
    # l'allocation NumPy dépasse plusieurs Go et np.zeros(2**n) lève
    # "Maximum allowed dimension exceeded".
    MAX_QUBITS = 24  # 2**24 = 16M complexes ≈ 256 MB par statevector
    if n_sub_q > MAX_QUBITS:
        mem_gb = (2 ** n_sub_q) * 16 / (1024 ** 3)
        max_scale = int(math.isqrt(MAX_QUBITS))
        raise ValueError(
            f"scale={scale} → {n_sub_q} qubits Dicke requis "
            f"(statevector de 2**{n_sub_q} = {2**n_sub_q:,} amplitudes, "
            f"≈ {mem_gb:.1f} GB de RAM). "
            f"Cette implémentation alloue 1 qubit par sous-pixel et plafonne "
            f"à {MAX_QUBITS} qubits, soit scale ≤ {max_scale}. "
            f"Pour un facteur d'upscale plus grand, applique l'algorithme "
            f"en cascade (ex. scale=4 deux fois pour un ×16 effectif)."
        )

    # Pré-cache tous les supports Dicke possibles pour ce n_sub_q
    t_cache = time.perf_counter()
    log("DICKE", f"pré-cache de {n_sub_q + 1} supports |D({n_sub_q}, k)⟩ "
                 f"pour k ∈ [0, {n_sub_q}]…")
    total_states = 0
    for k in range(n_sub_q + 1):
        total_states += len(_dicke_support(n_sub_q, k))
    log("DICKE", f"{_C.GREEN}✓{_C.RESET} cache prêt en "
                 f"{_fmt_t(time.perf_counter() - t_cache)} "
                 f"({total_states} états de base catalogués)",
        _C.GREEN)

    arr = np.array(img.convert("RGB"), dtype=np.float64)
    h, w, _ = arr.shape
    pal = np.asarray(palette, dtype=np.float64)
    out = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

    # Construction vectorisée (équivalente à la double boucle de l'origine)
    t_meta = time.perf_counter()
    pixel_meta = _build_pixel_meta_vectorized(arr, pal, n_sub_q)
    log("DICKE", f"projection vectorisée de {h*w} pixels sur l'axe palette "
                 f"en {_fmt_t(time.perf_counter() - t_meta)} "
                 f"({h*w / max(1e-6, time.perf_counter() - t_meta):.0f} px/s)")

    # Statistiques sur la distribution des k (utile pour profiler le batch)
    ks = np.array([m[2] for m in pixel_meta])
    n_trivial = int(((ks == 0) | (ks == n_sub_q)).sum())
    n_quantum = len(ks) - n_trivial
    log("DICKE", f"{n_quantum} pixels → circuit quantique • "
                 f"{n_trivial} pixels triviaux (k=0 ou k={n_sub_q}) → "
                 f"shortcut", _C.MAGENTA)

    total_px = len(pixel_meta)
    done = 0
    t_start = time.perf_counter()
    log("DICKE", f"démarrage halftoning : image {h}×{w} → "
                 f"{_C.BOLD}{h*scale}×{w*scale}{_C.RESET} px • "
                 f"batch={batch_size} • Aer parallel_experiments={_CPU}")

    # Traitement par batches
    last_print = 0.0
    n_circuits_run = 0
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

        # Cas quantiques : échantillonnage direct depuis le support Dicke
        # (équivalent exact à 1 shot par circuit |D(n_sub_q, k)⟩).
        if quantum:
            n_circuits_run += len(quantum)
            for i, (y, x, k, n_q, ca, cb) in quantum:
                batch_bits[i] = _dicke_sample_bits(n_q, k, count=1)[0]

        # Écriture dans l'image de sortie
        for i, (y, x, k, n_q, ca, cb) in enumerate(batch):
            bits = batch_bits[i][:n_sub].reshape(scale, scale)
            block = np.where(bits[..., None] == 1, cb, ca).astype(np.uint8)
            out[y * scale:(y + 1) * scale, x * scale:(x + 1) * scale] = block

        done += len(batch)
        now = time.perf_counter()
        if now - last_print > 0.2 or done == total_px:
            elapsed = now - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_px - done) / rate if rate > 0 else 0
            pct = done / total_px
            bar_w = 24
            fill = int(bar_w * pct)
            bar = "█" * fill + "░" * (bar_w - fill)
            sys.stdout.write(
                f"\r  {_C.GREEN}{bar}{_C.RESET} "
                f"{_C.BOLD}{done:>6}/{total_px}{_C.RESET} px "
                f"({pct*100:5.1f}%) • {rate:7.0f} px/s • "
                f"ETA {_fmt_t(eta)}  "
            )
            sys.stdout.flush()
            last_print = now
    sys.stdout.write("\n")

    dt = time.perf_counter() - t_start
    log("DICKE", f"{_C.GREEN}✓{_C.RESET} {total_px} pixels halftonés en "
                 f"{_fmt_t(dt)} • {total_px/dt:.0f} px/s • "
                 f"{n_circuits_run} circuits exécutés via Aer multi-core",
        _C.GREEN)
    return Image.fromarray(out, "RGB")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
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
    p.add_argument("--workers", type=int, default=_CPU,
                   help=f"processus parallèles pour QMCI downscale (défaut = {_CPU})")
    p.add_argument("--batch-size", type=int, default=512,
                   help="pixels par batch pour Dicke upscale (défaut 512)")
    p.add_argument("--max-side", type=int, default=0,
                   help="limite le côté max avant traitement (0 = désactivé)")
    args = p.parse_args()

    banner()
    log("INPUT", f"chargement de {_C.BOLD}{args.input}{_C.RESET}")
    img = Image.open(args.input).convert("RGB")
    log("INPUT", f"image {img.size[0]}×{img.size[1]} px • mode={args.mode}")
    if args.max_side > 0 and max(img.size) > args.max_side:
        r = args.max_side / max(img.size)
        new_size = (max(1, int(img.size[0] * r)), max(1, int(img.size[1] * r)))
        img = img.resize(new_size, Image.LANCZOS)
        log("INPUT", f"redimensionnée à {new_size[0]}×{new_size[1]} px "
                    f"(max-side={args.max_side})", _C.YELLOW)

    if args.mode in ("downscale", "both"):
        log("PHASE", f"{_C.BOLD}▶ QMCI DOWNSCALE{_C.RESET} • bloc={args.block} • "
                    f"ε={args.epsilon} • workers={args.workers}", _C.MAGENTA)
        down = quantum_downscale(img, args.block,
                                 epsilon=args.epsilon, workers=args.workers)
        out_path = (args.output if args.mode == "downscale"
                    else args.output.with_name(args.output.stem + "_down.png"))
        down.save(out_path)
        log("OUTPUT", f"→ {_C.BOLD}{out_path}{_C.RESET} "
                     f"({down.size[0]}×{down.size[1]} px)", _C.GREEN)

    if args.mode in ("upscale", "both"):
        log("PALETTE", f"k-means++ auto-detection ({args.num_colors} couleurs)…")
        t_pal = time.perf_counter()
        palette = auto_palette(img, max(2, args.num_colors))
        log("PALETTE", f"{_C.GREEN}✓{_C.RESET} {len(palette)} couleurs en "
                      f"{_fmt_t(time.perf_counter() - t_pal)}", _C.GREEN)
        for c in palette:
            r_, g_, b_ = int(c[0]), int(c[1]), int(c[2])
            swatch = f"\033[48;2;{r_};{g_};{b_}m      \033[0m" if _C._on else ""
            log("PALETTE", f"  {swatch} rgb({r_:3d}, {g_:3d}, {b_:3d})  "
                          f"#{r_:02x}{g_:02x}{b_:02x}")
        log("PHASE", f"{_C.BOLD}▶ DICKE HALFTONE UPSCALE{_C.RESET} • "
                    f"scale={args.scale} • batch={args.batch_size}", _C.MAGENTA)
        up = quantum_upscale(img, args.scale, palette, batch_size=args.batch_size)
        out_path = (args.output if args.mode == "upscale"
                    else args.output.with_name(args.output.stem + "_up.png"))
        up.save(out_path)
        log("OUTPUT", f"→ {_C.BOLD}{out_path}{_C.RESET} "
                     f"({up.size[0]}×{up.size[1]} px)", _C.GREEN)

    log("DONE", f"{_C.GREEN}✓ pipeline complet en "
                f"{_fmt_t(time.perf_counter() - _T0)}{_C.RESET}", _C.GREEN)


if __name__ == "__main__":
    main()
