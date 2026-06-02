"""
quantum_image.py — Quantum-assisted image processing HTTP server (Flask) on port 8081.

Overview
--------
This module implements a Flask-based HTTP API that exposes two quantum image-processing
operations — **downscaling** and **upscaling** — built on top of IBM's Qiskit framework.

Algorithms
----------
Downscaling — Quantum Monte Carlo Integration (QMCI):
    Each block of pixels is treated as a probability distribution over a uniform
    superposition of index states. An ``IterativeAmplitudeEstimation`` (IAE) circuit
    estimates the *mean* intensity of the block per colour channel. This replaces a
    classical average with a quantum amplitude-estimation subroutine, giving a
    quadratic speedup in query complexity over classical Monte Carlo sampling.

Upscaling — Dicke-state halftoning:
    Each source pixel is blended between its two nearest palette colours. The blend
    ratio ``t`` determines how many of the ``n_sub_q`` sub-pixels in the output block
    should take the "high" colour. A Dicke state |D(n, k)⟩ — the uniform
    superposition of all n-bit strings with exactly k ones — is sampled to choose
    *which* sub-pixels get the high colour, spreading ink-like dithering uniformly
    rather than clustering it spatially.

Endpoints
---------
GET  /ping
    Health check. Returns the plain string ``"ok"``.

POST /downscale
    Accepts multipart ``(image=<file>, factor=2|4)`` or JSON
    ``{"image": "<base64 png/jpg>", "factor": 2|4}``.
    Returns an ``image/png`` downscaled by ``factor`` in each dimension.

POST /upscale
    Same input shape as ``/downscale``.
    Returns an ``image/png`` upscaled by ``factor`` in each dimension.

Environment Variables
---------------------
QIMG_MAX_SIDE : int, default 64
    Maximum allowed side length (px) after clamping. Larger images are
    proportionally resized *before* quantum processing. Increase with caution —
    runtime grows roughly as O(MAX_SIDE^2) for downscale.
QIMG_EPSILON : float, default 0.05
    IAE target precision ``epsilon`` per QMCI oracle call. Smaller values raise
    accuracy but increase circuit depth and wall-clock time (scales as O(1/epsilon)).
QIMG_WORKERS : int, default 4
    Number of threads in the ``ThreadPoolExecutor`` used for parallel QMCI oracle
    calls. Each thread runs an independent Qiskit ``StatevectorSampler`` simulation.
QIMG_NUM_COLORS : int, default 6
    Number of palette colours extracted by the k-means ``auto_palette`` routine
    before upscaling.

Notes
-----
- The original quantum algorithms (QMCI for downscale, Dicke halftoning for
  upscale) are preserved exactly. Only the I/O layer is HTTP.
- Downscale parallelism uses a ``ThreadPoolExecutor`` (avoids Windows
  ``ProcessPoolExecutor`` spawn issues inside Flask request handlers).
- CORS is handled by ``flask-cors`` when available, falling back to manual
  ``Access-Control-Allow-*`` headers.

Dependencies
------------
qiskit, qiskit-algorithms, flask, flask-cors (optional), pillow, numpy

Example Usage
-------------
Start the server::

    python quantum_image.py --port 8081

Downscale via curl::

    curl -X POST http://localhost:8081/downscale \\
         -F "image=@photo.png" -F "factor=2" --output small.png

Upscale with base64 JSON::

    import base64, json, requests
    data = base64.b64encode(open("small.png","rb").read()).decode()
    r = requests.post("http://localhost:8081/upscale",
                      json={"image": data, "factor": 2})
    open("big.png","wb").write(r.content)
"""

from __future__ import annotations

import base64
import io
import math
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Tiny logger
# ──────────────────────────────────────────────────────────────────────────────
_T0 = time.perf_counter()
"""float: Process-start timestamp used as the reference epoch for log messages."""


def log(tag: str, msg: str) -> None:
    """Write a timestamped, tagged message to stdout.

    All log output is flushed immediately so it remains readable even when the
    process is running behind a buffered pipe or a process manager such as
    ``gunicorn``.

    Parameters
    ----------
    tag : str
        A short label (up to 10 characters) identifying the subsystem, e.g.
        ``"BOOT"``, ``"QMCI"``, ``"DICKE"``, ``"REQ"``, ``"OK"``, ``"ERROR"``.
        The field is left-padded to 10 characters for columnar alignment.
    msg : str
        Free-form message text describing the event.

    Output Format
    -------------
    ``[  3.14s ] [QMCI      ] Processing block (0,0) channel R``

    Notes
    -----
    The elapsed time is measured from ``_T0``, set once at import time, so it
    reflects the total lifetime of the process rather than individual request
    duration.
    """
    elapsed = time.perf_counter() - _T0
    sys.stdout.write(f"[{elapsed:7.2f}s ] [{tag:<10}] {msg}\n")
    sys.stdout.flush()


# ──────────────────────────────────────────────────────────────────────────────
# Quantum deps (heavy import) — done once at startup
# ──────────────────────────────────────────────────────────────────────────────
log("BOOT", "loading qiskit…")
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation
log("BOOT", "qiskit ready")

_CPU = os.cpu_count() or 4
"""int: Number of logical CPU cores available; used as a fallback worker count."""

_DICKE_SUPPORT_CACHE: dict[tuple[int, int], np.ndarray] = {}
"""dict[tuple[int,int], np.ndarray]: Module-level cache mapping
``(n_qubits, k)`` to an integer array of all n-bit integers with exactly k ones.
Populated lazily by :func:`_dicke_support` and reused across all upscale calls."""

_RNG = np.random.default_rng()
"""numpy.random.Generator: Module-level default random number generator used
for Dicke-state sampling and k-means++ initialisation. Seeded from OS entropy
at import time; individual operations that require reproducibility pass an
explicit seed."""


# ──────────────────────────────────────────────────────────────────────────────
# Dicke helpers (upscale)
# ──────────────────────────────────────────────────────────────────────────────

def _dicke_support(n_qubits: int, k: int) -> np.ndarray:
    """Return the support set of the Dicke state |D(n_qubits, k)>.

    The Dicke state |D(n, k)> is the uniform superposition of all n-qubit
    computational basis states that have exactly k bits set to 1::

        |D(n,k)> = 1/sqrt(C(n,k))  *  sum_{|x|=k} |x>

    This function returns the *integer* representation of each basis state in
    that superposition, enabling classical sampling of the Dicke distribution
    without constructing the full quantum circuit.

    Results are memoised in ``_DICKE_SUPPORT_CACHE`` because the same
    ``(n_qubits, k)`` combination is queried for every pixel during upscaling.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits (bits) in each basis state. Must satisfy
        ``0 <= k <= n_qubits``.
    k : int
        Number of bits set to 1 (Hamming weight of each state in the support).

    Returns
    -------
    numpy.ndarray, shape (C(n_qubits, k),), dtype int64
        Sorted array of integers whose binary representation has exactly ``k``
        ones among the ``n_qubits`` least-significant bits.

    Examples
    --------
    >>> _dicke_support(3, 1)
    array([1, 2, 4])   # binary: 001, 010, 100
    >>> _dicke_support(3, 2)
    array([3, 5, 6])   # binary: 011, 101, 110

    Notes
    -----
    Edge cases ``k == 0`` (only the all-zeros state) and ``k == n_qubits``
    (only the all-ones state) are handled without calling
    :func:`itertools.combinations` to avoid creating large intermediate objects.
    """
    key = (n_qubits, k)
    if key not in _DICKE_SUPPORT_CACHE:
        if k == 0:
            arr = np.array([0], dtype=np.int64)
        elif k == n_qubits:
            arr = np.array([(1 << n_qubits) - 1], dtype=np.int64)
        else:
            arr = np.fromiter(
                (sum(1 << b for b in bits)
                 for bits in combinations(range(n_qubits), k)),
                dtype=np.int64,
                count=math.comb(n_qubits, k),
            )
        _DICKE_SUPPORT_CACHE[key] = arr
    return _DICKE_SUPPORT_CACHE[key]


def _dicke_sample_bits(n_qubits: int, k: int, count: int = 1) -> np.ndarray:
    """Sample ``count`` independent bit-strings from the Dicke distribution D(n, k).

    Each sampled string is drawn uniformly at random from all ``C(n_qubits, k)``
    n-bit strings that contain exactly ``k`` ones, which corresponds to measuring
    the Dicke state |D(n_qubits, k)> in the computational basis.

    This classical simulation of Dicke measurement is valid because the Dicke
    state is a *uniform* superposition — every basis state in its support has
    equal probability 1/C(n, k).

    Parameters
    ----------
    n_qubits : int
        Length of each bit-string. Must be >= 1.
    k : int
        Number of 1-bits in each sample. Must satisfy ``0 <= k <= n_qubits``.
    count : int, optional
        Number of independent samples to draw (default: 1).

    Returns
    -------
    numpy.ndarray, shape (count, n_qubits), dtype uint8
        Each row is an independent sample; column ``q`` is the value of qubit
        ``q`` (0 or 1). Column order follows the convention that column 0
        corresponds to bit 0 (LSB) of the integer representation.

    Examples
    --------
    >>> bits = _dicke_sample_bits(4, 2, count=3)
    >>> bits.shape
    (3, 4)
    >>> bits.sum(axis=1)
    array([2, 2, 2])   # each row has exactly k=2 ones

    Notes
    -----
    Uses the module-level ``_RNG`` (``numpy.random.default_rng()``) for
    randomness. The support is fetched (and cached) via :func:`_dicke_support`.
    """
    support = _dicke_support(n_qubits, k)
    picks = support[_RNG.integers(0, len(support), size=count)]
    qs = np.arange(n_qubits, dtype=np.int64)
    return ((picks[:, None] >> qs[None, :]) & 1).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# QMCI worker (downscale)
# ──────────────────────────────────────────────────────────────────────────────

def _qmci_block(flat: np.ndarray, epsilon: float = 0.04) -> float:
    """Estimate the mean of a pixel block using Quantum Monte Carlo Integration.

    Constructs a Qiskit state-preparation circuit that encodes the pixel values
    as rotation angles, then applies Iterative Amplitude Estimation (IAE) to
    estimate the mean pixel intensity for that block. This is the fundamental
    quantum subroutine for the downscaling operation.

    The circuit architecture is::

        n_idx index qubits (uniform superposition via H gates)
            |
        For each index x: multi-controlled RY(2*arcsin(sqrt(val[x]))) on ancilla
            |
        Ancilla qubit -- amplitude of |1> = sqrt(mean of vals)

    IAE then estimates the probability that the ancilla is in state |1>, which
    equals the mean of the encoded values.

    Parameters
    ----------
    flat : numpy.ndarray, shape (N,), dtype float64
        Pixel intensities in [0, 1]. Length **must be a power of two** (N = 2^n)
        so that ``n_idx = log2(N)`` index qubits span the full state space
        uniformly. Values are clamped to [0, 1] before encoding.
    epsilon : float, optional
        IAE target precision (default: 0.04). The returned estimate lies within
        epsilon of the true mean with probability >= 1 - alpha, where alpha = 0.05
        is hard-coded. Smaller epsilon requires more IAE iterations and deeper circuits.

    Returns
    -------
    float
        Estimated mean intensity, clamped to [0.0, 1.0].

    Short-circuit Conditions
    ------------------------
    - If all values < 1e-12 (effectively zero), returns ``0.0`` immediately.
    - If all values > 1 - 1e-12 (effectively one), returns ``1.0`` immediately.
      These fast paths avoid building and running a circuit for trivial blocks
      (e.g. pure black or pure white regions).

    Raises
    ------
    Any Qiskit exception propagated from ``IterativeAmplitudeEstimation.estimate``.

    Notes
    -----
    Each call creates a fresh ``StatevectorSampler`` and runs an independent
    IAE instance. This is intentional — it allows multiple calls to execute
    concurrently inside a ``ThreadPoolExecutor`` without shared mutable state.

    The circuit contains up to ``N`` multi-controlled RY gates (one per index),
    making circuit depth O(N * n_idx) in the worst case. For N = 16 (4x4 block,
    factor=4) this is already significant; larger blocks are not recommended
    without hardware QPU access.

    References
    ----------
    Brassard et al., "Quantum Amplitude Amplification and Estimation",
    AMS Contemporary Mathematics 305, 2002.
    """
    N = len(flat)
    n_idx = int(math.log2(N))  # index qubits

    if np.all(flat < 1e-12):
        return 0.0
    if np.all(flat > 1.0 - 1e-12):
        return 1.0

    qc = QuantumCircuit(n_idx + 1)
    for q in range(n_idx):
        qc.h(q)
    for x, val in enumerate(flat):
        v = float(np.clip(val, 0.0, 1.0))
        if v < 1e-12:
            continue
        theta = 2.0 * math.asin(math.sqrt(v))
        ctrl_bits = format(x, f"0{n_idx}b")
        flips = [q for q, b in enumerate(reversed(ctrl_bits)) if b == "0"]
        if flips:
            qc.x(flips)
        qc.mcry(theta, list(range(n_idx)), n_idx)
        if flips:
            qc.x(flips)

    problem = EstimationProblem(state_preparation=qc, objective_qubits=[n_idx])
    iae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon,
        alpha=0.05,
        sampler=StatevectorSampler(),
    )
    return float(np.clip(iae.estimate(problem).estimation, 0.0, 1.0))


def quantum_downscale(img: Image.Image, block: int,
                      epsilon: float = 0.04,
                      workers: int = 4) -> Image.Image:
    """Downscale an image by a factor of ``block`` using Quantum Monte Carlo Integration.

    Divides the source image into non-overlapping ``block x block`` tiles and
    estimates the mean intensity of each tile per RGB channel via the QMCI
    subroutine (:func:`_qmci_block`). The estimated means become the pixel
    values in the output image, producing an output ``block`` times smaller in
    each spatial dimension.

    Parallelism is achieved through a ``ThreadPoolExecutor``: all
    ``out_h x out_w x 3`` oracle calls are submitted as independent tasks and
    executed concurrently up to ``workers`` threads. A ``ProcessPoolExecutor``
    is deliberately avoided because Qiskit's ``StatevectorSampler`` uses
    global interpreter state that is unsafe to ``fork`` inside Flask request
    handlers (particularly on Windows).

    Parameters
    ----------
    img : PIL.Image.Image
        Input image. Converted to ``"RGB"`` internally; any mode is accepted.
    block : int
        Downscaling factor. Must be a **power of two >= 2** (e.g. 2 or 4).
        Each output pixel corresponds to a ``block x block`` input region, so
        the output is ``1/block`` the size in each dimension.
    epsilon : float, optional
        IAE precision target passed to each :func:`_qmci_block` call
        (default: 0.04). Trades accuracy against runtime.
    workers : int, optional
        Maximum number of concurrent QMCI threads (default: 4).

    Returns
    -------
    PIL.Image.Image
        RGB image of size ``(w // block, h // block)`` where ``w`` and ``h``
        are the (possibly cropped) dimensions of ``img``.

    Raises
    ------
    ValueError
        If ``block`` is not a power of two >= 2.
    ValueError
        If either spatial dimension of ``img`` is smaller than ``block``
        (i.e. ``h // block == 0`` or ``w // block == 0``).

    Side Effects
    ------------
    Logs progress to stdout via :func:`log` at every 50 completed oracle calls
    and at completion. Progress lines include throughput in oracles per second.

    Examples
    --------
    >>> from PIL import Image
    >>> img = Image.open("photo.png")
    >>> small = quantum_downscale(img, block=2, epsilon=0.05, workers=8)
    >>> small.size  # half the original dimensions
    (w//2, h//2)

    Complexity
    ----------
    Wall-clock time is approximately
    ``(out_h * out_w * 3) / workers * T_QMCI(block, epsilon)``
    where ``T_QMCI`` is the per-oracle time, which grows as O(block^2 / epsilon).
    For a 64x64 input with ``block=2``, ``epsilon=0.05``, ``workers=4``:
    roughly 32x32x3/4 = 768 oracle calls running in batches of 4.

    Notes
    -----
    The input is cropped to the largest rectangle whose side lengths are
    multiples of ``block`` before processing. Border pixels that don't fit a
    complete block are silently discarded.
    """
    if block < 2 or (block & (block - 1)) != 0:
        raise ValueError("block must be a power of two ≥ 2")

    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    h2 = (h // block) * block
    w2 = (w // block) * block
    if h2 == 0 or w2 == 0:
        raise ValueError(
            f"image {h}×{w} too small for block={block}; "
            f"need both sides ≥ {block} px")
    arr = arr[:h2, :w2]
    out_h, out_w = h2 // block, w2 // block
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    tasks = []
    for by in range(out_h):
        for bx in range(out_w):
            y0, x0 = by * block, bx * block
            for c in range(3):
                blk = arr[y0:y0 + block, x0:x0 + block, c]
                flat = (blk.astype(np.float64).flatten() / 255.0)
                tasks.append((by, bx, c, flat))

    total = len(tasks)
    log("QMCI", f"{h}×{w} block={block} → {out_h}×{out_w} • "
                f"{total} oracles • workers={workers} • ε={epsilon}")
    t0 = time.perf_counter()
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_qmci_block, flat, epsilon): (by, bx, c)
                for by, bx, c, flat in tasks}
        for fut in as_completed(futs):
            by, bx, c = futs[fut]
            out[by, bx, c] = int(round(fut.result() * 255))
            done += 1
            if done % 50 == 0 or done == total:
                dt = time.perf_counter() - t0
                rate = done / dt if dt > 0 else 0
                log("QMCI", f"  {done}/{total} ({100*done/total:5.1f}%) "
                            f"• {rate:.1f}/s")

    log("QMCI", f"done in {time.perf_counter() - t0:.2f}s")
    return Image.fromarray(out, "RGB")


# ──────────────────────────────────────────────────────────────────────────────
# Dicke upscale
# ──────────────────────────────────────────────────────────────────────────────

def auto_palette(img: Image.Image, k: int) -> np.ndarray:
    """Extract a perceptually ordered colour palette from an image using k-means++.

    Runs a k-means++ initialisation followed by 15 Lloyd's-algorithm iterations
    on up to 20,000 randomly sampled pixels from the image. The ``k`` cluster
    centroids are returned sorted by luminance (darkest first) so that a
    monotone index into the palette corresponds to a rough brightness ordering.

    This palette is passed to :func:`quantum_upscale` to define the two nearest
    colours used in Dicke-state halftoning for each source pixel.

    Parameters
    ----------
    img : PIL.Image.Image
        Source image. Converted to ``"RGB"`` internally.
    k : int
        Number of palette colours to extract. Typical values: 4–8. Setting
        ``k`` too high risks under-populated clusters on small or low-variety
        images; setting it too low reduces dithering quality in upscaling.

    Returns
    -------
    numpy.ndarray, shape (k, 3), dtype float64
        Array of ``k`` RGB colour triplets in [0, 255], sorted by luminance
        (row 0 = darkest, row k-1 = brightest). Values are clipped to [0, 255]
        before return.

    Algorithm
    ---------
    1. Convert image to float64 RGB and flatten to (N, 3).
    2. If N > 20,000, draw a random subsample of 20,000 pixels (seeded with 0).
    3. K-means++ initialisation: place first centre uniformly at random, then
       iteratively choose each subsequent centre with probability proportional
       to squared distance from the nearest already-placed centre.
    4. Run Lloyd's algorithm for up to 15 iterations; stop early if cluster
       centres move by <= 0.5 on any channel (``np.allclose(atol=0.5)``).
    5. Sort centres by luminance ``L = 0.299*R + 0.587*G + 0.114*B``.

    Notes
    -----
    The random seed ``0`` is used *only* for the subsample selection, ensuring
    reproducible palette extraction regardless of module-level ``_RNG`` state.
    K-means++ initialisation still uses the seeded generator for determinism.

    Examples
    --------
    >>> pal = auto_palette(img, k=4)
    >>> pal.shape
    (4, 3)
    >>> pal[0]          # darkest colour
    array([ 12.,  10.,  15.])
    >>> pal[-1]         # brightest colour
    array([240., 235., 228.])
    """
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
            axis=1)
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


def _build_pixel_meta(arr: np.ndarray, pal: np.ndarray,
                      n_sub_q: int) -> list[tuple]:
    """Compute per-pixel Dicke parameters for the upscaling halftoning pass.

    For each source pixel, finds its two nearest palette colours (``ca``, ``cb``)
    and projects the pixel onto the line segment between them to obtain a blend
    parameter ``t`` in [0, 1]. That parameter is then quantised to an integer
    ``k`` in {0, ..., n_sub_q} representing how many of the ``n_sub_q`` output
    sub-pixels should take colour ``cb`` (the farther / brighter colour).

    The integer ``k`` is the Hamming weight of the Dicke state sampled during
    upscaling: ``k = 0`` means all sub-pixels are colour ``ca``; ``k = n_sub_q``
    means all sub-pixels are colour ``cb``; intermediate values produce a dithered mix.

    Parameters
    ----------
    arr : numpy.ndarray, shape (H, W, 3), dtype float64
        Source image pixel values in [0, 255].
    pal : numpy.ndarray, shape (P, 3), dtype float64
        Palette of ``P`` colour triplets, e.g. from :func:`auto_palette`.
    n_sub_q : int
        Number of Dicke qubits (= number of sub-pixels per output block,
        rounded down to the nearest power of two if necessary). Must be >= 1.

    Returns
    -------
    list of tuple
        One tuple per source pixel (row-major order), each containing::

            (y: int, x: int, k: int, n_sub_q: int,
             ca: np.ndarray shape (3,), cb: np.ndarray shape (3,))

        where ``y``, ``x`` are the pixel coordinates in ``arr``,
        ``k`` is the Dicke Hamming weight, and ``ca``, ``cb`` are the low and
        high palette colours respectively.

    Algorithm
    ---------
    1. Flatten ``arr`` to (H*W, 3) and compute L2 distance from every pixel to
       every palette colour in one vectorised broadcast.
    2. For each pixel, take the index of the nearest colour as ``i`` and the
       second-nearest as ``j``. Set ``ca = pal[i]``, ``cb = pal[j]``.
    3. Project pixel onto the segment ca -> cb:
       ``t = clip((pixel - ca) . (cb - ca) / ||cb - ca||^2, 0, 1)``.
    4. Quantise: ``k = clip(round(t * n_sub_q), 0, n_sub_q)``.

    Notes
    -----
    The projection step is numerically stable: when ``||cb - ca|| < 1e-12``
    (two identical palette colours), ``t`` defaults to 0.0 rather than NaN,
    and all sub-pixels take colour ``ca``.

    This function is O(H*W*P) in memory and time due to the full pairwise
    distance matrix. For very large images or palettes, consider chunked processing.
    """
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    d = np.linalg.norm(flat[:, None, :] - pal[None, :, :], axis=2)
    idx_sorted = np.argsort(d, axis=1)
    i_idx = idx_sorted[:, 0]
    j_idx = idx_sorted[:, 1]
    ca = pal[i_idx]
    cb = pal[j_idx]
    ab = cb - ca
    denom = (ab * ab).sum(axis=1)
    num = ((flat - ca) * ab).sum(axis=1)
    t = np.where(denom < 1e-12, 0.0,
                 num / np.where(denom < 1e-12, 1.0, denom))
    t = np.clip(t, 0.0, 1.0)
    k_arr = np.clip(np.round(t * n_sub_q).astype(int), 0, n_sub_q)
    ys, xs = np.divmod(np.arange(h * w), w)
    return [
        (int(ys[p]), int(xs[p]), int(k_arr[p]), n_sub_q, ca[p], cb[p])
        for p in range(h * w)
    ]


def quantum_upscale(img: Image.Image, scale: int,
                    palette: np.ndarray) -> Image.Image:
    """Upscale an image by ``scale`` using Dicke-state quantum halftoning.

    Each source pixel is expanded into a ``scale x scale`` output block. The
    block colours are determined by sampling a Dicke state whose Hamming weight
    encodes the blend ratio between the pixel's two nearest palette colours.
    This produces spatially uniform dithering, as opposed to ordered or
    error-diffusion halftoning, because the Dicke distribution places 1-bits
    uniformly rather than in a fixed pattern.

    Parameters
    ----------
    img : PIL.Image.Image
        Source image. Converted to ``"RGB"`` internally.
    scale : int
        Upscaling factor >= 2. The output is ``scale`` times larger in each
        spatial dimension, so a 32x32 source with ``scale=2`` produces 64x64.
    palette : numpy.ndarray, shape (P, 3), dtype float64
        Colour palette used for halftoning, e.g. from :func:`auto_palette`.
        Must contain at least 2 colours.

    Returns
    -------
    PIL.Image.Image
        RGB image of size ``(W*scale, H*scale)`` where ``W``, ``H`` are the
        dimensions of ``img``.

    Raises
    ------
    ValueError
        If ``scale < 2``.
    ValueError
        If ``n_sub_q = 2**floor(log2(scale^2))`` exceeds the hard cap of 24.
        (scale >= 6 triggers this with ``scale^2 = 36 -> n_sub_q = 32 > 24``.)

    Algorithm
    ---------
    1. Compute ``n_sub = scale^2``. Round down to nearest power of two to get
       ``n_sub_q`` (the number of Dicke qubits). For ``scale=2``, ``n_sub_q=4``;
       for ``scale=4``, ``n_sub_q=16``.
    2. Warm the Dicke support cache for all Hamming weights 0 ... n_sub_q.
    3. Call :func:`_build_pixel_meta` to obtain per-pixel blend parameters.
    4. For each pixel ``(y, x)`` with blend index ``k``:

       - Sample one bit-string from D(n_sub_q, k) via :func:`_dicke_sample_bits`.
       - Map each bit to colour ``cb`` (bit=1) or ``ca`` (bit=0).
       - Reshape the first ``n_sub`` bits into a ``scale x scale`` colour block.
       - Write the block to ``out[y*scale : (y+1)*scale, x*scale : (x+1)*scale]``.

    Notes
    -----
    The Dicke sampling step is *classical* — no actual quantum circuit is run
    during upscaling. The "quantum" aspect lies in the algorithmic origin of the
    Dicke state, which guarantees that the resulting dithering pattern is the
    classically optimal uniform distribution over all bit-strings of a given weight.

    When ``n_sub > n_sub_q`` (i.e. ``scale^2`` is not a power of two), only the
    first ``n_sub`` entries of the ``n_sub_q``-bit Dicke sample are used to fill
    the block. The remaining ``n_sub_q - n_sub`` bits are discarded. This slight
    asymmetry is acceptable for typical values of ``scale``.

    Side Effects
    ------------
    Logs progress via :func:`log` at start and completion, including elapsed time.

    Examples
    --------
    >>> from PIL import Image
    >>> import numpy as np
    >>> img = Image.open("small.png")
    >>> pal = auto_palette(img, k=6)
    >>> big = quantum_upscale(img, scale=2, palette=pal)
    >>> big.size  # 2x larger
    (img.width * 2, img.height * 2)
    """
    if scale < 2:
        raise ValueError("scale must be ≥ 2")

    n_sub = scale * scale
    if n_sub & (n_sub - 1) != 0:
        n_sub_q = 2 ** int(math.log2(n_sub))
    else:
        n_sub_q = n_sub

    MAX_QUBITS = 24
    if n_sub_q > MAX_QUBITS:
        raise ValueError(
            f"scale={scale} → {n_sub_q} Dicke qubits exceeds cap of {MAX_QUBITS}")

    # warm cache
    for k in range(n_sub_q + 1):
        _dicke_support(n_sub_q, k)

    arr = np.array(img.convert("RGB"), dtype=np.float64)
    h, w, _ = arr.shape
    pal = np.asarray(palette, dtype=np.float64)
    out = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

    pixel_meta = _build_pixel_meta(arr, pal, n_sub_q)
    log("DICKE", f"{h}×{w} → {h*scale}×{w*scale} • n_sub_q={n_sub_q} "
                 f"• {len(pixel_meta)} px")
    t0 = time.perf_counter()

    for (y, x, k, n_q, ca, cb) in pixel_meta:
        if k == 0:
            bits = np.zeros(n_q, dtype=np.uint8)
        elif k == n_q:
            bits = np.ones(n_q, dtype=np.uint8)
        else:
            bits = _dicke_sample_bits(n_q, k, count=1)[0]
        block = np.where(bits[:n_sub, None] == 1, cb, ca).astype(np.uint8)
        block = block.reshape(scale, scale, 3)
        out[y * scale:(y + 1) * scale, x * scale:(x + 1) * scale] = block

    log("DICKE", f"done in {time.perf_counter() - t0:.2f}s")
    return Image.fromarray(out, "RGB")


# ──────────────────────────────────────────────────────────────────────────────
# HTTP server
# ──────────────────────────────────────────────────────────────────────────────
from flask import Flask, request, jsonify, Response
try:
    from flask_cors import CORS
    _HAS_CORS = True
except ImportError:
    _HAS_CORS = False

app = Flask(__name__)
"""flask.Flask: The Flask application instance.

CORS is enabled either via the ``flask-cors`` extension (if installed) or
through a manual ``after_request`` hook that injects permissive
``Access-Control-Allow-*`` headers. This allows browser-based clients to call
the API from any origin without pre-flight failures.
"""

if _HAS_CORS:
    CORS(app)
else:
    @app.after_request
    def _cors(resp):
        """Add permissive CORS headers to every response.

        Used as a fallback when ``flask-cors`` is not installed. Allows
        cross-origin requests from any origin (``*``) with ``GET``, ``POST``,
        and ``OPTIONS`` methods and the ``Content-Type`` header.

        Parameters
        ----------
        resp : flask.Response
            The outgoing response object, mutated in-place.

        Returns
        -------
        flask.Response
            The same response object with CORS headers appended.
        """
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp


MAX_SIDE: int = int(os.environ.get("QIMG_MAX_SIDE", "64"))
"""int: Maximum image side length (px) before quantum processing.

Images larger than this value are proportionally resized (LANCZOS) so that
their longest dimension equals ``MAX_SIDE``. Set to 0 to disable clamping.
Controlled by the ``QIMG_MAX_SIDE`` environment variable (default: 64).

Warning
-------
Increasing this value raises processing time quadratically for downscale
and linearly for upscale. Values above 96 are not recommended without
a GPU-backed Qiskit simulator.
"""

QMCI_EPSILON: float = float(os.environ.get("QIMG_EPSILON", "0.05"))
"""float: IAE precision target epsilon for each QMCI oracle call.

Smaller values increase accuracy but increase the number of IAE iterations
and circuit depth, scaling as O(1/epsilon).
Controlled by the ``QIMG_EPSILON`` environment variable (default: 0.05).
"""

QMCI_WORKERS: int = int(os.environ.get("QIMG_WORKERS", "4"))
"""int: Number of concurrent QMCI threads per request.

Each thread independently simulates one Qiskit circuit for a single pixel
channel. Controlled by ``QIMG_WORKERS`` (default: 4). Setting this higher
than the number of CPU cores typically yields no benefit because Qiskit's
statevector simulation is already CPU-bound.
"""

NUM_COLORS: int = int(os.environ.get("QIMG_NUM_COLORS", "6"))
"""int: Number of palette colours extracted per upscale request.

Passed to :func:`auto_palette`. Controlled by ``QIMG_NUM_COLORS``
(default: 6).
"""


def _read_image_from_request() -> Image.Image:
    """Parse and return a PIL Image from the current Flask request.

    Supports two input modalities:

    1. **Multipart form-data** — A file field named ``"image"`` containing a
       PNG or JPEG (or any PIL-readable format). Accessed via
       ``request.files["image"]``.

    2. **JSON body** — A field named ``"image"`` whose value is a base64-encoded
       string of a PNG or JPEG file. Optional data-URL prefix
       (e.g. ``"data:image/png;base64,..."```) is stripped automatically before
       decoding.

    Returns
    -------
    PIL.Image.Image
        The decoded image, converted to ``"RGB"`` mode.

    Raises
    ------
    ValueError
        If neither a multipart file nor an ``"image"`` key in the JSON body is
        present in the request.
    PIL.UnidentifiedImageError
        If the provided bytes cannot be decoded as a recognised image format.
    binascii.Error
        If the base64 string is malformed.

    Notes
    -----
    Must be called within an active Flask request context.
    """
    if "image" in request.files:
        f = request.files["image"]
        return Image.open(f.stream).convert("RGB")
    data = request.get_json(silent=True) or {}
    b64 = data.get("image")
    if not b64:
        raise ValueError("missing 'image' (multipart file or base64 in JSON)")
    if "," in b64:  # data URL prefix
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _read_factor() -> int:
    """Parse and validate the ``factor`` parameter from the current Flask request.

    Reads ``factor`` from (in priority order):

    1. ``request.form`` (multipart form field)
    2. JSON body key ``"factor"``
    3. Query-string parameter ``?factor=``
    4. Default value ``2`` if none of the above is present

    Returns
    -------
    int
        The validated factor value; always one of ``{2, 4}``.

    Raises
    ------
    ValueError
        If the resolved factor is not ``2`` or ``4``.

    Notes
    -----
    Must be called within an active Flask request context.
    """
    raw = (request.form.get("factor")
           or (request.get_json(silent=True) or {}).get("factor")
           or request.args.get("factor")
           or 2)
    f = int(raw)
    if f not in (2, 4):
        raise ValueError(f"factor must be 2 or 4, got {f}")
    return f


def _clamp(img: Image.Image) -> Image.Image:
    """Resize ``img`` so its longest side does not exceed ``MAX_SIDE``.

    If ``MAX_SIDE == 0`` or ``max(img.size) <= MAX_SIDE``, the image is
    returned unchanged. Otherwise it is resized with LANCZOS resampling,
    preserving aspect ratio, so the longer dimension equals exactly ``MAX_SIDE``.
    Neither dimension is ever reduced below 1 px.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image in any mode/size.

    Returns
    -------
    PIL.Image.Image
        Possibly resized image; same mode as input.

    Side Effects
    ------------
    Logs the new size to stdout via :func:`log` when a resize is performed.

    Notes
    -----
    LANCZOS (formerly ANTIALIAS) is the highest-quality PIL downsampling filter,
    minimising aliasing artefacts before quantum processing.
    """
    if MAX_SIDE > 0 and max(img.size) > MAX_SIDE:
        r = MAX_SIDE / max(img.size)
        new_size = (max(1, int(img.size[0] * r)),
                    max(1, int(img.size[1] * r)))
        img = img.resize(new_size, Image.LANCZOS)
        log("INPUT", f"resized to {new_size[0]}×{new_size[1]} (max-side={MAX_SIDE})")
    return img


def _png_response(img: Image.Image) -> Response:
    """Serialise a PIL Image to a Flask PNG response.

    Encodes ``img`` as a lossless PNG into an in-memory buffer and wraps it in
    a Flask ``Response`` with MIME type ``image/png``.

    Parameters
    ----------
    img : PIL.Image.Image
        Image to serialise. Any mode is accepted (PIL handles mode conversion
        during ``save``).

    Returns
    -------
    flask.Response
        HTTP response with ``Content-Type: image/png`` and the PNG bytes as
        the body. No ``Content-Length`` header is set; Flask computes it from
        the response data.

    Notes
    -----
    PNG is chosen over JPEG because it is lossless, which is important when the
    downstream consumer may apply further quantum processing to the result.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), mimetype="image/png")


@app.get("/ping")
def ping():
    """Health-check endpoint.

    Returns the plain string ``"ok"`` (HTTP 200) to indicate that the server
    is running and Qiskit has loaded successfully. Useful for readiness probes
    in container orchestration systems (e.g. Kubernetes liveness/readiness
    checks, Docker health checks, load-balancer health endpoints).

    Returns
    -------
    str
        The literal string ``"ok"``.

    Example
    -------
    ::

        GET /ping HTTP/1.1
        Host: localhost:8081

        HTTP/1.1 200 OK
        Content-Type: text/html; charset=utf-8

        ok
    """
    return "ok"


@app.route("/downscale", methods=["POST", "OPTIONS"])
def downscale():
    """HTTP endpoint: quantum downscale an uploaded image.

    Accepts a POST request carrying an image and an optional downscaling factor,
    runs :func:`quantum_downscale` on the (clamped) input, and returns the
    result as a PNG.

    An OPTIONS pre-flight is answered with HTTP 204 (No Content) and the CORS
    headers injected by :func:`_cors`.

    Request Body
    ------------
    Multipart form-data::

        image=<file>   (required) PNG, JPEG, or any PIL-readable image
        factor=2|4     (optional, default 2)

    OR JSON::

        {
          "image": "<base64-encoded PNG or JPEG, optional data-URL prefix>",
          "factor": 2
        }

    Responses
    ---------
    200 OK
        ``Content-Type: image/png`` — the downscaled image.
    204 No Content
        Response to OPTIONS pre-flight; includes CORS headers.
    500 Internal Server Error
        ``Content-Type: application/json`` — ``{"error": "<message>"}`` if any
        step fails (bad factor, image too small, Qiskit error, etc.).

    Side Effects
    ------------
    Logs request parameters, QMCI progress, and final output size via
    :func:`log`. Prints a full traceback to stdout on error.

    Example
    -------
    ::

        curl -X POST http://localhost:8081/downscale \\
             -F "image=@photo.png" -F "factor=4" \\
             --output small.png
    """
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        img = _read_image_from_request()
        factor = _read_factor()
        log("REQ", f"/downscale factor={factor} input={img.size[0]}×{img.size[1]}")
        img = _clamp(img)
        # downscale: block = factor
        result = quantum_downscale(img, block=factor,
                                   epsilon=QMCI_EPSILON,
                                   workers=QMCI_WORKERS)
        log("OK", f"/downscale → {result.size[0]}×{result.size[1]}")
        return _png_response(result)
    except Exception as e:
        log("ERROR", f"/downscale : {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/upscale", methods=["POST", "OPTIONS"])
def upscale():
    """HTTP endpoint: quantum upscale an uploaded image using Dicke halftoning.

    Accepts a POST request carrying an image and an optional upscaling factor.
    Extracts a colour palette via :func:`auto_palette`, then runs
    :func:`quantum_upscale` on the (clamped) input and returns the result as
    a PNG.

    An OPTIONS pre-flight is answered with HTTP 204 (No Content) and the CORS
    headers injected by :func:`_cors`.

    Request Body
    ------------
    Multipart form-data::

        image=<file>   (required) PNG, JPEG, or any PIL-readable image
        factor=2|4     (optional, default 2)

    OR JSON::

        {
          "image": "<base64-encoded PNG or JPEG>",
          "factor": 2
        }

    Responses
    ---------
    200 OK
        ``Content-Type: image/png`` — the upscaled, halftoned image.
    204 No Content
        Response to OPTIONS pre-flight; includes CORS headers.
    500 Internal Server Error
        ``Content-Type: application/json`` — ``{"error": "<message>"}`` if any
        step fails (bad factor, scale exceeds Dicke qubit cap, etc.).

    Notes
    -----
    The palette is re-derived from the *clamped* image on every request. If you
    want a fixed palette across multiple requests, pre-compute it and pass it
    directly to :func:`quantum_upscale` outside the HTTP layer.

    Side Effects
    ------------
    Logs request parameters and Dicke progress via :func:`log`. Prints a full
    traceback on error.

    Example
    -------
    ::

        curl -X POST http://localhost:8081/upscale \\
             -F "image=@small.png" -F "factor=2" \\
             --output big.png
    """
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        img = _read_image_from_request()
        factor = _read_factor()
        log("REQ", f"/upscale factor={factor} input={img.size[0]}×{img.size[1]}")
        img = _clamp(img)
        palette = auto_palette(img, max(2, NUM_COLORS))
        result = quantum_upscale(img, scale=factor, palette=palette)
        log("OK", f"/upscale → {result.size[0]}×{result.size[1]}")
        return _png_response(result)
    except Exception as e:
        log("ERROR", f"/upscale : {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def main():
    """Parse command-line arguments and start the Flask development server.

    This is the entry point when the module is run directly via
    ``python quantum_image.py``. It configures Flask's built-in Werkzeug
    server with the options below. For production deployments, use a proper
    WSGI server (e.g. ``gunicorn quantum_image:app``).

    Command-line Arguments
    ----------------------
    --host : str, default ``"0.0.0.0"``
        Network interface to bind. Use ``"127.0.0.1"`` to restrict access to
        localhost only.
    --port : int, default 8081
        TCP port to listen on.
    --debug : flag
        Enable Werkzeug debug mode (auto-reloader is disabled regardless to
        avoid double-loading Qiskit; the interactive debugger is still active).

    Notes
    -----
    ``use_reloader=False`` is always set because the Qiskit import at module
    level is expensive (several seconds) and the reloader would trigger it on
    every detected file change. ``threaded=True`` allows multiple concurrent
    requests to be handled by the same process, which is important for the
    parallelism inside :func:`quantum_downscale`.

    Examples
    --------
    Default startup::

        python quantum_image.py

    Custom port, restricted to localhost, with debug mode::

        python quantum_image.py --host 127.0.0.1 --port 9000 --debug
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    log("SERVER", f"listening on http://{args.host}:{args.port}  "
                  f"(MAX_SIDE={MAX_SIDE}, workers={QMCI_WORKERS})")
    log("SERVER", "endpoints: GET /ping  POST /downscale  POST /upscale")
    app.run(host=args.host, port=args.port, debug=args.debug,
            threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
