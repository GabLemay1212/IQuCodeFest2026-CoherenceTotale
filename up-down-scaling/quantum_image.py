"""
Quantum image processing — HTTP server (Flask) on port 8081.

Endpoints
---------
GET  /ping              -> "ok"
POST /downscale         -> multipart (image=<file>, factor=2|4)  OR
                           JSON ({"image": "<base64 png/jpg>", "factor": 2|4})
                           Returns: image/png (downscaled by `factor`)
POST /upscale           -> same input shape; returns image/png upscaled by `factor`

Notes
-----
- The original quantum algorithms (QMCI for downscale, Dicke halftoning for
  upscale) are preserved exactly. Only the I/O layer is HTTP.
- Inputs are clamped to QIMG_MAX_SIDE px (default 64) so requests finish in
  reasonable time. Tune via env var, e.g. `set QIMG_MAX_SIDE=96`.
- Downscale parallelism uses a ThreadPoolExecutor (avoids Windows
  ProcessPoolExecutor spawn issues inside Flask request handlers).
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

def log(tag: str, msg: str) -> None:
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
_DICKE_SUPPORT_CACHE: dict[tuple[int, int], np.ndarray] = {}
_RNG = np.random.default_rng()


# ──────────────────────────────────────────────────────────────────────────────
# Dicke helpers (upscale)
# ──────────────────────────────────────────────────────────────────────────────
def _dicke_support(n_qubits: int, k: int) -> np.ndarray:
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
    support = _dicke_support(n_qubits, k)
    picks = support[_RNG.integers(0, len(support), size=count)]
    qs = np.arange(n_qubits, dtype=np.int64)
    return ((picks[:, None] >> qs[None, :]) & 1).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# QMCI worker (downscale)
# ──────────────────────────────────────────────────────────────────────────────
def _qmci_block(flat: np.ndarray, epsilon: float = 0.04) -> float:
    """Estimate the mean of `flat` (values in [0,1]) via Iterative Amplitude
    Estimation. flat length MUST be a power of two."""
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


def _build_pixel_meta(arr: np.ndarray, pal: np.ndarray, n_sub_q: int):
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
if _HAS_CORS:
    CORS(app)
else:
    @app.after_request
    def _cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp


MAX_SIDE = int(os.environ.get("QIMG_MAX_SIDE", "64"))
QMCI_EPSILON = float(os.environ.get("QIMG_EPSILON", "0.05"))
QMCI_WORKERS = int(os.environ.get("QIMG_WORKERS", "4"))
NUM_COLORS = int(os.environ.get("QIMG_NUM_COLORS", "6"))


def _read_image_from_request() -> Image.Image:
    """Accept multipart (image=<file>) or JSON ({image: "<base64>"})."""
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
    raw = (request.form.get("factor")
           or (request.get_json(silent=True) or {}).get("factor")
           or request.args.get("factor")
           or 2)
    f = int(raw)
    if f not in (2, 4):
        raise ValueError(f"factor must be 2 or 4, got {f}")
    return f


def _clamp(img: Image.Image) -> Image.Image:
    if MAX_SIDE > 0 and max(img.size) > MAX_SIDE:
        r = MAX_SIDE / max(img.size)
        new_size = (max(1, int(img.size[0] * r)),
                    max(1, int(img.size[1] * r)))
        img = img.resize(new_size, Image.LANCZOS)
        log("INPUT", f"resized to {new_size[0]}×{new_size[1]} (max-side={MAX_SIDE})")
    return img


def _png_response(img: Image.Image) -> Response:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), mimetype="image/png")


@app.get("/ping")
def ping():
    return "ok"


@app.route("/downscale", methods=["POST", "OPTIONS"])
def downscale():
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
