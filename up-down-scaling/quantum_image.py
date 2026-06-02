"""
Traitement d'image par algorithmes quantiques.

Deux opérations indépendantes :

1) POST-FILTRAGE quantique  -> DOWNSCALING massif (image très floue)
   Pour chaque bloc BxB de l'image source, les intensités normalisées des
   pixels sont encodées en amplitudes dans un registre quantique
   (amplitude encoding). On applique ensuite un filtre passe-bas quantique
   (transformée de Hadamard sur tous les qubits de position) : la
   probabilité de mesurer l'état |0...0> devient proportionnelle au carré
   de la moyenne du bloc. Cette mesure constitue le pixel "downscalé".
   Plus B est grand, plus l'image résultante est petite et floue.

2) HALFTONING quantique -> UPSCALING massif (image très lisse)
   Chaque pixel source est projeté sur la droite reliant deux couleurs
   choisies (color_a, color_b), donnant t ∈ [0, 1]. Pour chaque sous-pixel
   de la grille upscalée (facteur S), on prépare un qubit dans
   RY(2*arcsin(sqrt(t)))|0> puis on le mesure :
     - 0 -> color_a
     - 1 -> color_b
   Le rendu visuel des SxS sous-pixels reproduit un dégradé entre les
   deux couleurs (teintes intermédiaires par densité de points), ce qui
   lisse fortement l'image.

Les deux opérations sont sélectionnables indépendamment via --mode.

Usage :
    python quantum_image.py --mode downscale --input in.jpg --output out.png --block 16
    python quantum_image.py --mode upscale   --input in.jpg --output out.png --scale 8 \
        --color-a 20,20,60 --color-b 240,220,180
    python quantum_image.py --mode both ...
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

SIM = AerSimulator()

# Circuit paramétré pour le halftoning (1 qubit), transpilé une seule fois.
_THETA = Parameter("theta")
_HALFTONE_QC = QuantumCircuit(1, 1)
_HALFTONE_QC.ry(_THETA, 0)
_HALFTONE_QC.measure(0, 0)
_HALFTONE_TQC = transpile(_HALFTONE_QC, SIM)


# ---------------------------------------------------------------------------
# 1) POST-FILTRAGE QUANTIQUE  (downscaling)
# ---------------------------------------------------------------------------
def quantum_post_filter_block(block: np.ndarray) -> float:
    """
    Filtre passe-bas quantique sur un bloc (carré, taille puissance de 2).
    Retourne l'intensité moyenne dans [0, 1] (un canal).

    Méthode :
      - amplitude encoding des intensités normalisées sur n qubits
        (n = log2(nombre de pixels du bloc))
      - transformée de Hadamard sur tous les qubits => l'amplitude de
        |0...0> vaut (1/sqrt(N)) * sum(amplitudes_in) = sqrt(sum(I)) * 1/N
      - la probabilité mesurée de |0...0> = (mean sqrt(I))^2
        => on prend sa racine pour récupérer une intensité moyenne lissée.
    """
    flat = block.astype(np.float64).flatten() / 255.0
    n_pixels = flat.size
    n_qubits = int(math.log2(n_pixels))

    # Vecteur d'état : amplitudes = sqrt(intensite) (le carré donne l'intensité,
    # cohérent avec une lecture probabiliste).
    amps = np.sqrt(np.clip(flat, 0.0, 1.0))
    norm = np.linalg.norm(amps)
    if norm < 1e-12:
        return 0.0
    amps = amps / norm

    qc = QuantumCircuit(n_qubits)
    qc.initialize(amps, range(n_qubits))
    # Filtre passe-bas quantique : Hadamard sur tous les qubits de position
    for q in range(n_qubits):
        qc.h(q)

    # Simulation directe par Statevector (rapide, pas de transpile à chaque appel)
    sv = Statevector.from_instruction(qc).data
    p0 = float(np.abs(sv[0]) ** 2)
    # p0 = (mean sqrt(I))^2 quand l'état est normalisé ; norm**2 = sum(I).
    # On divise par le nombre de pixels pour récupérer une vraie moyenne
    # dans [0, 1] (sinon l'intensité est multipliée par N -> image blanche).
    mean_intensity = p0 * (norm ** 2) / n_pixels
    return float(np.clip(mean_intensity, 0.0, 1.0))


def quantum_downscale(img: Image.Image, block: int) -> Image.Image:
    """Downscale une image RGB en appliquant le post-filtrage quantique
    sur chaque bloc BxB de chaque canal. B doit être une puissance de 2."""
    if block & (block - 1) != 0:
        raise ValueError("La taille du bloc doit être une puissance de 2.")

    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    # On rogne pour que h et w soient multiples de block
    h2, w2 = (h // block) * block, (w // block) * block
    arr = arr[:h2, :w2]
    out_h, out_w = h2 // block, w2 // block
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    total = out_h * out_w
    done = 0
    for by in range(out_h):
        for bx in range(out_w):
            y0, x0 = by * block, bx * block
            for c in range(3):
                blk = arr[y0:y0 + block, x0:x0 + block, c]
                mean = quantum_post_filter_block(blk)
                out[by, bx, c] = int(round(mean * 255))
            done += 1
        print(f"  post-filter: {done}/{total} blocs", end="\r")
    print()
    return Image.fromarray(out, "RGB")


# ---------------------------------------------------------------------------
# 2) HALFTONING QUANTIQUE  (upscaling avec teintes entre deux couleurs)
# ---------------------------------------------------------------------------
def project_on_color_axis(pixel: np.ndarray,
                          color_a: np.ndarray,
                          color_b: np.ndarray) -> float:
    """Projette un pixel RGB sur la droite (color_a -> color_b).
    Retourne t ∈ [0, 1] : 0 = color_a, 1 = color_b."""
    ab = color_b - color_a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return 0.0
    t = float(np.dot(pixel - color_a, ab) / denom)
    return max(0.0, min(1.0, t))


def quantum_halftone_samples(t: float, n_samples: int) -> np.ndarray:
    """Prépare un qubit dans RY(2*arcsin(sqrt(t)))|0> puis le mesure
    n_samples fois. Renvoie un tableau de bits (0 -> color_a, 1 -> color_b).
    P(1) = t exactement -> teintes intermédiaires par densité de points."""
    theta = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, t))))
    bound = _HALFTONE_TQC.assign_parameters({_THETA: theta})
    result = SIM.run(bound, shots=n_samples, memory=True).result()
    mem = result.get_memory(bound)
    return np.fromiter((int(b) for b in mem), dtype=np.uint8, count=n_samples)


def auto_palette(img: Image.Image, k: int) -> np.ndarray:
    """Auto-détecte une palette de k couleurs dominantes via k-means
    (Lloyd) sur un sous-échantillon des pixels RGB.
    Retourne un tableau (k, 3) de couleurs (float)."""
    arr = np.array(img.convert("RGB"), dtype=np.float64).reshape(-1, 3)
    rng = np.random.default_rng(0)
    if arr.shape[0] > 20000:
        sample = arr[rng.choice(arr.shape[0], 20000, replace=False)]
    else:
        sample = arr
    # Init k-means++ simplifié : premier point aléatoire, suivants proportionnels
    # à la distance au plus proche centre.
    centers = [sample[rng.integers(0, sample.shape[0])]]
    for _ in range(k - 1):
        d2 = np.min(((sample[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(-1), axis=1)
        probs = d2 / (d2.sum() + 1e-12)
        centers.append(sample[rng.choice(sample.shape[0], p=probs)])
    centers = np.array(centers, dtype=np.float64)
    # Lloyd
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
    # Tri par luminance pour stabilité d'affichage
    lum = centers @ np.array([0.299, 0.587, 0.114])
    centers = centers[np.argsort(lum)]
    return np.clip(centers, 0, 255)


def quantum_upscale(img: Image.Image, scale: int,
                    palette: np.ndarray) -> Image.Image:
    """Upscale par halftoning quantique multi-couleurs.

    Pour chaque pixel source :
      - on cherche les 2 couleurs de la palette les plus proches
      - on projette le pixel sur le segment qui les relie -> t ∈ [0, 1]
      - on prépare un qubit dans RY(2*arcsin(sqrt(t)))|0>, on le mesure
        scale*scale fois -> mix de sous-pixels entre ces deux couleurs.
    Toutes les teintes intermédiaires de la palette sont donc reproduites
    par densité de points (vrai halftoning quantique multi-teintes)."""
    if scale < 2:
        raise ValueError("scale doit être >= 2.")
    arr = np.array(img.convert("RGB"), dtype=np.float64)
    h, w, _ = arr.shape
    pal = np.asarray(palette, dtype=np.float64)

    out = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    n_sub = scale * scale
    total = h * w
    done = 0
    for y in range(h):
        for x in range(w):
            px = arr[y, x]
            # 2 couleurs de palette les plus proches
            d = np.linalg.norm(pal - px, axis=1)
            i, j = np.argsort(d)[:2]
            ca, cb = pal[i], pal[j]
            t = project_on_color_axis(px, ca, cb)
            bits = quantum_halftone_samples(t, n_sub).reshape(scale, scale)
            block = np.where(bits[..., None] == 1, cb, ca).astype(np.uint8)
            out[y * scale:(y + 1) * scale, x * scale:(x + 1) * scale] = block
            done += 1
        print(f"  halftone: {done}/{total} pixels", end="\r")
    print()
    return Image.fromarray(out, "RGB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_color(s: str) -> tuple[int, int, int]:
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("couleur attendue : R,G,B")
    return tuple(max(0, min(255, p)) for p in parts)  # type: ignore[return-value]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["downscale", "upscale", "both"],
                   required=True,
                   help="downscale = post-filtrage quantique ; "
                        "upscale = halftoning quantique ; both = les deux")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path,
                   help="Pour --mode both, deux fichiers sont écrits : "
                        "<output>_down.png et <output>_up.png")
    p.add_argument("--block", type=int, default=8,
                   help="taille du bloc pour downscale (puissance de 2, défaut 8)")
    p.add_argument("--scale", type=int, default=6,
                   help="facteur d'upscale pour halftoning (défaut 6)")
    p.add_argument("--num-colors", type=int, default=6,
                   help="nombre de couleurs auto-détectées dans la palette "
                        "halftoning (défaut 6)")
    p.add_argument("--max-side", type=int, default=0,
                   help="redimensionne d'abord pour limiter le côté max "
                        "avant traitement quantique (perf). 0 = aucun (défaut).")
    args = p.parse_args()

    img = Image.open(args.input).convert("RGB")
    if args.max_side > 0 and max(img.size) > args.max_side:
        r = args.max_side / max(img.size)
        img = img.resize((max(1, int(img.size[0] * r)),
                          max(1, int(img.size[1] * r))), Image.LANCZOS)
        print(f"Image redimensionnée à {img.size} pour le traitement quantique.")

    if args.mode in ("downscale", "both"):
        print(f"[Post-filtrage quantique] bloc = {args.block}")
        down = quantum_downscale(img, args.block)
        out_path = (args.output if args.mode == "downscale"
                    else args.output.with_name(args.output.stem + "_down.png"))
        down.save(out_path)
        print(f"  -> {out_path}  ({down.size[0]}x{down.size[1]})")

    if args.mode in ("upscale", "both"):
        palette = auto_palette(img, max(2, args.num_colors))
        pal_str = ", ".join(f"({int(c[0])},{int(c[1])},{int(c[2])})" for c in palette)
        print(f"[Halftoning quantique] scale={args.scale} "
              f"palette auto ({len(palette)} couleurs) : {pal_str}")
        up = quantum_upscale(img, args.scale, palette)
        out_path = (args.output if args.mode == "upscale"
                    else args.output.with_name(args.output.stem + "_up.png"))
        up.save(out_path)
        print(f"  -> {out_path}  ({up.size[0]}x{up.size[1]})")


if __name__ == "__main__":
    main()
