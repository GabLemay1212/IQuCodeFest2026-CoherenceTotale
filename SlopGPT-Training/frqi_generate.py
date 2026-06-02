"""Generate from the 64x64 FRQI-angle checkpoint."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "POC_DiffusionModel"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frqi_tools import FRQI_SIZE, recommended_shots, simulate_frqi_channel, theta_to_intensity  # noqa: E402
from tiny_imagenet_adapter import match_tiny_imagenet_prompt  # noqa: E402
from train_frqi_tiny_imagenet_64 import CHECKPOINT_PATH, META_PATH  # noqa: E402


@dataclass(frozen=True)
class FRQISample:
    prompt: str
    class_id: str
    class_name: str
    image: np.ndarray
    latent_seed: int
    frqi_qubits_per_channel: int = 13


def model_available() -> bool:
    return CHECKPOINT_PATH.exists() and META_PATH.exists()


def generate(prompt: str, *, seed: int | None = None, latent_scale: float = 0.45) -> FRQISample | None:
    if not model_available():
        return None
    match = match_tiny_imagenet_prompt(prompt)
    if match is None:
        return None

    data = np.load(CHECKPOINT_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    class_bias = data["class_bias"]
    weights = data["weights"]
    latent_dim = int(meta["latent_dim"])

    latent_seed = int(seed if seed is not None else np.random.default_rng().integers(0, 2**31 - 1))
    rng = np.random.default_rng(latent_seed)
    latent = rng.normal(0.0, latent_scale, size=(latent_dim,)).astype(np.float32)

    theta = class_bias[match.label] + np.einsum("d,dyxc->yxc", latent, weights[match.label])
    theta = np.clip(theta, 0.02, np.pi / 2.0 - 0.02)
    image = theta_to_intensity(theta)

    return FRQISample(
        prompt=prompt,
        class_id=match.class_id,
        class_name=match.class_name,
        image=np.clip(image, 0.0, 1.0),
        latent_seed=latent_seed,
    )


def save_report(sample: FRQISample, output_path: Path) -> None:
    fig, axis = plt.subplots(1, 1, figsize=(5.2, 5.5))
    axis.imshow(sample.image, interpolation="nearest")
    axis.set_title(
        f"{sample.prompt} -> {sample.class_name}\n"
        f"FRQI-angle 64x64 RGB, 13 qubits/channel, seed {sample.latent_seed}"
    )
    axis.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def sample_difference(prompt: str, *, seed_a: int = 101, seed_b: int = 202) -> float | None:
    """Return mean absolute difference between two generated samples."""
    a = generate(prompt, seed=seed_a)
    b = generate(prompt, seed=seed_b)
    if a is None or b is None:
        return None
    return float(np.mean(np.abs(a.image - b.image)))


def simulate_red_channel_demo(sample: FRQISample, output_path: Path, *, samples_per_pixel: int = 16) -> None:
    """Run an actual FRQI simulation for the red channel of one sample.

    This is expensive compared with direct angle decoding because FRQI
    reconstruction needs enough shots to revisit each pixel address.
    """
    shots = recommended_shots(samples_per_pixel)
    recon = simulate_frqi_channel(sample.image[:, :, 0], shots=shots)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
    axes[0].imshow(sample.image[:, :, 0], cmap="Reds", vmin=0, vmax=1)
    axes[0].set_title("Target red channel")
    axes[0].axis("off")
    axes[1].imshow(recon.image, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title(f"FRQI measured red\n{shots} shots, observed {recon.observed_pixels}/{FRQI_SIZE * FRQI_SIZE}")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
