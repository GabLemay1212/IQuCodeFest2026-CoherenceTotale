"""Train an 8x8 RGB variational quantum generator on Tiny ImageNet targets.

This training loop learns one set of quantum RY angles per Tiny ImageNet class.
The model is intentionally small:

- output: 8x8 RGB
- one measured qubit per output pixel/channel
- probability model: P(1) = sin(theta / 2)^2
- loss: mean squared error to a Tiny ImageNet representative image

Run:
    python POC_DiffusionModel/train_quantum_generator.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tiny_imagenet_adapter import CACHE_DIR, ensure_tiny_imagenet_cache
from trained_quantum_generator import MODEL_META_PATH, MODEL_PATH, TRAINED_SHAPE


def _resize_targets(prototypes: np.ndarray) -> np.ndarray:
    targets = []
    for image in prototypes:
        pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
        pil = pil.resize(TRAINED_SHAPE[:2][::-1], Image.Resampling.BICUBIC)
        targets.append(np.asarray(pil, dtype=np.float32) / 255.0)
    return np.stack(targets, axis=0)


def train(
    *,
    epochs: int = 350,
    learning_rate: float = 1.35,
    seed: int = 11,
) -> tuple[np.ndarray, list[float]]:
    """Train all Tiny ImageNet class parameters."""
    prototypes, label_ids, class_names = ensure_tiny_imagenet_cache()
    targets = _resize_targets(prototypes)

    rng = np.random.default_rng(seed)
    theta = rng.normal(loc=np.pi / 2.0, scale=0.08, size=targets.shape).astype(np.float32)
    losses: list[float] = []

    for epoch in range(epochs):
        probs = np.sin(theta / 2.0) ** 2
        error = probs - targets
        loss = float(np.mean(error**2))
        losses.append(loss)

        # d/dtheta sin(theta/2)^2 = 0.5 * sin(theta)
        grad = 2.0 * error * (0.5 * np.sin(theta)) / np.prod(TRAINED_SHAPE)
        theta -= learning_rate * grad.astype(np.float32)
        theta = np.clip(theta, 0.02, np.pi - 0.02)

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:04d} loss={loss:.6f}")

    final_probs = np.sin(theta / 2.0) ** 2
    per_class_loss = np.mean((final_probs - targets) ** 2, axis=(1, 2, 3)).astype(float).tolist()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL_PATH, theta=theta.astype(np.float32))
    MODEL_META_PATH.write_text(
        json.dumps(
            {
                "label_ids": label_ids,
                "class_names": class_names,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "losses": per_class_loss,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return theta, losses


def save_loss_plot(losses: list[float]) -> None:
    path = CACHE_DIR / "trained_vqg_8x8_rgb_loss.png"
    fig, axis = plt.subplots(1, 1, figsize=(6, 3.5))
    axis.plot(losses)
    axis.set_title("Trained 8x8 Quantum Generator Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE")
    axis.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved loss plot: {path}")


def main() -> None:
    _, losses = train()
    save_loss_plot(losses)
    print(f"saved model: {MODEL_PATH}")
    print(f"saved metadata: {MODEL_META_PATH}")


if __name__ == "__main__":
    main()
