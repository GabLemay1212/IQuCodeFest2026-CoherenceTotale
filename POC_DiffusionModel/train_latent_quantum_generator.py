"""Train a small latent conditional quantum generator.

This is a more AI-like generator than the class-prototype model:

- It trains on many Tiny ImageNet examples per class.
- Each training image has a latent vector.
- The model learns class parameters + latent parameters.
- Sampling a new latent vector makes the same prompt produce different images.

The differentiable training model is:

    phi = class_bias[class] + latent @ weights[class]
    P(pixel=1) = sin(phi / 2)^2

Generation turns those learned probabilities into quantum RY gates and measures
the circuits with Qiskit.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tiny_imagenet_adapter import CACHE_DIR, load_tiny_imagenet_class_samples, match_tiny_imagenet_prompt


MODEL_PATH = CACHE_DIR / "latent_vqg_8x8_gray.npz"
MODEL_META_PATH = CACHE_DIR / "latent_vqg_8x8_gray_metadata.json"
LOSS_PATH = CACHE_DIR / "latent_vqg_8x8_gray_loss.png"


DEFAULT_PROMPTS = ["cat", "fish", "panda", "mushroom", "school bus"]
IMAGE_SIZE = 8
LATENT_DIM = 8


def _labels_from_prompts(prompts: list[str]) -> tuple[list[int], list[str], list[str]]:
    labels: list[int] = []
    class_ids: list[str] = []
    class_names: list[str] = []
    for prompt in prompts:
        match = match_tiny_imagenet_prompt(prompt)
        if match is None:
            print(f"Skipping prompt without Tiny ImageNet match: {prompt}")
            continue
        if match.label in labels:
            continue
        labels.append(match.label)
        class_ids.append(match.class_id)
        class_names.append(match.class_name)
    if not labels:
        raise RuntimeError("No Tiny ImageNet classes matched the training prompts.")
    return labels, class_ids, class_names


def train(
    *,
    prompts: list[str] | None = None,
    samples_per_class: int = 48,
    epochs: int = 900,
    learning_rate: float = 0.28,
    seed: int = 23,
) -> list[float]:
    """Train and save the latent quantum generator."""
    prompts = prompts or DEFAULT_PROMPTS
    labels, class_ids, class_names = _labels_from_prompts(prompts)
    samples_by_label = load_tiny_imagenet_class_samples(
        labels,
        samples_per_class=samples_per_class,
        size=IMAGE_SIZE,
        grayscale=True,
    )

    rng = np.random.default_rng(seed)
    class_count = len(labels)
    max_samples = min(len(samples_by_label[label]) for label in labels)
    targets = np.stack([samples_by_label[label][:max_samples] for label in labels], axis=0)
    latents = rng.normal(0.0, 1.0, size=(class_count, max_samples, LATENT_DIM)).astype(np.float32)

    # Invert the quantum probability map to initialize class bias near the class
    # mean. This makes training stable and lets weights learn variations.
    class_means = np.clip(targets.mean(axis=1), 0.02, 0.98)
    class_bias = (2.0 * np.arcsin(np.sqrt(class_means))).astype(np.float32)
    weights = rng.normal(0.0, 0.035, size=(class_count, LATENT_DIM, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)

    losses: list[float] = []
    normalizer = float(class_count * max_samples * IMAGE_SIZE * IMAGE_SIZE)
    for epoch in range(epochs):
        phi = class_bias[:, None, :, :] + np.einsum("cnd,cdhw->cnhw", latents, weights)
        phi = np.clip(phi, 0.02, np.pi - 0.02)
        probs = np.sin(phi / 2.0) ** 2
        error = probs - targets
        loss = float(np.mean(error**2))
        losses.append(loss)

        dprob_dphi = 0.5 * np.sin(phi)
        grad_phi = 2.0 * error * dprob_dphi / normalizer
        grad_bias = grad_phi.sum(axis=1)
        grad_weights = np.einsum("cnd,cnhw->cdhw", latents, grad_phi)

        class_bias -= learning_rate * grad_bias.astype(np.float32)
        weights -= learning_rate * grad_weights.astype(np.float32)
        class_bias = np.clip(class_bias, 0.02, np.pi - 0.02)
        weights = np.clip(weights, -1.2, 1.2)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:04d} loss={loss:.6f}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODEL_PATH,
        class_bias=class_bias.astype(np.float32),
        weights=weights.astype(np.float32),
    )
    MODEL_META_PATH.write_text(
        json.dumps(
            {
                "prompts": prompts,
                "labels": labels,
                "class_ids": class_ids,
                "class_names": class_names,
                "samples_per_class": max_samples,
                "latent_dim": LATENT_DIM,
                "image_size": IMAGE_SIZE,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "final_loss": losses[-1],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, axis = plt.subplots(1, 1, figsize=(6, 3.5))
    axis.plot(losses)
    axis.set_title("Latent Conditional Quantum Generator Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE")
    axis.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(LOSS_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved model: {MODEL_PATH}")
    print(f"saved metadata: {MODEL_META_PATH}")
    print(f"saved loss plot: {LOSS_PATH}")
    return losses


if __name__ == "__main__":
    train()
