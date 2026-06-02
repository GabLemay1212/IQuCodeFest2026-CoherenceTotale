"""Train a full Tiny ImageNet grayscale latent quantum generator.

This script is separate from the small POC. It is designed to use all
100,000 Tiny ImageNet training images in streaming mini-batches.

Model:
  class label + random latent vector -> 8x8 grayscale probabilities
  probabilities are later realized with quantum RY gates and measurements.

The training pass itself is differentiable classical optimization of quantum
gate angles. This keeps training feasible; generation still uses Qiskit
measurement sampling.
"""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "POC_DiffusionModel"))

from tiny_imagenet_adapter import DATA_DIR, _load_class_names, _load_dataset_label_ids  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs"
CHECKPOINT_PATH = OUT_DIR / "full_tiny_imagenet_latent_vqg_16x16_gray.npz"
META_PATH = OUT_DIR / "full_tiny_imagenet_latent_vqg_16x16_gray_metadata.json"
LOSS_PATH = OUT_DIR / "full_tiny_imagenet_latent_vqg_16x16_loss.png"

IMAGE_SIZE = 16
NUM_CLASSES = 200


def decode_grayscale_image(image_record: dict) -> np.ndarray:
    raw = image_record.get("bytes")
    if raw is None:
        raise ValueError("Image record is missing bytes.")

    img = Image.open(BytesIO(raw)).convert("L")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)

    return np.asarray(img, dtype=np.float32) / 255.0

def iter_tiny_imagenet_batches(batch_size: int):
    parquet_files = sorted(DATA_DIR.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No train parquet files found in {DATA_DIR}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["image", "label"]):
            rows = batch.to_pydict()
            for image_record, label in zip(rows["image"], rows["label"]):
                images.append(decode_grayscale_image(image_record))
                labels.append(int(label))
                if len(images) >= batch_size:
                    yield np.stack(images, axis=0), np.asarray(labels, dtype=np.int64)
                    images.clear()
                    labels.clear()

    if images:
        yield np.stack(images, axis=0), np.asarray(labels, dtype=np.int64)


def initialize_model(latent_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    class_bias = rng.normal(
        loc=np.pi / 2.0,
        scale=0.12,
        size=(NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE),
    ).astype(np.float32)
    weights = rng.normal(
        loc=0.0,
        scale=0.035,
        size=(NUM_CLASSES, latent_dim, IMAGE_SIZE, IMAGE_SIZE),
    ).astype(np.float32)
    return class_bias, weights


def load_or_initialize(latent_dim: int, seed: int, resume: bool):
    if resume and CHECKPOINT_PATH.exists():
        data = np.load(CHECKPOINT_PATH)
        print(f"resuming from {CHECKPOINT_PATH}")
        return data["class_bias"], data["weights"], int(data["step"])
    class_bias, weights = initialize_model(latent_dim, seed)
    return class_bias, weights, 0


def save_checkpoint(
    class_bias: np.ndarray,
    weights: np.ndarray,
    step: int,
    losses: list[float],
    *,
    latent_dim: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHECKPOINT_PATH,
        class_bias=class_bias.astype(np.float32),
        weights=weights.astype(np.float32),
        step=np.asarray(step, dtype=np.int64),
    )

    label_ids = _load_dataset_label_ids()
    class_names = _load_class_names(label_ids)
    META_PATH.write_text(
        json.dumps(
            {
                "image_size": IMAGE_SIZE,
                "num_classes": NUM_CLASSES,
                "latent_dim": latent_dim,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "step": step,
                "latest_loss": losses[-1] if losses else None,
                "label_ids": label_ids,
                "class_names": class_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if losses:
        fig, axis = plt.subplots(1, 1, figsize=(6.2, 3.6))
        axis.plot(losses)
        axis.set_title("Full Tiny ImageNet Latent VQG Training Loss")
        axis.set_xlabel("Logged step")
        axis.set_ylabel("MSE")
        axis.grid(alpha=0.25)
        plt.tight_layout()
        fig.savefig(LOSS_PATH, dpi=160, bbox_inches="tight")
        plt.close(fig)


def train(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    class_bias, weights, step = load_or_initialize(args.latent_dim, args.seed, args.resume)
    losses: list[float] = []

    for epoch in range(args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")
        for images, labels in iter_tiny_imagenet_batches(args.batch_size):
            batch_size = images.shape[0]
            latents = rng.normal(0.0, 1.0, size=(batch_size, args.latent_dim)).astype(np.float32)

            bias_batch = class_bias[labels]
            weights_batch = weights[labels]
            phi = bias_batch + np.einsum("bd,bdhw->bhw", latents, weights_batch)
            phi = np.clip(phi, 0.02, np.pi - 0.02)

            probs = np.sin(phi / 2.0) ** 2
            error = probs - images
            loss = float(np.mean(error**2))

            dprob_dphi = 0.5 * np.sin(phi)
            grad_phi = 2.0 * error * dprob_dphi / float(batch_size * IMAGE_SIZE * IMAGE_SIZE)
            grad_bias_batch = grad_phi
            grad_weights_batch = latents[:, :, None, None] * grad_phi[:, None, :, :]

            np.add.at(class_bias, labels, -args.learning_rate * grad_bias_batch.astype(np.float32))
            np.add.at(weights, labels, -args.learning_rate * grad_weights_batch.astype(np.float32))
            class_bias = np.clip(class_bias, 0.02, np.pi - 0.02)
            weights = np.clip(weights, -1.5, 1.5)

            step += 1
            if step % args.log_every == 0:
                losses.append(loss)
                print(f"step {step:06d} loss={loss:.6f}")

            if step % args.save_every == 0:
                save_checkpoint(
                    class_bias,
                    weights,
                    step,
                    losses,
                    latent_dim=args.latent_dim,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                )
                print(f"checkpoint saved at step {step}")

    save_checkpoint(
        class_bias,
        weights,
        step,
        losses,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    print(f"final checkpoint saved: {CHECKPOINT_PATH}")
    print(f"metadata saved: {META_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
