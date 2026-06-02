"""Train a 64x64 RGB FRQI-angle generator on Tiny ImageNet.

This model uses the FRQI intensity relation:

    intensity = sin(theta)^2

The trainable model predicts theta fields for 64x64 RGB images:

    class label + latent vector -> theta[y, x, channel]

At generation time, those theta fields can be encoded into FRQI circuits:

    12 position qubits + 1 color qubit per RGB channel

Training does not build FRQI circuits for every batch; it optimizes the FRQI
angle parameters directly, which is the only practical way to train on 100k
images locally.
"""

from __future__ import annotations

import argparse
import hashlib
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frqi_tools import FRQI_SIZE, intensity_to_theta, theta_to_intensity  # noqa: E402
from tiny_imagenet_adapter import DATA_DIR, _load_class_names, _load_dataset_label_ids  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs"
CHECKPOINT_PATH = OUT_DIR / "frqi_64x64_rgb_latent_vqg.npz"
META_PATH = OUT_DIR / "frqi_64x64_rgb_latent_vqg_metadata.json"
LOSS_PATH = OUT_DIR / "frqi_64x64_rgb_latent_vqg_loss.png"
NUM_CLASSES = 200
CHANNELS = 3


def decode_rgb_64(image_record: dict) -> np.ndarray:
    raw = image_record.get("bytes")
    if raw is None:
        raise ValueError("Image record is missing bytes.")
    img = Image.open(BytesIO(raw)).convert("RGB")
    img = img.resize((FRQI_SIZE, FRQI_SIZE), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def latent_from_image_bytes(raw: bytes, latent_dim: int) -> np.ndarray:
    """Create a stable latent vector for one training image.

    The earlier trainer sampled a new latent every time an image appeared. That
    makes the best solution collapse toward a class average. A stable latent
    lets the model associate different images with different locations in latent
    space.
    """
    digest = hashlib.sha256(raw).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(latent_dim,)).astype(np.float32)


def augment_batch(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply lightweight augmentation to reduce pixel memorization."""
    augmented = images.copy()
    batch_size = augmented.shape[0]

    flip_mask = rng.random(batch_size) < 0.5
    augmented[flip_mask] = augmented[flip_mask, :, ::-1, :]

    brightness = rng.uniform(0.90, 1.10, size=(batch_size, 1, 1, 1)).astype(np.float32)
    contrast = rng.uniform(0.90, 1.10, size=(batch_size, 1, 1, 1)).astype(np.float32)
    mean = augmented.mean(axis=(1, 2), keepdims=True)
    augmented = (augmented - mean) * contrast + mean
    augmented = augmented * brightness

    noise = rng.normal(0.0, 0.01, size=augmented.shape).astype(np.float32)
    augmented = augmented + noise
    return np.clip(augmented, 0.0, 1.0)


def iter_batches(batch_size: int, latent_dim: int):
    parquet_files = sorted(DATA_DIR.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No train parquet files found in {DATA_DIR}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    latents: list[np.ndarray] = []
    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["image", "label"]):
            rows = batch.to_pydict()
            for image_record, label in zip(rows["image"], rows["label"]):
                raw = image_record.get("bytes")
                if raw is None:
                    continue
                images.append(decode_rgb_64(image_record))
                labels.append(int(label))
                latents.append(latent_from_image_bytes(raw, latent_dim))
                if len(images) >= batch_size:
                    yield (
                        np.stack(images, axis=0),
                        np.asarray(labels, dtype=np.int64),
                        np.stack(latents, axis=0),
                    )
                    images.clear()
                    labels.clear()
                    latents.clear()

    if images:
        yield (
            np.stack(images, axis=0),
            np.asarray(labels, dtype=np.int64),
            np.stack(latents, axis=0),
        )


def initialize(latent_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    class_bias = initialize_class_bias_from_data()
    weights = rng.normal(
        0.0,
        0.012,
        size=(NUM_CLASSES, latent_dim, FRQI_SIZE, FRQI_SIZE, CHANNELS),
    ).astype(np.float32)
    return np.clip(class_bias, 0.02, np.pi / 2.0 - 0.02), weights


def initialize_class_bias_from_data(samples_per_class: int = 8) -> np.ndarray:
    """Initialize FRQI angles from representative real images.

    A random FRQI angle field looks like colored static. For a minimum useful
    model, start each class from a real representative image, then let training
    refine class and latent variation.
    """
    samples: list[list[np.ndarray]] = [[] for _ in range(NUM_CLASSES)]
    parquet_files = sorted(DATA_DIR.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No train parquet files found in {DATA_DIR}")

    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=512, columns=["image", "label"]):
            rows = batch.to_pydict()
            for image_record, label in zip(rows["image"], rows["label"]):
                label = int(label)
                if len(samples[label]) >= samples_per_class:
                    continue
                samples[label].append(decode_rgb_64(image_record))
            if all(len(class_samples) >= samples_per_class for class_samples in samples):
                break
        if all(len(class_samples) >= samples_per_class for class_samples in samples):
            break

    class_images = np.zeros((NUM_CLASSES, FRQI_SIZE, FRQI_SIZE, CHANNELS), dtype=np.float32)
    for label, class_samples in enumerate(samples):
        if not class_samples:
            class_images[label] = 0.5
            continue
        stack = np.stack(class_samples, axis=0)
        mean = stack.mean(axis=0)
        distances = np.mean((stack - mean) ** 2, axis=(1, 2, 3))
        class_images[label] = stack[int(np.argmin(distances))]

    return intensity_to_theta(class_images)


def load_or_initialize(latent_dim: int, seed: int, resume: bool):
    if resume and CHECKPOINT_PATH.exists():
        data = np.load(CHECKPOINT_PATH)
        if data["weights"].shape[1] != latent_dim:
            raise ValueError(
                "Checkpoint latent_dim does not match requested latent_dim. "
                f"checkpoint={data['weights'].shape[1]}, requested={latent_dim}. "
                "Use --reset to start a new model with this latent dimension."
            )
        print(f"resuming from {CHECKPOINT_PATH}")
        return data["class_bias"], data["weights"], int(data["step"])
    print("initializing FRQI class angles from Tiny ImageNet representatives...")
    class_bias, weights = initialize(latent_dim, seed)
    return class_bias, weights, 0


def save_checkpoint(
    class_bias: np.ndarray,
    weights: np.ndarray,
    step: int,
    losses: list[float],
    args: argparse.Namespace,
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
                "encoding": "FRQI angle field",
                "image_size": FRQI_SIZE,
                "channels": CHANNELS,
                "num_classes": NUM_CLASSES,
                "latent_dim": args.latent_dim,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "step": step,
                "latest_loss": losses[-1] if losses else None,
                "label_ids": label_ids,
                "class_names": class_names,
                "note": "Training optimizes FRQI theta fields; generation may build FRQI circuits.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if losses:
        fig, axis = plt.subplots(1, 1, figsize=(6.2, 3.6))
        axis.plot(losses)
        axis.set_title("FRQI 64x64 RGB Training Loss")
        axis.set_xlabel("Logged step")
        axis.set_ylabel("MSE")
        axis.grid(alpha=0.25)
        plt.tight_layout()
        fig.savefig(LOSS_PATH, dpi=160, bbox_inches="tight")
        plt.close(fig)


def train(args: argparse.Namespace) -> None:
    if args.reset:
        for path in (CHECKPOINT_PATH, META_PATH, LOSS_PATH):
            if path.exists():
                path.unlink()
        print("reset requested; previous FRQI checkpoint files removed")

    rng = np.random.default_rng(args.seed)
    class_bias, weights, step = load_or_initialize(args.latent_dim, args.seed, args.resume)
    losses: list[float] = []

    for epoch in range(args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")
        for images, labels, base_latents in iter_batches(args.batch_size, args.latent_dim):
            batch_size = images.shape[0]
            if args.augment:
                images = augment_batch(images, rng)

            bias_batch = class_bias[labels]
            weights_batch = weights[labels]

            grad_bias = np.zeros_like(bias_batch, dtype=np.float32)
            grad_weights = np.zeros_like(weights_batch, dtype=np.float32)
            loss_accum = 0.0

            for latent_pass in range(args.latents_per_image):
                if latent_pass == 0:
                    latents = base_latents
                else:
                    jitter = rng.normal(
                        0.0,
                        args.latent_jitter,
                        size=base_latents.shape,
                    ).astype(np.float32)
                    latents = base_latents + jitter
                theta = bias_batch + np.einsum("bd,bdyxc->byxc", latents, weights_batch)
                theta = np.clip(theta, 0.02, np.pi / 2.0 - 0.02)
                probs = theta_to_intensity(theta)

                error = probs - images
                loss_accum += float(np.mean(error**2))

                # d/dtheta sin(theta)^2 = sin(2 theta)
                grad_theta = 2.0 * error * np.sin(2.0 * theta)
                grad_theta /= float(
                    args.latents_per_image * batch_size * FRQI_SIZE * FRQI_SIZE * CHANNELS
                )
                grad_bias += grad_theta.astype(np.float32)
                grad_weights += (latents[:, :, None, None, None] * grad_theta[:, None, :, :, :]).astype(np.float32)

            loss = loss_accum / float(args.latents_per_image)

            np.add.at(class_bias, labels, -args.learning_rate * grad_bias.astype(np.float32))
            np.add.at(weights, labels, -args.learning_rate * grad_weights.astype(np.float32))
            weights *= (1.0 - args.weight_decay)
            class_bias = np.clip(class_bias, 0.02, np.pi / 2.0 - 0.02)
            weights = np.clip(weights, -0.65, 0.65)

            step += 1
            if step % args.log_every == 0:
                losses.append(loss)
                print(f"step {step:06d} loss={loss:.6f}")

            if step % args.save_every == 0:
                save_checkpoint(class_bias, weights, step, losses, args)
                print(f"checkpoint saved at step {step}")

            if args.max_steps and step >= args.max_steps:
                save_checkpoint(class_bias, weights, step, losses, args)
                print(f"max steps reached; checkpoint saved: {CHECKPOINT_PATH}")
                return

    save_checkpoint(class_bias, weights, step, losses, args)
    print(f"final checkpoint saved: {CHECKPOINT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--latents-per-image", type=int, default=2)
    parser.add_argument("--latent-jitter", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    if args.reset:
        args.resume = False
    if args.latents_per_image < 1:
        raise ValueError("--latents-per-image must be >= 1")
    return args


if __name__ == "__main__":
    train(parse_args())
