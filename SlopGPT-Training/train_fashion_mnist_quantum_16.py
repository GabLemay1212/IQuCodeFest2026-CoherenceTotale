"""Train a 16x16 grayscale Fashion-MNIST latent quantum generator.

The model keeps the project quantum-inspired structure:

    class label + latent vector -> quantum gate angles -> probabilities
    probability = sin(phi / 2)^2

Training targets are loaded directly from Fashion-MNIST gzip IDX files:

    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz

Images are resized from 28x28 to 16x16 and normalized to [0, 1].
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "Fashion-MNIST"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
CHECKPOINT_PATH = OUT_DIR / "fashion_mnist_latent_vqg_16x16_grayscale.npz"
META_PATH = OUT_DIR / "fashion_mnist_latent_vqg_16x16_grayscale_metadata.json"
LOSS_PATH = OUT_DIR / "fashion_mnist_latent_vqg_16x16_grayscale_loss.png"

IMAGE_SIZE = 16
NUM_CLASSES = 10
THRESHOLD = 0.35

FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def resolve_data_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    data_dir = Path(args.data_dir)
    images_path = Path(args.train_images) if args.train_images else data_dir / "train-images-idx3-ubyte.gz"
    labels_path = Path(args.train_labels) if args.train_labels else data_dir / "train-labels-idx1-ubyte.gz"

    if not images_path.exists() or not labels_path.exists():
        try:
            import torchvision  # noqa: F401
            from torchvision.datasets import FashionMNIST
        except Exception as exc:  # noqa: BLE001
            missing = []
            if not images_path.exists():
                missing.append(str(images_path))
            if not labels_path.exists():
                missing.append(str(labels_path))
            raise FileNotFoundError(
                "Fashion-MNIST gzip IDX files are required. Missing: "
                + ", ".join(missing)
                + ". Expected train-images-idx3-ubyte.gz and train-labels-idx1-ubyte.gz "
                "inside --data-dir, or install torchvision for optional download fallback."
            ) from exc

        print("local gzip IDX files missing; attempting torchvision FashionMNIST download fallback...")
        FashionMNIST(root=str(data_dir), train=True, download=True)
        fallback = data_dir / "FashionMNIST" / "raw"
        images_path = fallback / "train-images-idx3-ubyte.gz"
        labels_path = fallback / "train-labels-idx1-ubyte.gz"
        if not images_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                "torchvision download did not produce expected gzip IDX files. "
                f"Looked for {images_path} and {labels_path}."
            )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Fashion-MNIST labels file is required and was not found: {labels_path}"
        )

    return images_path, labels_path


def load_idx_images(images_path: Path) -> np.ndarray:
    with gzip.open(images_path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid IDX image magic number {magic}; expected 2051.")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def load_idx_labels(labels_path: Path) -> np.ndarray:
    with gzip.open(labels_path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid IDX label magic number {magic}; expected 2049.")
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
    if labels.shape[0] != count:
        raise ValueError(f"Expected {count} labels, got {labels.shape[0]}.")
    return labels.astype(np.int64)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(image.astype(np.uint8), mode="L")
    pil = pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def load_fashion_mnist_targets(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    images_path, labels_path = resolve_data_paths(args)
    images = load_idx_images(images_path)
    labels = load_idx_labels(labels_path)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"Image/label count mismatch: {images.shape[0]} vs {labels.shape[0]}")
    targets = np.stack([preprocess_image(image) for image in images], axis=0)
    return targets, labels, images_path, labels_path


def stable_latent_from_index(index: int, label: int, latent_dim: int) -> np.ndarray:
    key = f"fashion-mnist:{index}:{label}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(latent_dim,)).astype(np.float32)


def initialize_model(latent_dim: int, targets: np.ndarray, labels: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    class_bias = np.zeros((NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for label in range(NUM_CLASSES):
        class_targets = targets[labels == label]
        if len(class_targets) == 0:
            mean = np.full((IMAGE_SIZE, IMAGE_SIZE), 0.5, dtype=np.float32)
        else:
            mean = np.clip(class_targets.mean(axis=0), 0.02, 0.98)
        class_bias[label] = 2.0 * np.arcsin(np.sqrt(mean))

    weights = rng.normal(
        0.0,
        0.03,
        size=(NUM_CLASSES, latent_dim, IMAGE_SIZE, IMAGE_SIZE),
    ).astype(np.float32)
    return np.clip(class_bias, 0.02, np.pi - 0.02), weights


def load_or_initialize(
    args: argparse.Namespace,
    targets: np.ndarray,
    labels: np.ndarray,
):
    if args.reset:
        for path in (CHECKPOINT_PATH, META_PATH, LOSS_PATH):
            if path.exists():
                path.unlink()
        print("reset requested; previous Fashion-MNIST checkpoint removed")

    if args.resume and CHECKPOINT_PATH.exists():
        data = np.load(CHECKPOINT_PATH)
        if data["weights"].shape[1] != args.latent_dim:
            raise ValueError(
                "Checkpoint latent_dim does not match requested latent_dim. "
                f"checkpoint={data['weights'].shape[1]}, requested={args.latent_dim}. "
                "Use --reset to start a new model with this latent dimension."
            )
        print(f"resuming from {CHECKPOINT_PATH}")
        return data["class_bias"], data["weights"], int(data["step"])

    print("initializing Fashion-MNIST class angles from class means...")
    class_bias, weights = initialize_model(args.latent_dim, targets, labels, args.seed)
    return class_bias, weights, 0


def save_checkpoint(
    class_bias: np.ndarray,
    weights: np.ndarray,
    step: int,
    losses: list[float],
    args: argparse.Namespace,
    images_path: Path,
    labels_path: Path,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHECKPOINT_PATH,
        class_bias=class_bias.astype(np.float32),
        weights=weights.astype(np.float32),
        step=np.asarray(step, dtype=np.int64),
    )
    META_PATH.write_text(
        json.dumps(
            {
                "dataset": "Fashion-MNIST",
                "encoding": "latent quantum RY probability field",
                "probability": "sin(phi / 2)^2",
                "output_mode": "grayscale",
                "image_size": IMAGE_SIZE,
                "debug_binary_threshold": THRESHOLD,
                "num_classes": NUM_CLASSES,
                "classes": FASHION_CLASSES,
                "latent_dim": args.latent_dim,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "step": step,
                "latest_loss": losses[-1] if losses else None,
                "train_images": str(images_path),
                "train_labels": str(labels_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if losses:
        fig, axis = plt.subplots(1, 1, figsize=(6.2, 3.6))
        axis.plot(losses)
        axis.set_title("Fashion-MNIST 16x16 Grayscale VQG Loss")
        axis.set_xlabel("Logged step")
        axis.set_ylabel("MSE")
        axis.grid(alpha=0.25)
        plt.tight_layout()
        fig.savefig(LOSS_PATH, dpi=160, bbox_inches="tight")
        plt.close(fig)


def train(args: argparse.Namespace) -> None:
    targets, labels, images_path, labels_path = load_fashion_mnist_targets(args)
    class_bias, weights, step = load_or_initialize(args, targets, labels)
    rng = np.random.default_rng(args.seed)
    losses: list[float] = []
    indices = np.arange(targets.shape[0])

    for epoch in range(args.epochs):
        rng.shuffle(indices)
        print(f"epoch {epoch + 1}/{args.epochs}")
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            batch_targets = targets[batch_indices]
            batch_labels = labels[batch_indices]
            batch_size = len(batch_indices)
            base_latents = np.stack(
                [
                    stable_latent_from_index(int(index), int(label), args.latent_dim)
                    for index, label in zip(batch_indices, batch_labels)
                ],
                axis=0,
            )

            bias_batch = class_bias[batch_labels]
            weights_batch = weights[batch_labels]
            grad_bias = np.zeros_like(bias_batch, dtype=np.float32)
            grad_weights = np.zeros_like(weights_batch, dtype=np.float32)
            loss_accum = 0.0

            for latent_pass in range(args.latents_per_image):
                if latent_pass == 0:
                    latents = base_latents
                else:
                    latents = base_latents + rng.normal(
                        0.0,
                        args.latent_jitter,
                        size=base_latents.shape,
                    ).astype(np.float32)

                phi = bias_batch + np.einsum("bd,bdhw->bhw", latents, weights_batch)
                phi = np.clip(phi, 0.02, np.pi - 0.02)
                probs = np.sin(phi / 2.0) ** 2
                probs = np.clip(probs, 0.0, 1.0)

                error = probs - batch_targets
                loss_accum += float(np.mean(error**2))

                dprob_dphi = 0.5 * np.sin(phi)
                grad_phi = 2.0 * error * dprob_dphi
                grad_phi /= float(args.latents_per_image * batch_size * IMAGE_SIZE * IMAGE_SIZE)
                grad_bias += grad_phi.astype(np.float32)
                grad_weights += (latents[:, :, None, None] * grad_phi[:, None, :, :]).astype(np.float32)

            loss = loss_accum / float(args.latents_per_image)
            np.add.at(class_bias, batch_labels, -args.learning_rate * grad_bias)
            np.add.at(weights, batch_labels, -args.learning_rate * grad_weights)
            weights *= 1.0 - args.weight_decay
            class_bias = np.clip(class_bias, 0.02, np.pi - 0.02)
            weights = np.clip(weights, -1.5, 1.5)

            step += 1
            if step % args.log_every == 0:
                losses.append(loss)
                print(f"step {step:06d} loss={loss:.6f}")

            if step % args.save_every == 0:
                save_checkpoint(class_bias, weights, step, losses, args, images_path, labels_path)
                print(f"checkpoint saved at step {step}")

            if args.max_steps and step >= args.max_steps:
                save_checkpoint(class_bias, weights, step, losses, args, images_path, labels_path)
                print(f"max steps reached; checkpoint saved: {CHECKPOINT_PATH}")
                return

    save_checkpoint(class_bias, weights, step, losses, args, images_path, labels_path)
    print(f"final checkpoint saved: {CHECKPOINT_PATH}")
    print(f"metadata saved: {META_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-images", type=Path, default=None)
    parser.add_argument("--train-labels", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--latents-per-image", type=int, default=2)
    parser.add_argument("--latent-jitter", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    if args.reset:
        args.resume = False
    if args.latents_per_image < 1:
        raise ValueError("--latents-per-image must be >= 1")
    return args


if __name__ == "__main__":
    train(parse_args())
