"""Tiny simulator-trained quantum generator for Fashion-MNIST 16x16.

This is intentionally small and slow compared with the main trainer. Its purpose
is to demonstrate training through quantum-circuit probabilities:

    class label + latent vector -> trainable RY angles -> simulator probabilities -> image

Gradients can use either parameter-shift or SPSA. Parameter-shift is cleaner for
explaining the derivative of RY probabilities:

    d p(phi) / d phi = 0.5 * [p(phi + pi/2) - p(phi - pi/2)]

The forward pass uses Qiskit Aer row circuits, so every training step is tied to
the quantum simulator rather than only to a closed-form classical formula.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "Datasets" / "Fashion-MNIST"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
CHECKPOINT_PATH = OUT_DIR / "quantum_sim_fashion_mnist_16x16_angles.npz"
META_PATH = OUT_DIR / "quantum_sim_fashion_mnist_16x16_angles_metadata.json"
LOSS_PATH = OUT_DIR / "quantum_sim_fashion_mnist_16x16_loss.png"
SAMPLE_PATH = OUT_DIR / "quantum_sim_fashion_mnist_16x16_sample.png"

IMAGE_SIZE = 16
NUM_CLASSES = 10

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
    missing = [str(path) for path in (images_path, labels_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Fashion-MNIST gzip IDX files are required. Missing: " + ", ".join(missing)
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
    return np.clip(np.asarray(pil, dtype=np.float32) / 255.0, 0.0, 1.0)


def load_subset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    images_path, labels_path = resolve_data_paths(args)
    images = load_idx_images(images_path)
    labels = load_idx_labels(labels_path)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"Image/label count mismatch: {images.shape[0]} vs {labels.shape[0]}")

    rng = np.random.default_rng(args.seed)
    chosen: list[int] = []
    for label in range(NUM_CLASSES):
        label_indices = np.flatnonzero(labels == label)
        if len(label_indices) == 0:
            continue
        take = min(args.samples_per_class, len(label_indices))
        chosen.extend(rng.choice(label_indices, size=take, replace=False).tolist())

    rng.shuffle(chosen)
    targets = np.stack([preprocess_image(images[index]) for index in chosen], axis=0)
    return targets, labels[chosen], images_path, labels_path


def row_circuit(phi_row: np.ndarray, row: int) -> QuantumCircuit:
    qreg = QuantumRegister(IMAGE_SIZE, f"q{row}")
    creg = ClassicalRegister(IMAGE_SIZE, "c")
    circuit = QuantumCircuit(qreg, creg)
    for col in range(IMAGE_SIZE):
        circuit.ry(float(phi_row[col]), qreg[col])
    circuit.measure(qreg, creg)
    return circuit


def counts_to_row(counts: dict[str, int], shots: int) -> np.ndarray:
    ones = np.zeros(IMAGE_SIZE, dtype=np.float32)
    for bitstring, count in counts.items():
        for col, bit in enumerate(bitstring[::-1]):
            if bit == "1":
                ones[col] += count
    return ones / shots


def simulator_forward(
    angles: np.ndarray,
    label: int,
    simulator: AerSimulator,
    *,
    shots: int,
) -> np.ndarray:
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for row in range(IMAGE_SIZE):
        circuit = row_circuit(angles[label, row], row)
        counts = simulator.run(circuit, shots=shots).result().get_counts()
        image[row] = counts_to_row(counts, shots)
    return np.clip(image, 0.0, 1.0)


def compose_angles(
    base_angles: np.ndarray,
    basis: np.ndarray,
    label: int,
    latent: np.ndarray,
) -> np.ndarray:
    phi = base_angles[label] + np.einsum("d,dhw->hw", latent, basis[label])
    return np.clip(phi, 0.02, np.pi - 0.02).astype(np.float32)


def simulator_forward_phi(
    phi: np.ndarray,
    simulator: AerSimulator,
    *,
    shots: int,
) -> np.ndarray:
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for row in range(IMAGE_SIZE):
        circuit = row_circuit(phi[row], row)
        counts = simulator.run(circuit, shots=shots).result().get_counts()
        image[row] = counts_to_row(counts, shots)
    return np.clip(image, 0.0, 1.0)


def parameter_shift_dp_dphi(phi: np.ndarray) -> np.ndarray:
    plus = np.sin((phi + np.pi / 2.0) / 2.0) ** 2
    minus = np.sin((phi - np.pi / 2.0) / 2.0) ** 2
    return (0.5 * (plus - minus)).astype(np.float32)


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def parameter_shift_update(
    base_angles: np.ndarray,
    basis: np.ndarray,
    label: int,
    latent: np.ndarray,
    target: np.ndarray,
    simulator: AerSimulator,
    args: argparse.Namespace,
) -> tuple[float, np.ndarray]:
    phi = compose_angles(base_angles, basis, label, latent)
    prediction = simulator_forward_phi(phi, simulator, shots=args.shots)
    error = prediction - target
    loss = mse(prediction, target)
    dp_dphi = parameter_shift_dp_dphi(phi)
    grad = (2.0 / float(IMAGE_SIZE * IMAGE_SIZE)) * error * dp_dphi
    base_angles[label] -= args.learning_rate * grad
    basis[label] -= args.learning_rate * args.basis_lr_scale * latent[:, None, None] * grad[None, :, :]
    base_angles[label] = np.clip(base_angles[label], 0.02, np.pi - 0.02)
    basis[label] = np.clip(basis[label], -args.basis_clip, args.basis_clip)
    return loss, prediction


def spsa_update(
    base_angles: np.ndarray,
    basis: np.ndarray,
    label: int,
    latent: np.ndarray,
    target: np.ndarray,
    simulator: AerSimulator,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[float, np.ndarray]:
    delta_base = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(IMAGE_SIZE, IMAGE_SIZE))
    delta_basis = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(args.latent_dim, IMAGE_SIZE, IMAGE_SIZE),
    )
    original_base = base_angles[label].copy()
    original_basis = basis[label].copy()

    base_angles[label] = np.clip(original_base + args.spsa_c * delta_base, 0.02, np.pi - 0.02)
    basis[label] = np.clip(
        original_basis + args.spsa_c * args.basis_lr_scale * delta_basis,
        -args.basis_clip,
        args.basis_clip,
    )
    plus_phi = compose_angles(base_angles, basis, label, latent)
    plus_prediction = simulator_forward_phi(plus_phi, simulator, shots=args.shots)
    plus_loss = mse(plus_prediction, target)

    base_angles[label] = np.clip(original_base - args.spsa_c * delta_base, 0.02, np.pi - 0.02)
    basis[label] = np.clip(
        original_basis - args.spsa_c * args.basis_lr_scale * delta_basis,
        -args.basis_clip,
        args.basis_clip,
    )
    minus_phi = compose_angles(base_angles, basis, label, latent)
    minus_prediction = simulator_forward_phi(minus_phi, simulator, shots=args.shots)
    minus_loss = mse(minus_prediction, target)

    scale = (plus_loss - minus_loss) / (2.0 * args.spsa_c)
    base_angles[label] = np.clip(
        original_base - args.learning_rate * scale * delta_base,
        0.02,
        np.pi - 0.02,
    )
    basis[label] = np.clip(
        original_basis - args.learning_rate * args.basis_lr_scale * scale * delta_basis,
        -args.basis_clip,
        args.basis_clip,
    )
    return 0.5 * (plus_loss + minus_loss), 0.5 * (plus_prediction + minus_prediction)


def angles_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, 0.02, 0.98)
    return (2.0 * np.arcsin(np.sqrt(probabilities))).astype(np.float32)


def initialize_angles(
    args: argparse.Namespace,
    targets: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    if args.init == "class-average":
        angles = np.zeros((NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        for label in range(NUM_CLASSES):
            class_targets = targets[labels == label]
            if len(class_targets) == 0:
                mean_image = np.full((IMAGE_SIZE, IMAGE_SIZE), 0.5, dtype=np.float32)
            else:
                mean_image = class_targets.mean(axis=0)
            angles[label] = angles_from_probabilities(mean_image)
        angles += rng.normal(0.0, args.init_noise, size=angles.shape).astype(np.float32)
    else:
        angles = np.full((NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE), np.pi / 2.0, dtype=np.float32)
        angles += rng.normal(0.0, 0.08, size=angles.shape).astype(np.float32)
    basis = rng.normal(
        0.0,
        args.basis_init_scale,
        size=(NUM_CLASSES, args.latent_dim, IMAGE_SIZE, IMAGE_SIZE),
    ).astype(np.float32)
    return np.clip(angles, 0.02, np.pi - 0.02), np.clip(basis, -args.basis_clip, args.basis_clip)


def load_or_initialize(
    args: argparse.Namespace,
    targets: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, list[float]]:
    if args.reset:
        for path in (CHECKPOINT_PATH, META_PATH, LOSS_PATH, SAMPLE_PATH):
            if path.exists():
                path.unlink()

    if args.resume and CHECKPOINT_PATH.exists():
        data = np.load(CHECKPOINT_PATH)
        base_angles = data["angles"].astype(np.float32)
        if "basis" in data:
            basis = data["basis"].astype(np.float32)
            if basis.shape[1] != args.latent_dim:
                raise ValueError(
                    "Checkpoint latent_dim does not match. "
                    f"checkpoint={basis.shape[1]}, requested={args.latent_dim}. Use --reset."
                )
        else:
            rng = np.random.default_rng(args.seed)
            basis = rng.normal(
                0.0,
                args.basis_init_scale,
                size=(NUM_CLASSES, args.latent_dim, IMAGE_SIZE, IMAGE_SIZE),
            ).astype(np.float32)
        return base_angles, basis, int(data["step"]), data["losses"].astype(float).tolist()

    print(f"initializing angles with {args.init}")
    base_angles, basis = initialize_angles(args, targets, labels)
    return base_angles, basis, 0, []


def save_grayscale(image: np.ndarray, path: Path, *, scale: int = 18) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    pil = pil.resize((image.shape[1] * scale, image.shape[0] * scale), Image.Resampling.NEAREST)
    pil.save(path)


def sample_latent(rng: np.random.Generator, args: argparse.Namespace) -> np.ndarray:
    return rng.normal(0.0, args.latent_scale, size=(args.latent_dim,)).astype(np.float32)


def augment_target(
    target: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> np.ndarray:
    augmented = target
    if args.augment_shift > 0:
        shift_y = int(rng.integers(-args.augment_shift, args.augment_shift + 1))
        shift_x = int(rng.integers(-args.augment_shift, args.augment_shift + 1))
        augmented = np.roll(augmented, shift=(shift_y, shift_x), axis=(0, 1))
        if shift_y > 0:
            augmented[:shift_y, :] = 0.0
        elif shift_y < 0:
            augmented[shift_y:, :] = 0.0
        if shift_x > 0:
            augmented[:, :shift_x] = 0.0
        elif shift_x < 0:
            augmented[:, shift_x:] = 0.0
    if args.augment_noise > 0:
        augmented = augmented + rng.normal(0.0, args.augment_noise, size=augmented.shape).astype(np.float32)
    return np.clip(augmented, 0.0, 1.0)


def save_checkpoint(
    angles: np.ndarray,
    basis: np.ndarray,
    step: int,
    losses: list[float],
    args: argparse.Namespace,
    images_path: Path,
    labels_path: Path,
    sample_image: np.ndarray | None,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHECKPOINT_PATH,
        angles=angles.astype(np.float32),
        basis=basis.astype(np.float32),
        step=np.asarray(step, dtype=np.int64),
        losses=np.asarray(losses, dtype=np.float32),
    )
    META_PATH.write_text(
        json.dumps(
            {
                "dataset": "Fashion-MNIST",
                "demo_type": "quantum-simulator training",
                "encoding": "class label + latent vector -> trainable RY angles",
                "probability": "measured P(qubit=1)",
                "optimizer": args.optimizer,
                "init": args.init,
                "init_noise": args.init_noise,
                "latent_dim": args.latent_dim,
                "latent_scale": args.latent_scale,
                "basis_init_scale": args.basis_init_scale,
                "basis_lr_scale": args.basis_lr_scale,
                "basis_clip": args.basis_clip,
                "augment_shift": args.augment_shift,
                "augment_noise": args.augment_noise,
                "spsa_c": args.spsa_c,
                "image_size": IMAGE_SIZE,
                "num_classes": NUM_CLASSES,
                "classes": FASHION_CLASSES,
                "shots": args.shots,
                "epochs": args.epochs,
                "samples_per_class": args.samples_per_class,
                "learning_rate": args.learning_rate,
                "step": step,
                "latest_loss": losses[-1] if losses else None,
                "train_images": str(images_path),
                "train_labels": str(labels_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if sample_image is not None:
        save_grayscale(sample_image, SAMPLE_PATH)
    if losses:
        fig, axis = plt.subplots(1, 1, figsize=(6.2, 3.6))
        axis.plot(losses)
        axis.set_title("Quantum Simulator Training Loss")
        axis.set_xlabel("Step")
        axis.set_ylabel("MSE")
        axis.grid(alpha=0.25)
        fig.savefig(LOSS_PATH, dpi=160, bbox_inches="tight")
        plt.close(fig)


def train(args: argparse.Namespace) -> None:
    targets, labels, images_path, labels_path = load_subset(args)
    angles, basis, step, losses = load_or_initialize(args, targets, labels)
    simulator = AerSimulator(seed_simulator=args.seed)
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(targets))
    sample_image: np.ndarray | None = None

    print(f"loaded tiny subset: {len(targets)} images ({args.samples_per_class}/class)")
    print(
        f"shots={args.shots}, epochs={args.epochs}, learning_rate={args.learning_rate}, "
        f"optimizer={args.optimizer}, init={args.init}"
    )

    for epoch in range(args.epochs):
        rng.shuffle(indices)
        print(f"epoch {epoch + 1}/{args.epochs}")
        for index in indices:
            label = int(labels[index])
            latent = sample_latent(rng, args)
            target = augment_target(targets[index], rng, args)
            if args.optimizer == "spsa":
                loss, prediction = spsa_update(angles, basis, label, latent, target, simulator, rng, args)
            else:
                loss, prediction = parameter_shift_update(angles, basis, label, latent, target, simulator, args)
            losses.append(loss)

            step += 1
            sample_image = prediction

            if step % args.log_every == 0:
                print(f"step {step:05d} class={FASHION_CLASSES[label]:>11} loss={loss:.6f}")

            if args.max_steps and step >= args.max_steps:
                save_checkpoint(angles, basis, step, losses, args, images_path, labels_path, sample_image)
                print(f"max steps reached; checkpoint saved: {CHECKPOINT_PATH}")
                print(f"sample saved: {SAMPLE_PATH}")
                return

    save_checkpoint(angles, basis, step, losses, args, images_path, labels_path, sample_image)
    print(f"checkpoint saved: {CHECKPOINT_PATH}")
    print(f"sample saved: {SAMPLE_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-images", type=Path, default=None)
    parser.add_argument("--train-labels", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.8)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--latent-scale", type=float, default=1.0)
    parser.add_argument("--basis-init-scale", type=float, default=0.04)
    parser.add_argument("--basis-lr-scale", type=float, default=0.35)
    parser.add_argument("--basis-clip", type=float, default=0.45)
    parser.add_argument("--augment-shift", type=int, default=1)
    parser.add_argument("--augment-noise", type=float, default=0.02)
    parser.add_argument("--optimizer", choices=("parameter-shift", "spsa"), default="spsa")
    parser.add_argument("--init", choices=("class-average", "random"), default="class-average")
    parser.add_argument("--init-noise", type=float, default=0.03)
    parser.add_argument("--spsa-c", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()
    if args.reset:
        args.resume = False
    return args


if __name__ == "__main__":
    train(parse_args())

    
