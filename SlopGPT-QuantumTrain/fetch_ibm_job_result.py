"""Fetch an IBM Quantum Runtime job and save the generated 16x16 image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from qiskit_ibm_runtime import QiskitRuntimeService

from generate_quantum_sim_sample import (
    HARDCODED_IBM_QUANTUM_INSTANCE,
    HARDCODED_IBM_QUANTUM_TOKEN,
)
from train_quantum_sim_fashion_mnist_16 import IMAGE_SIZE, counts_to_row


OUT_DIR = Path(__file__).resolve().parent / "outputs"


def save_image(image: np.ndarray, path: Path, scale: int = 18) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    pil = pil.resize((image.shape[1] * scale, image.shape[0] * scale), Image.Resampling.NEAREST)
    pil.save(path)


def status_name(job: object) -> str:
    status = job.status()
    return getattr(status, "name", str(status))


def counts_shots(counts: dict[str, int]) -> int:
    total = int(sum(counts.values()))
    if total <= 0:
        raise ValueError("IBM job returned empty counts.")
    return total


def fetch_job_image(job_id: str, output_path: Path) -> Path | None:
    token = os.environ.get("IBM_QUANTUM_TOKEN") or HARDCODED_IBM_QUANTUM_TOKEN
    instance = os.environ.get("IBM_QUANTUM_INSTANCE") or HARDCODED_IBM_QUANTUM_INSTANCE

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
    )
    job = service.job(job_id)

    print("Job ID:", job.job_id())
    print("Status:", status_name(job))

    if "DONE" not in status_name(job).upper():
        print("The IBM job is not done yet. Run this command again later.")
        return None

    result = job.result()
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    row_shots: list[int] = []

    for row, pub_result in enumerate(result):
        counts = pub_result.data.c.get_counts()
        shots = counts_shots(counts)
        row_shots.append(shots)
        image[row] = counts_to_row(counts, shots)

    save_image(image, output_path)
    print("Row shots:", row_shots)
    print("Saved image:", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or OUT_DIR / f"ibm_job_{args.job_id}_result.png"
    fetch_job_image(args.job_id, output)


if __name__ == "__main__":
    main()


