"""Tiny ImageNet local dataset adapter.

The dataset in this repo is stored as Hugging Face-style Parquet files:

    tiny-imagenet/data/train-*.parquet

This adapter builds a small cache of 32x32 RGB class prototypes. A prototype is
the average of a few real images from one class. Prompt text is matched against
the 200 Tiny ImageNet class names.
"""

from __future__ import annotations

import json
import re
import runpy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "tiny-imagenet"
DATA_DIR = DATASET_DIR / "data"
CACHE_DIR = Path(__file__).resolve().parent / "outputs" / "cache"
PROTOTYPE_CACHE = CACHE_DIR / "tiny_imagenet_representatives_32x32_rgb_v2.npz"
META_CACHE = CACHE_DIR / "tiny_imagenet_representatives_metadata_v2.json"
IMAGE_SHAPE = (32, 32, 3)

PROMPT_ALIASES = {
    "fish": "goldfish",
    "panda": "lesser panda red panda panda bear cat bear",
    "red panda": "lesser panda red panda panda bear cat bear",
    "cat": "tabby cat",
    "dog": "golden retriever",
    "fruit": "orange lemon banana",
    "bird": "goose",
    "car": "sports car",
    "bus": "school bus",
    "truck": "moving van",
    "flower": "sunflower",
    "tree": "poplar",
    "boat": "gondola",
}


@dataclass(frozen=True)
class TinyImageNetMatch:
    label: int
    class_id: str
    class_name: str
    score: int
    prototype: np.ndarray


def _load_dataset_label_ids() -> list[str]:
    info = json.loads((DATASET_DIR / "dataset_infos.json").read_text(encoding="utf-8"))
    dataset_info = next(iter(info.values()))
    return list(dataset_info["features"]["label"]["names"])


def _load_class_names(label_ids: list[str]) -> list[str]:
    namespace = runpy.run_path(str(DATASET_DIR / "classes.py"))
    i2d = namespace["i2d"]
    return [i2d.get(class_id, class_id) for class_id in label_ids]


def _tokens(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "the",
        "of",
        "to",
        "make",
        "draw",
        "generate",
        "image",
        "picture",
        "photo",
        "quantum",
        "colored",
        "colour",
        "color",
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "pink",
        "white",
        "black",
        "brown",
        "gray",
        "grey",
    }
    return {tok for tok in re.findall(r"[a-zA-Z]+", text.lower()) if len(tok) > 1 and tok not in stop}


def _class_aliases(class_name: str) -> set[str]:
    aliases = set()
    for part in class_name.lower().split(","):
        aliases.update(_tokens(part))
    aliases.update(_tokens(class_name))
    return aliases


def _decode_image(image_record: dict) -> np.ndarray:
    raw = image_record.get("bytes")
    if raw is None:
        raise ValueError("Tiny ImageNet row does not contain image bytes.")
    img = Image.open(BytesIO(raw)).convert("RGB")
    img = img.resize(IMAGE_SHAPE[:2][::-1], Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def ensure_tiny_imagenet_cache(samples_per_class: int = 16) -> tuple[np.ndarray, list[str], list[str]]:
    """Build or load cached Tiny ImageNet class prototypes."""
    if PROTOTYPE_CACHE.exists() and META_CACHE.exists():
        cache = np.load(PROTOTYPE_CACHE)
        meta = json.loads(META_CACHE.read_text(encoding="utf-8"))
        return cache["prototypes"], meta["label_ids"], meta["class_names"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    label_ids = _load_dataset_label_ids()
    class_names = _load_class_names(label_ids)
    num_classes = len(label_ids)

    samples: list[list[np.ndarray]] = [[] for _ in range(num_classes)]
    parquet_files = sorted(DATA_DIR.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Tiny ImageNet train parquet found in {DATA_DIR}")

    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=512, columns=["image", "label"]):
            rows = batch.to_pydict()
            for image_record, label in zip(rows["image"], rows["label"]):
                label = int(label)
                if len(samples[label]) >= samples_per_class:
                    continue
                samples[label].append(_decode_image(image_record))
            if all(len(class_samples) >= samples_per_class for class_samples in samples):
                break
        if all(len(class_samples) >= samples_per_class for class_samples in samples):
            break

    valid = [len(class_samples) > 0 for class_samples in samples]
    if not all(valid):
        missing = [label_ids[idx] for idx, ok in enumerate(valid) if not ok]
        raise RuntimeError(f"Missing Tiny ImageNet examples for labels: {missing[:10]}")

    prototypes = np.zeros((num_classes, *IMAGE_SHAPE), dtype=np.float32)
    for label, class_samples in enumerate(samples):
        stack = np.stack(class_samples, axis=0)
        mean = stack.mean(axis=0)
        distances = np.mean((stack - mean) ** 2, axis=(1, 2, 3))
        prototypes[label] = stack[int(np.argmin(distances))]

    np.savez_compressed(PROTOTYPE_CACHE, prototypes=prototypes)
    META_CACHE.write_text(
        json.dumps(
            {
                "label_ids": label_ids,
                "class_names": class_names,
                "samples_per_class": samples_per_class,
                "cache_kind": "representative_image_closest_to_sample_mean",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return prototypes, label_ids, class_names


def match_tiny_imagenet_prompt(prompt: str) -> TinyImageNetMatch | None:
    """Return the best Tiny ImageNet class match for prompt text."""
    if not DATASET_DIR.exists():
        return None

    prototypes, label_ids, class_names = ensure_tiny_imagenet_cache()
    prompt_for_match = prompt
    lowered = prompt.lower()
    for alias, expanded in PROMPT_ALIASES.items():
        if alias in lowered:
            prompt_for_match = f"{prompt} {expanded}"
            break

    prompt_tokens = _tokens(prompt_for_match)
    if not prompt_tokens:
        return None

    best_label = -1
    best_score = 0.0
    for label, class_name in enumerate(class_names):
        aliases = _class_aliases(class_name)
        overlap = prompt_tokens & aliases
        score = float(len(overlap))
        for prompt_token in prompt_tokens:
            for alias in aliases:
                if prompt_token == "fish" and alias == "goldfish":
                    score += 1.25
                if prompt_token in alias and prompt_token != alias and len(prompt_token) >= 4:
                    score += 0.45
                elif alias in prompt_token and prompt_token != alias and len(alias) >= 4:
                    score += 0.35
        primary_name = class_name.split(",", 1)[0].lower()
        if primary_name in prompt_for_match.lower():
            score += 2.5
        if score > best_score:
            best_label = label
            best_score = score

    if best_label < 0 or best_score < 1.0:
        return None

    return TinyImageNetMatch(
        label=best_label,
        class_id=label_ids[best_label],
        class_name=class_names[best_label],
        score=best_score,
        prototype=prototypes[best_label],
    )


def load_tiny_imagenet_class_samples(
    labels: list[int],
    *,
    samples_per_class: int = 48,
    size: int = 8,
    grayscale: bool = True,
) -> dict[int, np.ndarray]:
    """Load multiple real images for selected labels from local Tiny ImageNet.

    Returns:
        dict[label] -> array shaped (N, size, size) if grayscale else
        (N, size, size, 3), normalized to [0, 1].
    """
    wanted = set(labels)
    samples: dict[int, list[np.ndarray]] = {label: [] for label in labels}
    parquet_files = sorted(DATA_DIR.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Tiny ImageNet train parquet found in {DATA_DIR}")

    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=512, columns=["image", "label"]):
            rows = batch.to_pydict()
            for image_record, label in zip(rows["image"], rows["label"]):
                label = int(label)
                if label not in wanted or len(samples[label]) >= samples_per_class:
                    continue

                raw = image_record.get("bytes")
                if raw is None:
                    continue
                img = Image.open(BytesIO(raw)).convert("RGB")
                if grayscale:
                    img = img.convert("L")
                img = img.resize((size, size), Image.Resampling.BICUBIC)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                samples[label].append(arr)

            if all(len(samples[label]) >= samples_per_class for label in labels):
                break
        if all(len(samples[label]) >= samples_per_class for label in labels):
            break

    return {label: np.stack(values, axis=0) for label, values in samples.items() if values}
