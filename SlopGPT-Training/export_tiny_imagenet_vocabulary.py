"""Export Tiny ImageNet class vocabulary for prompt matching."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "POC_DiffusionModel"))

from tiny_imagenet_adapter import _class_aliases, _load_class_names, _load_dataset_label_ids  # noqa: E402


OUTPUT_PATH = Path(__file__).resolve().parent / "tiny_imagenet_200_classes.txt"


def main() -> None:
    label_ids = _load_dataset_label_ids()
    class_names = _load_class_names(label_ids)

    lines = [
        "# Tiny ImageNet 200 classes",
        "# Format: index | WordNet id | class names | prompt words",
        "",
    ]
    for index, (label_id, class_name) in enumerate(zip(label_ids, class_names)):
        prompt_words = ", ".join(sorted(_class_aliases(class_name)))
        lines.append(f"{index:03d} | {label_id} | {class_name} | {prompt_words}")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(label_ids)} classes to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
