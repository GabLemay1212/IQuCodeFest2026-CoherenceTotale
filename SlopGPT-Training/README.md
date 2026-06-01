# SlopGPT Full Tiny ImageNet Training

This folder is for the larger grayscale training experiment. It keeps the
current `POC_DiffusionModel` code intact.

Goal:

```text
Use all 100,000 Tiny ImageNet training images
Train a class + latent conditional quantum generator
Generate varied 8x8 grayscale samples
```

## Export The 200 Promptable Classes

```bash
python SlopGPT-Training/export_tiny_imagenet_vocabulary.py
```

This writes:

```text
SlopGPT-Training/tiny_imagenet_200_classes.txt
```

Use that file to know which words/classes the dataset can support.

## Smoke Test Training

Run one small pass first:

```bash
python SlopGPT-Training/train_full_tiny_imagenet_quantum.py --epochs 1 --batch-size 128 --save-every 50 --log-every 10
```

This confirms the Parquet loader, resizing, gradients, and checkpoint writing.

## Full Training

Start with:

```bash
python SlopGPT-Training/train_full_tiny_imagenet_quantum.py --epochs 10 --batch-size 256 --latent-dim 16
```

Resume later:

```bash
python SlopGPT-Training/train_full_tiny_imagenet_quantum.py --epochs 10 --batch-size 256 --latent-dim 16 --resume
```

Outputs:

```text
SlopGPT-Training/outputs/full_tiny_imagenet_latent_vqg_8x8_gray.npz
SlopGPT-Training/outputs/full_tiny_imagenet_latent_vqg_8x8_gray_metadata.json
SlopGPT-Training/outputs/full_tiny_imagenet_latent_vqg_loss.png
```

## What The Model Learns

The model is:

```text
class label + random latent vector
  -> quantum RY gate angles
  -> measurement probabilities
  -> 8x8 grayscale image
```

Training uses mini-batches streamed from the local Parquet dataset. Generation
uses Qiskit measurements.

## Why 8x8 Grayscale?

This is the practical starting point for a quantum-simulated generator.

```text
8x8 grayscale = 64 output values
8x8 RGB       = 192 output values
32x32 RGB     = 3072 output values
```

The full dataset is already a big step. After this works, the next upgrades are
`8x8 RGB` or `16x16 grayscale`.
