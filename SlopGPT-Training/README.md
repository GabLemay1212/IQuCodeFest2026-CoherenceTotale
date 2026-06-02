# SlopGPT Full Tiny ImageNet Training

This folder is for the larger grayscale training experiment. It keeps the
current `POC_DiffusionModel` code intact.

Goal:

```text
Use all 100,000 Tiny ImageNet training images
Train a class + latent conditional quantum generator
Generate varied 8x8 grayscale samples
```

## Fashion-MNIST 16x16 Grayscale Model

This is the faster, cleaner training path for recognizable outputs. It reads
Fashion-MNIST directly from gzip IDX files, resizes 28x28 images to 16x16, and
keeps grayscale pixel values from 0.0 to 1.0.

Expected local files:

```text
Fashion-MNIST/train-images-idx3-ubyte.gz
Fashion-MNIST/train-labels-idx1-ubyte.gz
```

Classes:

```text
0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot
```

Smoke test:

```bash
python SlopGPT-Training/train_fashion_mnist_quantum_16.py --epochs 1 --batch-size 128 --max-steps 100 --data-dir "C:\Personal Files\Universite\Quantum\IQuCodeFest2026-CoherenceTotale\Fashion-MNIST"
```

Full training:

```bash
python SlopGPT-Training/train_fashion_mnist_quantum_16.py --epochs 20 --batch-size 256 --latent-dim 16 --data-dir "C:\Personal Files\Universite\Quantum\IQuCodeFest2026-CoherenceTotale\Fashion-MNIST"
```

Resume:

```bash
python SlopGPT-Training/train_fashion_mnist_quantum_16.py --epochs 20 --batch-size 256 --latent-dim 16 --resume --data-dir "C:\Personal Files\Universite\Quantum\IQuCodeFest2026-CoherenceTotale\Fashion-MNIST"
```

Prompts:

```text
t-shirt, tshirt, shirt, trouser, pants, pullover, dress, coat,
sandal, sneaker, shoe, bag, ankle boot, boot
```

The Flask backend in this folder now uses only this Fashion-MNIST model. Prompts
outside the Fashion-MNIST vocabulary return an unsupported-prompt error instead
of falling back to Tiny ImageNet, FRQI, shapes, or the older POC models.

The generator still saves both debug outputs:

```text
*_grayscale.png
*_binary_debug.png
```

The UI returns the final `*_grayscale.png` image.

UI generation modes:

```text
Fast         128 shots, 1 candidate,   latent scale 0.7
Balanced     512 shots, 2 candidates,  latent scale 1.0
DeepThinking 1024 shots, 24 candidates, latent scale 1.7
RealQuantumDemo 128 shots, 1 candidate, latent scale 1.0, IBM hardware
```

Shots reduce quantum measurement noise. Candidates and latent scale create the
visible differences between repeated generations. DeepThinking intentionally
does more work and should take several seconds before the chat receives the PNG.

RealQuantumDemo submits the 16 row circuits for one 16x16 image to IBM Quantum
hardware. Training still runs locally. Install the runtime package first:

```bash
pip install qiskit-ibm-runtime
```

Then set these environment variables before starting the Flask backend:

```powershell
$env:IBM_QUANTUM_TOKEN="your_ibm_quantum_token"
$env:IBM_QUANTUM_INSTANCE="your_instance_crn"
$env:IBM_QUANTUM_BACKEND="optional_backend_name"
python SlopGPT-Training/SlopGPT.py
```

`IBM_QUANTUM_BACKEND` is optional. If omitted, the runtime service tries to pick
a least-busy real backend with at least 16 qubits.

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

## FRQI 64x64 RGB Version

FRQI is useful because it reduces qubit count:

```text
64x64 image = 4096 pixels
position qubits = log2(4096) = 12
color qubit = 1
FRQI grayscale = 13 qubits
FRQI RGB = 3 separate 13-qubit circuits, one per channel
```

The tradeoff is that the circuit can be deep: exact FRQI encoding needs a
controlled rotation for many pixels, and reconstruction needs many shots because
the position register must sample enough pixel addresses.

Files:

```text
frqi_tools.py
train_frqi_tiny_imagenet_64.py
frqi_generate.py
```

Smoke test:

```bash
python SlopGPT-Training/train_frqi_tiny_imagenet_64.py --epochs 1 --batch-size 4 --log-every 1 --max-steps 5
```

Longer FRQI-angle training:

```bash
python SlopGPT-Training/train_frqi_tiny_imagenet_64.py --epochs 3 --batch-size 32 --latent-dim 8
```

Resume:

```bash
python SlopGPT-Training/train_frqi_tiny_imagenet_64.py --epochs 3 --batch-size 32 --latent-dim 8 --resume
```

### Anti-Overfitting Training

The FRQI trainer now includes:

```text
data augmentation
weight decay
multiple latent vectors per image
stable image-derived latent codes
checkpoint latent-dim safety checks
```

If you want to move from `latent_dim=8` to `latent_dim=16`, start a new
checkpoint once:

```bash
python SlopGPT-Training/train_frqi_tiny_imagenet_64.py --epochs 5 --batch-size 32 --latent-dim 16 --reset
```

Use `--reset` once after trainer changes that affect latent behavior. Otherwise
the checkpoint may still behave like the old overfit/class-representative model.

Then continue training the same model:

```bash
python SlopGPT-Training/train_frqi_tiny_imagenet_64.py --epochs 5 --batch-size 32 --latent-dim 16 --resume
```

You do not need to run the command multiple times to train on multiple images.
Each epoch streams through the Tiny ImageNet training Parquet file. Running the
command again with `--resume` simply continues training for more epochs.

Useful knobs:

```text
--latents-per-image 2      train each image against multiple latent samples
--latent-jitter 0.05       small noise around each stable image latent
--weight-decay 0.0005      discourages memorizing noise
--augment / --no-augment   enables/disables flips, brightness, contrast, noise
```

Outputs:

```text
SlopGPT-Training/outputs/frqi_64x64_rgb_latent_vqg.npz
SlopGPT-Training/outputs/frqi_64x64_rgb_latent_vqg_metadata.json
SlopGPT-Training/outputs/frqi_64x64_rgb_latent_vqg_loss.png
```

Important: training optimizes FRQI angle fields directly using
`intensity = sin(theta)^2`. Building a full FRQI circuit for every training
batch would be too slow. The actual FRQI circuit is used for selected
generation/demonstration samples.

### Getting A Minimum Working FRQI Image

If the FRQI output looks like random static, the checkpoint is probably from a
short smoke test. Reinitialize it from real Tiny ImageNet representatives:

```bash
python SlopGPT-Training/train_frqi_tiny_imagenet_64.py --epochs 0 --batch-size 16 --latent-dim 8
```

Then train more:

```bash
python SlopGPT-Training/train_frqi_tiny_imagenet_64.py --epochs 3 --batch-size 32 --latent-dim 8 --resume
```

For the UI image path, shots are not the main issue: the UI decodes the trained
FRQI angle fields directly for speed. Shots matter when running an actual FRQI
circuit measurement demo. A useful rule of thumb:

```text
64x64 channel = 4096 pixel addresses
minimum demo:  16 samples/pixel -> 65,536 shots per channel
better demo:   64 samples/pixel -> 262,144 shots per channel
RGB demo: multiply by 3 channels
```

So do not try to fix random-looking UI images by increasing shots. Fix the
checkpoint first.

## Use FRQI In The Chat UI

Run the training backend:

```bash
python SlopGPT-Training/SlopGPT.py
```

This backend now tries the FRQI 64x64 RGB checkpoint first:

```text
prompt -> Tiny ImageNet class -> FRQI 64x64 RGB checkpoint -> PNG
```

Check status:

```text
GET http://localhost:8080/ping
```

The JSON should include:

```text
frqi_64_imported: true
frqi_64_available: true
```

As you train more with:

```bash
python SlopGPT-Training/train_frqi_tiny_imagenet_64.py --epochs 3 --batch-size 32 --latent-dim 8 --resume
```

the checkpoint at:

```text
SlopGPT-Training/outputs/frqi_64x64_rgb_latent_vqg.npz
```

is updated. New UI generations will use that updated checkpoint.
