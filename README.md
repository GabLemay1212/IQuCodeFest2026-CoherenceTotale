# SlopGPT Quantum Image Demo

SlopGPT is a small local demo for quantum-inspired image generation and image
up/down scaling.

It has two image-generation models:

- Model 1: Fashion-MNIST pixel generator trained from `Datasets/Fashion-MNIST`.
- Model 2: vector shape generator from `SlopGPT-QuantumTrain/VectorImage.py`.

It also has a separate image transform server for quantum-assisted upscale and
downscale in `up-down-scaling/quantum_image.py`.

## Install

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset Layout

Fashion-MNIST should be here:

```text
Datasets/Fashion-MNIST/train-images-idx3-ubyte.gz
Datasets/Fashion-MNIST/train-labels-idx1-ubyte.gz
```

The trainer uses that path by default, so you normally do not need to pass
`--data-dir`.

## Train Model 1

Quick smoke test:

```powershell
python SlopGPT-QuantumTrain/train_quantum_sim_fashion_mnist_16.py --epochs 1 --samples-per-class 1 --shots 64 --max-steps 10 --optimizer spsa --init class-average --reset
```

Better small demo:

```powershell
python SlopGPT-QuantumTrain/train_quantum_sim_fashion_mnist_16.py --epochs 2 --samples-per-class 2 --shots 128 --max-steps 40 --optimizer spsa --init class-average --reset
```

Larger demo:

```powershell
python SlopGPT-QuantumTrain/train_quantum_sim_fashion_mnist_16.py --epochs 5 --samples-per-class 20 --shots 256 --optimizer spsa --init class-average --latent-dim 4 --latent-scale 1.0 --basis-init-scale 0.04 --basis-lr-scale 0.35 --basis-clip 0.45 --augment-shift 1 --augment-noise 0.02 --learning-rate 0.25 --spsa-c 0.08 --log-every 20 --max-steps 0 --reset
```

Resume training:

```powershell
python SlopGPT-QuantumTrain/train_quantum_sim_fashion_mnist_16.py --epochs 5 --samples-per-class 20 --shots 256 --optimizer spsa --latent-dim 4 --latent-scale 1.0 --basis-lr-scale 0.35 --basis-clip 0.45 --augment-shift 1 --augment-noise 0.02 --learning-rate 0.25 --spsa-c 0.08 --log-every 20 --max-steps 0 --resume
```

Training writes the checkpoint here:

```text
SlopGPT-QuantumTrain/outputs/quantum_sim_fashion_mnist_16x16_angles.npz
```

## Generate From The Command Line

```powershell
python SlopGPT-QuantumTrain/generate_quantum_sim_sample.py sneaker --shots 256 --candidates 8 --latent-scale 1.2
python SlopGPT-QuantumTrain/generate_quantum_sim_sample.py "cool futuristic sneakers" --shots 256 --variation 0.08
python SlopGPT-QuantumTrain/generate_quantum_sim_sample.py "small boot" --shots 128 --seed 1
```

Outputs are saved in:

```text
SlopGPT-QuantumTrain/outputs/
```

## Run The UI

Start the SlopGPT generation server:

```powershell
python SlopGPT-QuantumTrain/SlopGPT.py
```

Then open:

```text
slopgpt.html
```

The UI calls `http://localhost:8080`.

For upscale/downscale buttons, also start the transform server:

```powershell
python up-down-scaling/quantum_image.py --port 8081
```

The UI calls `http://localhost:8081` for `/upscale` and `/downscale`.

## Notebook Version

There is also a small version that works directly inside the IPYNB file:

```text
SlopGPT-QuantumTrain/notebook-SlopGPT.ipynb
```

Open it in Jupyter/JupyterLab, choose a prompt in the prompt cell, then run the
generation cell. It uses the already-trained checkpoint in `outputs`; it does
not train a new model.

## Words That Work

Model 1 needs one Fashion-MNIST class word in the prompt.

Accepted class words:

```text
t-shirt, tshirt, tee, top
trouser, trousers, pants
pullover
dress
coat, jacket
sandal
shirt
sneaker, shoe
bag
ankle boot, boot
```

Optional style words for Model 1:

```text
cool, futuristic, future, cyber, cyberpunk, tech, sci-fi, scifi
tall, high, long
small, tiny, mini, little, compact
short, low, stubby
chunky, wide, thick, bulky, large
slim, thin, narrow, sleek
heavy, combat, rugged, sturdy
simple, plain, minimal, clean
sharp, pointy, angular, edgy
```

Model 1 examples:

```text
cool futuristic sneakers
small boot
tall boots
simple bag
sharp dress
chunky sneaker
```

Model 2 uses vector shape words, optional counts, and color words.

Accepted shape words:

```text
circle, star, hexagon, triangle, square, diamond, cross, pentagon, octagon, heart
```

Accepted shape aliases:

```text
round, ball, disc, disk, six-sided, 6-sided, rect, rectangle, box,
rhombus, plus, 5-pointed, five-pointed, 8-sided, eight-sided,
5-sided, five-sided, 3-sided, three-sided, tri, 4-sided, four-sided, poly
```

Accepted color words:

```text
red, orange, yellow, green, cyan, blue, purple, pink, white, black,
gray, brown, gold, silver, teal, lime, maroon, navy, coral, violet,
magenta, indigo, crimson, sky
```

Counts can be digits or words:

```text
1, 2, 3, ..., 20
one, two, three, four, five, six, seven, eight, nine, ten
```

Model 2 examples:

```text
red circle
4 red circles
blue star and 2 yellow triangles
purple diamond, green hexagon
```

## IBM Quantum

IBM credentials are not stored in the code. If you use IBM modes, set them in
your environment before starting the server:

```powershell
$env:IBM_QUANTUM_TOKEN="your_token"
$env:IBM_QUANTUM_INSTANCE="your_instance_crn"
$env:IBM_QUANTUM_BACKEND="optional_backend_name"
```

If `IBM_QUANTUM_BACKEND` is omitted, the code asks IBM Runtime for a least-busy
hardware backend with enough qubits.
