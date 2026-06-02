# SlopGPT QuantumTrain

Tiny proof-of-concept training loop that uses the Qiskit Aer simulator during
training.

This is separate from `SlopGPT-Training` on purpose. The main trainer is faster
and better for the UI. This folder is for demonstrating:

```text
Fashion-MNIST image
  -> class base angles + latent basis patterns
  -> trainable quantum RY angles
  -> simulator-measured probabilities
  -> SPSA or parameter-shift gradient update
```

## Smoke Test

```powershell
python SlopGPT-QuantumTrain/train_quantum_sim_fashion_mnist_16.py --epochs 1 --samples-per-class 1 --shots 64 --max-steps 10 --optimizer spsa --init class-average --data-dir Datasets/Fashion-MNIST --reset
```

This trains on only 10 images total, one per class.

## Slightly Longer Demo

```powershell
python SlopGPT-QuantumTrain/train_quantum_sim_fashion_mnist_16.py --epochs 2 --samples-per-class 2 --shots 128 --max-steps 40 --optimizer spsa --init class-average --data-dir Datasets/Fashion-MNIST --reset
```

## Bigger Quantum-Training Demo

This is the command for 5 epochs, 256 shots, and 20 images per class:

```powershell
python SlopGPT-QuantumTrain/train_quantum_sim_fashion_mnist_16.py --epochs 5 --samples-per-class 20 --shots 256 --optimizer spsa --init class-average --latent-dim 4 --latent-scale 1.0 --basis-init-scale 0.04 --basis-lr-scale 0.35 --basis-clip 0.45 --augment-shift 1 --augment-noise 0.02 --learning-rate 0.25 --spsa-c 0.08 --log-every 20 --max-steps 0 --data-dir Datasets/Fashion-MNIST --reset
```

## Resume

```powershell
python SlopGPT-QuantumTrain/train_quantum_sim_fashion_mnist_16.py --epochs 5 --samples-per-class 20 --shots 256 --optimizer spsa --latent-dim 4 --latent-scale 1.0 --basis-lr-scale 0.35 --basis-clip 0.45 --augment-shift 1 --augment-noise 0.02 --learning-rate 0.25 --spsa-c 0.08 --log-every 20 --max-steps 0 --resume --data-dir Datasets/Fashion-MNIST
```

## Generate A Sample

```powershell
python SlopGPT-QuantumTrain/generate_quantum_sim_sample.py sneaker --shots 256 --candidates 8 --latent-scale 1.2
python SlopGPT-QuantumTrain/generate_quantum_sim_sample.py bag --shots 256 --candidates 8 --latent-scale 1.2
python SlopGPT-QuantumTrain/generate_quantum_sim_sample.py "ankle boot" --shots 256 --candidates 8 --latent-scale 1.2
```

## Prompt Attribute Conditioning

This is keyword-based, not true natural-language understanding. The system maps
descriptive words to deterministic transformations of the quantum angle matrix
before circuit execution. The Fashion-MNIST class still comes from keywords like
`sneaker`, `boot`, `bag`, or `trouser`.

Examples:

```powershell
python SlopGPT-QuantumTrain/generate_quantum_sim_sample.py "cool futuristic sneakers" --shots 256 --variation 0.08
python SlopGPT-QuantumTrain/generate_quantum_sim_sample.py "tall boots" --shots 256 --variation 0.05
python SlopGPT-QuantumTrain/generate_quantum_sim_sample.py "small boot" --shots 128 --seed 1
python SlopGPT-QuantumTrain/generate_quantum_sim_sample.py "futuristic boots" --shots 128 --seed 1 --variation 0.08
python SlopGPT-QuantumTrain/generate_quantum_sim_sample.py "chunky sneaker" --shots 128
python SlopGPT-QuantumTrain/generate_quantum_sim_sample.py "simple bag" --shots 256 --variation 0.02
```

Supported attribute words include:

```text
cool, futuristic, tall, small, tiny, short, chunky, slim, heavy, simple, sharp
```

## Launch The UI Backend

After training a checkpoint, run:

```powershell
python SlopGPT-QuantumTrain/SlopGPT.py
```

Then open `slopgpt.html` like before. This backend uses only the simulator-trained
checkpoint from `SlopGPT-QuantumTrain/outputs/`.

The IBM modes submit one 16-row image job to IBM Quantum. If the IBM job stays
pending/running longer than the configured timeout, the server returns a JSON
`202 Accepted` response with the IBM job id/status instead of pretending a
simulator fallback image came from hardware. Completed hardware jobs return PNGs
with these headers:

```text
X-Quantum-Backend
X-Ibm-Backend
X-Ibm-Job-Id
X-Ibm-Status
```

This fixes the confusing case where a queued IBM job takes longer than the HTTP
request timeout and the UI would otherwise show a simulator image too early.

IBM UI modes:

```text
IBM Fast         256 shots
IBM Balanced     512 shots
IBM DeepThinking 1024 shots
```

The IBM modes use one candidate only, because running candidate search on real
hardware would submit multiple jobs and make pending time much worse. The local
Fast/Balanced/DeepThinking modes still use the Aer simulator.

Command-line IBM demo:

```powershell
python SlopGPT-QuantumTrain/generate_quantum_sim_sample.py sneaker --shots 512 --backend ibm --ibm-timeout-seconds 45
```

Outputs are saved in:

```text
SlopGPT-QuantumTrain/outputs/
```

## What Is Quantum Here?

Training uses simulator-measured quantum circuits for the forward pass. Each row
of a 16x16 image is a 16-qubit circuit. Pixel brightness is estimated from
measurement counts. The model learns a class base image plus latent basis
patterns, so the same prompt can produce different samples.

The trainer supports two optimizers:

```text
--optimizer parameter-shift
--optimizer spsa
```

`parameter-shift` is easiest to explain mathematically:

```text
d p(phi) / d phi = 0.5 * [p(phi + pi/2) - p(phi - pi/2)]
```

`spsa` is more realistic for quantum hardware because it estimates an update
with two perturbed circuit evaluations, instead of needing one gradient per
parameter.

The trainer also supports:

```text
--init class-average
--init random
```

`class-average` initializes each class from the average 16x16 Fashion-MNIST image
for that class, then quantum simulator training fine-tunes the angles.

Anti-overfit knobs:

```text
--latent-dim          number of learned variation patterns per class
--latent-scale        strength of random latent variation
--basis-lr-scale      learning speed for the latent basis
--augment-shift       random pixel shift during training
--augment-noise       small grayscale noise during training
```

This is intentionally tiny because full quantum training would require many
circuit evaluations per image, parameter, and epoch.
