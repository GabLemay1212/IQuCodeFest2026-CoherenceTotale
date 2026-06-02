# Quantum Patchwork v5

A quantum-classical hybrid system that generates SVG canvases containing multiple geometric shapes. You give it a text prompt like `2 purple heart and 2 red triangle`, and it produces a scalable vector image where every shape's position, size, color shade, and angle was determined by a variational quantum circuit.

## Why quantum computing?

The core challenge in generative art is **diversity**: the same prompt should produce a visually distinct output on every run. Classical approaches inject random noise into a latent space. Here we use a variational quantum circuit as the latent space sampler instead.

Fresh Gaussian noise (`ε ~ N(0, σ²π²)`) is added to the circuit's input embedding before each shape is generated. The quantum measurement outcomes — 10 `⟨Z⟩` expectation values — vary with this noise and act as the latent vector `z` that controls all downstream geometry: position, scale, rotation, and color shade. The circuit's entangled weights encode a learned style that biases outputs toward coherent compositions, while the noise injection ensures no two runs produce the same layout.

The key advantage over a classical MLP latent space is entanglement: the 10 output measurements are correlated through a high-dimensional entangled state in a way that would require a significantly larger classical network to replicate. The `StronglyEntanglingLayers` structure creates long-range correlations across all 10 qubits, so changing the noise on one wire affects the full output vector in a non-trivial way. This is what gives each canvas its distinctive character beyond simple random placement.

## Architecture

```
Text prompt
  → parse into (count, shape, color) tuples
  → per (shape, color): sine-encoded class embedding + fresh ε noise
      ↓
  Quantum encoder  (10 qubits, 5 StronglyEntanglingLayers, adjoint diff)
      ↓  z = tanh(⟨Z⟩₀…⟨Z⟩₉)
  Shape decoder MLP  (Fourier features + z → 11 raw params per instance)
      ↓  constrain_params
  Size spread + iterative overlap repulsion
      ↓
  SVG canvas  (infinite resolution, one file per run)
```

## Install

```
pip install torch pennylane
```

Optionally install `pennylane-lightning` for GPU-accelerated simulation:

```
pip install pennylane-lightning[gpu]
```

The system automatically falls back to `default.qubit` (CPU state-vector simulator) if `lightning.gpu` is unavailable.

## Usage

```
python quantum_patchwork_sds.py
```

Enter a prompt when asked:

```
2 purple heart and 2 red triangle
3 blue circle, green hexagon, 2 pink star
red circle
1 gold star and 3 cyan circle
```

Each run produces:
- One SVG file: `qshape_{prompt}_{run_id}_v01.svg`
- One weights file per unique (shape, color) pair: `qshape_{shape}_{run_id}_weights.pth`

---

## How it works

### Prompt parsing

The prompt is split on `and` and `,`. Each segment can optionally start with a count (digit or English word like `two`), followed by a color and a shape name in any order. Aliases like `ball → circle` or `rect → square` are resolved. When no color is specified, a default color for that shape is used. The result is a list of `(count, shape_name, rgb_tuple)`.

Supported shapes: `circle`, `star`, `hexagon`, `triangle`, `square`, `diamond`, `cross`, `pentagon`, `octagon`, `heart`

Supported colors: `red`, `orange`, `yellow`, `green`, `cyan`, `blue`, `purple`, `pink`, `white`, `black`, `gray`, `brown`, `gold`, `silver`, `teal`, `lime`, `maroon`, `navy`, `coral`, `violet`, `magenta`, `indigo`, `crimson`, `sky`

### Class embedding

Each unique `(shape, color)` pair gets a 10-dimensional input vector. Shape identity is sine-encoded at frequencies `1…10` applied to the shape's normalized index. Color is cosine-encoded from its luminance value. This is analogous to positional encoding in transformers — it gives the quantum circuit a compact, continuous input that separates different shape/color combinations in embedding space.

### Quantum encoder

| Property | Value |
|----------|-------|
| Backend | PennyLane `lightning.gpu` (falls back to `default.qubit`) |
| Qubits | 10 |
| State-vector size | $2^{10} = 1{,}024$ complex amplitudes |
| Layers | 5 `StronglyEntanglingLayers` |
| Trainable parameters | $5 \times 10 \times 3 = 150$ rotation angles |

One `QuantumVectorModel` is trained per unique `(shape, color)` pair in the prompt. During each forward pass:

1. **Noise injection** — `ε ~ N(0, σ²π²)` is added to the class embedding. `σ = 0.05` during training, `0.80` at sampling time. The large sampling noise is the sole source of layout diversity across runs.
2. **`AngleEmbedding`** — the noisy 10-dim vector is encoded as Y-rotations on all 10 wires.
3. **`StronglyEntanglingLayers`** — 5 blocks of parameterized CNOT ring + Rz rotations create entanglement across all wires.
4. **Measurement** — `⟨Z⟩` is measured on all 10 wires. The result is a 10-dim vector in `[-1, 1]`, which becomes the latent `z` after a `tanh`.

The circuit is wrapped in `qml.qnn.TorchLayer`, making it a standard `nn.Module` with trainable parameter tensor `weights` of shape `(5, 10, 3)`.

### Why adjoint differentiation

Training requires gradients through the quantum circuit. The two common methods are:

- **Parameter-shift rule** — evaluates the circuit at `θ + π/2` and `θ - π/2` for each parameter. With 150 parameters, that's 300 circuit evaluations per training step.
- **Adjoint differentiation** — propagates gradients backward through the unitary evolution with a single reverse sweep alongside the forward state vector. Cost is $O(p)$ state-vector operations rather than $O(2p)$ circuit evaluations.

We use `diff_method="adjoint"`, which reduces circuit evaluations to a single forward pass plus one reverse sweep — roughly a **150× reduction** in cost compared to parameter-shift. This makes training 150-parameter quantum circuits on CPU practical without GPU quantum simulation.

The adjoint gradient for a parameter $\theta$ is:

$$\frac{\partial L}{\partial \theta} = 2 \cdot \text{Re}\left\langle \lambda \,\middle|\, \frac{\partial U(\theta)}{\partial \theta} \,\middle|\, \psi \right\rangle$$

where $|\psi\rangle$ is the forward state and $|\lambda\rangle$ is the adjoint state propagated backward through the circuit.

### Shape decoder

A 4-layer MLP (`Linear → SiLU`, 128 hidden units) maps the concatenation of Fourier-encoded shape position and quantum latent `z` to 11 unconstrained parameters per shape instance. Fourier encoding uses `sin` and `cos` at frequencies `1…8 × π`, giving 16 dims that help the network distinguish shapes within the same canvas.

### Parameter constraining

The 11 raw outputs are mapped to physically meaningful values:

| Parameter | Range | Quantum control |
|-----------|-------|-----------------|
| `cx`, `cy` (center) | `[scale+pad, 1-scale-pad]` | `z[0]`, `z[1]` shift by ±0.18 |
| `scale` (radius) | `[0.04, 0.20]` | `z[2]` shifts by ±0.04 |
| `angle` | `[-π, π]` | MLP only |
| `r`, `g`, `b` | `[0, 1]` | `z[3]` brightness, `z[4]` warm/cool, `z[5]` saturation |
| `alpha` | `[0.65, 1.00]` | MLP only |
| `stroke width` | `[0.002, 0.008]` | MLP only |

Qubits 0–5 directly influence position, size, and color shade. This is intentional: quantum measurements have a direct, interpretable effect on the visual output.

### Training

The rasterizer (`ShapeRasterizer`) renders shapes as Gaussian blobs for training, enabling full gradient flow from pixel values back through the decoder and into the quantum circuit via adjoint differentiation. SVG export uses exact geometric primitives and only happens at inference time.

Loss function:

$$L = L_\text{color} + 0.5 \cdot L_\text{coverage} + 0.3 \cdot L_\text{diversity}$$

- **Color loss** — MSE between non-white pixels on the rasterized canvas and the target RGB.
- **Coverage loss** — squared difference between actual covered area fraction and the shape's target coverage (`SHAPE_PRIORS`).
- **Diversity loss** — penalty when position variance across instances falls below 0.05 (forces spread).

Optimizer: Adam with `lr=0.02` for the quantum encoder and `lr=5e-3` for the MLP. Learning rate is cosine-annealed over 400 steps with `eta_min=1e-5`.

### Layout post-processing

After quantum sampling, two classical passes clean up the composition:

1. **Size spread** — same-shape instances get sizes distributed across `[0.5×, 1.5×]` of their quantum-sampled scale, randomly shuffled.
2. **Iterative repulsion** — up to 400 passes push overlapping pairs apart along the line connecting their centers until `dist ≥ (r_i + r_j) × 2.2 + 0.04`. Stops early when no overlaps remain.

This is a classical heuristic layer handling edge cases that the quantum circuit isn't explicitly trained to avoid.

### SVG export

Each shape is rendered as a native SVG primitive (`<circle>`, `<polygon>`, `<rect>`, `<path>`). The stroke is a darkened version of the fill. Output is pretty-printed XML at 512×512 logical units, which scales to any resolution in a browser or vector editor.

## Quantum latent vector

After each shape instance is generated, the system prints the 10 `⟨Z⟩` measurements and their Shannon entropy:

$$H = -\sum_i \frac{|z_i|}{\sum_j |z_j|} \log \frac{|z_i|}{\sum_j |z_j|}$$

Higher entropy means the measurements are more spread across `[-1, 1]`, which correlates with more spatially diverse layouts. This is diagnostic information — it lets you see what the quantum circuit is "doing" for each shape instance.
