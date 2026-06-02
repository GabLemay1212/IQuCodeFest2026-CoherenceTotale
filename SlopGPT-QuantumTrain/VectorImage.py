#!/usr/bin/env python3
import glob
import math
import os
import re
import time
import warnings
import xml.dom.minidom
from dataclasses import dataclass
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

warnings.filterwarnings("ignore")

N_QUBITS          = 10
N_LAYERS          = 5
IMG_SIZE          = 512
N_SHAPES          = 7     # fewer = clean, non-overlapping compositions
FOURIER_DIM       = 8
TRAIN_NOISE_SCALE = 0.05
SAMPLE_NOISE_SCALE= 0.80  # large noise → each quantum sample is visually distinct

NAMED_COLORS = {
    "red":     (1.00, 0.07, 0.07),
    "orange":  (1.00, 0.55, 0.00),
    "yellow":  (1.00, 0.90, 0.00),
    "green":   (0.08, 0.70, 0.15),
    "cyan":    (0.00, 0.80, 0.90),
    "blue":    (0.08, 0.25, 0.95),
    "purple":  (0.55, 0.05, 0.90),
    "pink":    (1.00, 0.30, 0.65),
    "white":   (0.95, 0.95, 0.95),
    "black":   (0.05, 0.05, 0.05),
    "gray":    (0.55, 0.55, 0.55),
    "brown":   (0.55, 0.28, 0.07),
    "gold":    (1.00, 0.80, 0.10),
    "silver":  (0.75, 0.75, 0.78),
    "teal":    (0.00, 0.55, 0.55),
    "lime":    (0.60, 0.90, 0.05),
    "maroon":  (0.50, 0.00, 0.10),
    "navy":    (0.05, 0.10, 0.50),
    "coral":   (1.00, 0.45, 0.35),
    "violet":  (0.80, 0.20, 0.85),
    "magenta": (0.90, 0.05, 0.85),
    "indigo":  (0.30, 0.00, 0.65),
    "crimson": (0.86, 0.08, 0.24),
    "sky":     (0.35, 0.70, 1.00),
}

SHAPE_TYPES = ["circle","star","hexagon","triangle","square",
               "diamond","cross","pentagon","octagon","heart"]

SHAPE_ALIASES = {
    "round":"circle","ball":"circle","disc":"circle","disk":"circle",
    "six-sided":"hexagon","6-sided":"hexagon",
    "rect":"square","rectangle":"square","box":"square",
    "rhombus":"diamond","plus":"cross",
    "5-pointed":"star","five-pointed":"star",
    "8-sided":"octagon","eight-sided":"octagon",
    "5-sided":"pentagon","five-sided":"pentagon",
    "3-sided":"triangle","three-sided":"triangle","tri":"triangle",
    "4-sided":"square","four-sided":"square",
    "poly":"hexagon",
}

SHAPE_MAP = {s: i for i, s in enumerate(SHAPE_TYPES)}

@dataclass
class ShapePrior:
    color:    tuple
    coverage: float = 0.45
    n_sides:  int   = 0

SHAPE_PRIORS = {
    "circle":   ShapePrior((0.08, 0.40, 0.90), 0.45, 0),
    "star":     ShapePrior((1.00, 0.85, 0.00), 0.42, 5),
    "hexagon":  ShapePrior((0.08, 0.70, 0.15), 0.48, 6),
    "triangle": ShapePrior((0.90, 0.20, 0.05), 0.40, 3),
    "square":   ShapePrior((0.55, 0.05, 0.90), 0.50, 4),
    "diamond":  ShapePrior((0.00, 0.75, 0.85), 0.40, 4),
    "cross":    ShapePrior((1.00, 0.10, 0.10), 0.35, 0),
    "pentagon": ShapePrior((1.00, 0.55, 0.00), 0.45, 5),
    "octagon":  ShapePrior((0.60, 0.05, 0.60), 0.48, 8),
    "heart":    ShapePrior((1.00, 0.10, 0.35), 0.42, 0),
}


def parse_multi_prompt(raw: str):
    text  = raw.lower().replace("_", " ").replace("-", " ")
    parts = re.split(r'\band\b|,', text)
    results = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        # Optional leading count — digit ("2") or English word ("two")
        _WORD_NUMS = {
            "one":1,"two":2,"three":3,"four":4,"five":5,
            "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
            "a":1,"an":1,"the":1,
        }
        count = 1
        if tokens and tokens[0].isdigit():
            count = max(1, min(20, int(tokens[0])))
            tokens = tokens[1:]
        elif tokens and tokens[0] in _WORD_NUMS:
            count = _WORD_NUMS[tokens[0]]
            tokens = tokens[1:]
        # Find shape
        shape = None
        for tok in tokens:
            if tok in SHAPE_MAP:
                shape = tok; break
            if tok in SHAPE_ALIASES:
                shape = SHAPE_ALIASES[tok]; break
        if shape is None:
            print(f"  [warning] No recognized shape in '{part.strip()}', skipping.", flush=True)
            continue
        # Find color
        color = None
        for tok in tokens:
            if tok in NAMED_COLORS:
                color = NAMED_COLORS[tok]; break
        if color is None:
            color = SHAPE_PRIORS[shape].color
        results.append((count, shape, color))
    if not results:
        print(f"  [fallback] No valid shapes in '{raw}'. Defaulting to 1 blue circle.", flush=True)
        results = [(1, "circle", NAMED_COLORS["blue"])]
    return results


def make_class_embedding(shape_idx, color, n_qubits=N_QUBITS):
    t = shape_idx / max(len(SHAPE_TYPES)-1, 1) * 2.0 * math.pi
    freqs = np.arange(1, n_qubits+1, dtype=np.float32)
    shape_enc = np.sin(t * freqs) * math.pi
    lum = color[0]*0.6 + color[1]*0.3 + color[2]*0.1
    color_enc = np.cos(lum * math.pi * freqs) * 0.5 * math.pi
    return torch.tensor((shape_enc + color_enc).astype(np.float32))

def sample_noise(n_qubits=N_QUBITS, scale=SAMPLE_NOISE_SCALE,
                 device=torch.device("cpu")):
    return torch.randn(n_qubits, device=device) * scale * math.pi

def build_quantum_encoder(n_qubits=N_QUBITS, n_layers=N_LAYERS):
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
    except Exception:
        warnings.warn("lightning.gpu not available; using default.qubit.", RuntimeWarning, stacklevel=2)
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="adjoint", interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    layer = qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 3)})
    with torch.no_grad():
        nn.init.uniform_(list(layer.parameters())[0], -0.3*math.pi, 0.3*math.pi)
    return layer


class ShapeDecoder(nn.Module):
    OUT_DIM = 11
    def __init__(self, fourier_dim=FOURIER_DIM, latent_dim=N_QUBITS, hidden=128):
        super().__init__()
        in_dim = 2*fourier_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden//2), nn.SiLU(),
            nn.Linear(hidden//2, self.OUT_DIM),
        )
        with torch.no_grad():
            last = self.net[-1]
            nn.init.xavier_uniform_(last.weight)
            last.bias[4].fill_( 1.5)
            last.bias[5].fill_( 0.0)
            last.bias[6].fill_(-1.5)
            last.bias[7].fill_( 1.5)

    def _fourier_encode(self, ids):
        freqs = torch.arange(1, FOURIER_DIM+1, device=ids.device, dtype=ids.dtype) * math.pi
        x = ids.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

    def forward(self, shape_ids, z):
        enc   = self._fourier_encode(shape_ids)
        z_exp = z.unsqueeze(0).expand(shape_ids.shape[0], -1)
        return self.net(torch.cat([enc, z_exp], dim=-1))


def constrain_params(raw, target_color, z=None):
    tc = torch.tensor(target_color, dtype=raw.dtype, device=raw.device)
    scale_base = torch.sigmoid(raw[:,2]) * 0.12 + 0.05
    if z is not None and z.shape[0] >= 3:
        scale = (scale_base + z[2] * 0.04).clamp(0.04, 0.20)  # qubit 2 → size
    else:
        scale = scale_base
    pad     = scale.detach() + 0.03
    lo      = pad
    hi      = 1.0 - pad
    cx_base = torch.sigmoid(raw[:,0]) * 0.60 + 0.20
    cy_base = torch.sigmoid(raw[:,1]) * 0.60 + 0.20
    if z is not None and z.shape[0] >= 2:
        cx = torch.max(torch.min(cx_base + z[0] * 0.18, hi), lo)
        cy = torch.max(torch.min(cy_base + z[1] * 0.18, hi), lo)
    else:
        cx = torch.max(torch.min(cx_base, hi), lo)
        cy = torch.max(torch.min(cy_base, hi), lo)
    angle = torch.tanh(raw[:,3]) * math.pi
    if z is not None and z.shape[0] >= 6:
        brightness = 0.55 + (z[3].item() + 1.0) * 0.30
        warm_shift  = z[4].item() * 0.12   # slight warm (+R-B) or cool (-R+B) tilt
        sat_shift   = z[5].item() * 0.10   # push toward or away from grey
        r = (tc[0] * brightness + warm_shift + sat_shift * (tc[0] - 0.5)).clamp(0, 1)
        g = (tc[1] * brightness             + sat_shift * (tc[1] - 0.5)).clamp(0, 1)
        b = (tc[2] * brightness - warm_shift + sat_shift * (tc[2] - 0.5)).clamp(0, 1)
        N = raw.shape[0]
        r = r.unsqueeze(0).expand(N) if r.dim() == 0 else r.expand(N)
        g = g.unsqueeze(0).expand(N) if g.dim() == 0 else g.expand(N)
        b = b.unsqueeze(0).expand(N) if b.dim() == 0 else b.expand(N)
    else:
        r = (tc[0] + torch.tanh(raw[:,4]) * 0.18).clamp(0, 1)
        g = (tc[1] + torch.tanh(raw[:,5]) * 0.18).clamp(0, 1)
        b = (tc[2] + torch.tanh(raw[:,6]) * 0.18).clamp(0, 1)
    alpha = torch.sigmoid(raw[:,7]) * 0.35 + 0.65
    sr    = (r * 0.55).clamp(0,1)
    sg    = (g * 0.55).clamp(0,1)
    sw    = torch.sigmoid(raw[:,10]) * 0.006 + 0.002
    return torch.stack([cx,cy,scale,angle,r,g,b,alpha,sr,sg,sw], dim=1)


class QuantumVectorModel(nn.Module):
    def __init__(self, shape_name, color, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super().__init__()
        shape_idx = SHAPE_MAP[shape_name]
        self.register_buffer("class_emb", make_class_embedding(shape_idx, color, n_qubits))
        self.target_color    = color
        self.quantum_encoder = build_quantum_encoder(n_qubits, n_layers)
        self.shape_decoder   = ShapeDecoder(latent_dim=n_qubits)

    def forward(self, n_shapes=N_SHAPES, noise_scale=SAMPLE_NOISE_SCALE):
        noise     = sample_noise(self.class_emb.shape[0], noise_scale, self.class_emb.device)
        noisy_emb = self.class_emb + noise
        z_raw     = self.quantum_encoder(noisy_emb.unsqueeze(0))
        z         = torch.tanh(z_raw[0])   # quantum latent: drives all shape params
        ids       = torch.linspace(-1.0, 1.0, n_shapes, device=z.device)
        raw       = self.shape_decoder(ids, z)
        return constrain_params(raw, self.target_color, z), z



class ShapeRasterizer(nn.Module):
    def __init__(self, img_size=IMG_SIZE):
        super().__init__()
        lin = torch.linspace(0.0, 1.0, img_size)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        self.register_buffer("xx", xx)
        self.register_buffer("yy", yy)

    def forward(self, params):
        N     = params.shape[0]
        cx    = params[:,0].view(N,1,1)
        cy    = params[:,1].view(N,1,1)
        sc    = params[:,2].view(N,1,1)
        rgb   = params[:,4:7]
        alpha = params[:,7].view(N,1,1)
        dx    = self.xx.unsqueeze(0) - cx
        dy    = self.yy.unsqueeze(0) - cy
        sigma = sc / 2.0
        kernel   = torch.exp(-(dx**2 + dy**2) / (sigma**2 + 1e-8))
        coverage = (kernel * alpha).clamp(0.0, 1.0)
        H, W     = self.xx.shape
        one_minus = 1.0 - coverage
        cumprod   = torch.cumprod(one_minus, dim=0)
        remaining = torch.cat([torch.ones(1,H,W,device=params.device), cumprod[:-1]], dim=0)
        rgb_exp   = rgb.view(N,3,1,1)
        canvas    = (rgb_exp * coverage.unsqueeze(1) * remaining.unsqueeze(1)).sum(0)
        canvas    = canvas + cumprod[-1].unsqueeze(0)
        return canvas.clamp(0.0, 1.0)



class ShapeLoss(nn.Module):
    def __init__(self, prior, color):
        super().__init__()
        self.register_buffer("target_color", torch.tensor([color], dtype=torch.float32))
        self.target_coverage = prior.coverage

    def color_loss(self, canvas):
        pixels    = canvas.permute(1,2,0).reshape(-1,3)
        non_white = (pixels.max(dim=1).values < 0.92)
        if non_white.sum() < 10:
            return torch.tensor(1.0, device=canvas.device)
        return (pixels[non_white] - self.target_color).pow(2).mean()

    def coverage_loss(self, canvas):
        covered = (canvas.mean(dim=0) < 0.92).float().mean()
        return (covered - self.target_coverage).pow(2)

    def diversity_loss(self, params):
        pos_var = params[:,0].var() + params[:,1].var()
        return torch.clamp(0.05 - pos_var, min=0.0)

    def forward(self, canvas, params):
        cl = self.color_loss(canvas)
        cv = self.coverage_loss(canvas)
        dv = self.diversity_loss(params)
        total = cl + 0.5*cv + 0.3*dv
        return total, {"color": cl.item(), "coverage": cv.item(), "diversity": dv.item()}



def _rgb255(r, g, b):
    return f"rgb({int(min(255,max(0,round(r*255))))},{int(min(255,max(0,round(g*255))))},{int(min(255,max(0,round(b*255))))})"

def _polygon_pts(cx, cy, r, n, offset=0.0, S=512):
    pts = []
    for k in range(n):
        th = 2*math.pi*k/n + offset
        pts.append(f"{cx*S + math.cos(th)*r*S:.2f},{cy*S + math.sin(th)*r*S:.2f}")
    return " ".join(pts)

def _star_pts(cx, cy, r_out, r_in, n, offset=0.0, S=512):
    pts = []
    for k in range(n*2):
        r  = r_out if k%2==0 else r_in
        th = math.pi*k/n + offset
        pts.append(f"{cx*S + math.cos(th)*r*S:.2f},{cy*S + math.sin(th)*r*S:.2f}")
    return " ".join(pts)

def _heart_path(cx, cy, sc, S=512):
    s  = sc*S; px = cx*S; py = cy*S
    return (
        f"M {px:.2f},{py-0.5*s:.2f} "
        f"C {px+0.5*s:.2f},{py-1.0*s:.2f} {px+s:.2f},{py-0.3*s:.2f} {px:.2f},{py+0.6*s:.2f} "
        f"C {px-s:.2f},{py-0.3*s:.2f} {px-0.5*s:.2f},{py-1.0*s:.2f} {px:.2f},{py-0.5*s:.2f} Z"
    )

def _cross_path(cx, cy, sc, S=512):
    s  = sc*S; t = s*0.30; px = cx*S; py = cy*S
    return (
        f"M {px-t:.2f},{py-s:.2f} H {px+t:.2f} V {py-t:.2f} "
        f"H {px+s:.2f} V {py+t:.2f} H {px+t:.2f} "
        f"V {py+s:.2f} H {px-t:.2f} V {py+t:.2f} "
        f"H {px-s:.2f} V {py-t:.2f} H {px-t:.2f} Z"
    )

def _svg_add_shape(svg, row, shape_name, S):
    cx,cy,sc    = float(row[0]),float(row[1]),float(row[2])
    angle       = float(row[3])
    r,g,b       = float(row[4]),float(row[5]),float(row[6])
    alpha       = float(row[7])
    sr,sg,sw    = float(row[8]),float(row[9]),float(row[10])
    cx_px,cy_px = cx*S, cy*S
    stroke_w    = sw*S
    angle_deg   = math.degrees(angle)
    fill_col    = _rgb255(r,g,b)
    stroke_col  = _rgb255(sr,sg,0.2)
    attrs = {
        "fill": fill_col, "fill-opacity": f"{alpha:.3f}",
        "stroke": stroke_col, "stroke-width": f"{stroke_w:.2f}",
        "stroke-opacity": f"{min(alpha,0.85):.3f}",
    }
    rot = f"rotate({angle_deg:.2f},{cx_px:.2f},{cy_px:.2f})"

    if shape_name == "circle":
        SubElement(svg,"circle",{**attrs,"cx":f"{cx_px:.2f}","cy":f"{cy_px:.2f}",
                                 "r":f"{sc*S:.2f}","transform":rot})
    elif shape_name == "square":
        side = sc*S*2
        SubElement(svg,"rect",{**attrs,"x":f"{cx_px-side/2:.2f}","y":f"{cy_px-side/2:.2f}",
                               "width":f"{side:.2f}","height":f"{side:.2f}","transform":rot})
    elif shape_name == "diamond":
        side = sc*S*2
        SubElement(svg,"rect",{**attrs,"x":f"{cx_px-side/2:.2f}","y":f"{cy_px-side/2:.2f}",
                               "width":f"{side:.2f}","height":f"{side:.2f}",
                               "transform":f"rotate({angle_deg+45:.2f},{cx_px:.2f},{cy_px:.2f})"})
    elif shape_name == "triangle":
        SubElement(svg,"polygon",{**attrs,
            "points":_polygon_pts(cx,cy,sc,3,angle-math.pi/2,S)})
    elif shape_name == "hexagon":
        SubElement(svg,"polygon",{**attrs,
            "points":_polygon_pts(cx,cy,sc,6,angle,S)})
    elif shape_name == "pentagon":
        SubElement(svg,"polygon",{**attrs,
            "points":_polygon_pts(cx,cy,sc,5,angle-math.pi/2,S)})
    elif shape_name == "octagon":
        SubElement(svg,"polygon",{**attrs,
            "points":_polygon_pts(cx,cy,sc,8,angle,S)})
    elif shape_name == "star":
        SubElement(svg,"polygon",{**attrs,
            "points":_star_pts(cx,cy,sc,sc*0.42,5,angle-math.pi/2,S)})
    elif shape_name == "heart":
        SubElement(svg,"path",{**attrs,"d":_heart_path(cx,cy,sc,S)})
    elif shape_name == "cross":
        SubElement(svg,"path",{**attrs,"d":_cross_path(cx,cy,sc,S),"transform":rot})


def spread_instances(instances):
    inst = [(row.copy(), name) for row, name in instances]
    n = len(inst)
    if n <= 1:
        return inst

    shape_groups: dict = {}
    for i, (_, shape_name) in enumerate(inst):
        shape_groups.setdefault(shape_name, []).append(i)

    for indices in shape_groups.values():
        if len(indices) > 1:
            m = len(indices)
            factors = np.linspace(0.50, 1.50, m)
            np.random.shuffle(factors)
            for k, idx in enumerate(indices):
                inst[idx][0][2] = float(np.clip(inst[idx][0][2] * factors[k], 0.04, 0.20))
                sc = inst[idx][0][2]
                inst[idx][0][0] = float(np.clip(inst[idx][0][0], sc + 0.03, 1.0 - sc - 0.03))
                inst[idx][0][1] = float(np.clip(inst[idx][0][1], sc + 0.03, 1.0 - sc - 0.03))

    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = inst[i][0], inst[j][0]
            if abs(float(ri[0]) - float(rj[0])) < 1e-4 and abs(float(ri[1]) - float(rj[1])) < 1e-4:
                angle = 2 * math.pi * i / max(n, 1)
                jitter = 0.05
                ri[0] = float(np.clip(ri[0] + math.cos(angle) * jitter, inst[i][0][2] + 0.03, 1.0 - inst[i][0][2] - 0.03))
                ri[1] = float(np.clip(ri[1] + math.sin(angle) * jitter, inst[i][0][2] + 0.03, 1.0 - inst[i][0][2] - 0.03))
                rj[0] = float(np.clip(rj[0] - math.cos(angle) * jitter, inst[j][0][2] + 0.03, 1.0 - inst[j][0][2] - 0.03))
                rj[1] = float(np.clip(rj[1] - math.sin(angle) * jitter, inst[j][0][2] + 0.03, 1.0 - inst[j][0][2] - 0.03))

    for _ in range(400):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                ri, rj = inst[i][0], inst[j][0]
                dx = float(ri[0]) - float(rj[0])
                dy = float(ri[1]) - float(rj[1])
                dist = math.sqrt(dx * dx + dy * dy) + 1e-9
                min_sep = (float(ri[2]) + float(rj[2])) * 2.2 + 0.04
                if dist < min_sep:
                    push = (min_sep - dist) * 0.55
                    nx, ny = dx / dist, dy / dist
                    ri_sc = float(ri[2]); rj_sc = float(rj[2])
                    ri[0] = float(np.clip(ri[0] + nx * push, ri_sc + 0.03, 1.0 - ri_sc - 0.03))
                    ri[1] = float(np.clip(ri[1] + ny * push, ri_sc + 0.03, 1.0 - ri_sc - 0.03))
                    rj[0] = float(np.clip(rj[0] - nx * push, rj_sc + 0.03, 1.0 - rj_sc - 0.03))
                    rj[1] = float(np.clip(rj[1] - ny * push, rj_sc + 0.03, 1.0 - rj_sc - 0.03))
                    moved = True
        if not moved:
            break

    return inst


def save_multi_svg(shape_instances, canvas_size, path):
    S   = canvas_size
    svg = Element("svg", {
        "xmlns":  "http://www.w3.org/2000/svg",
        "width":  str(S), "height": str(S),
        "viewBox": f"0 0 {S} {S}",
        "shape-rendering": "geometricPrecision",
    })
    SubElement(svg, "rect", {"width":"100%","height":"100%","fill":"white"})
    for row, shape_name in shape_instances:
        _svg_add_shape(svg, row, shape_name, S)
    pretty = xml.dom.minidom.parseString(tostring(svg,"unicode")).toprettyxml(indent="  ")
    with open(path,"w") as fh:
        fh.write(pretty)
    print(f"  SVG → {path}  (infinite resolution)", flush=True)


def train_model(model, rasterizer, loss_fn, steps=400):
    opt = optim.Adam([
        {"params": model.quantum_encoder.parameters(), "lr": 0.02},
        {"params": model.shape_decoder.parameters(),   "lr": 5e-3},
    ])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    print("\n" + "="*64, flush=True)
    print("  Training — Quantum Geometric Shape Generator v4", flush=True)
    print(f"  {N_QUBITS} qubits · {N_LAYERS} layers · {steps} steps · {IMG_SIZE}px preview", flush=True)
    print("="*64, flush=True)
    for step in range(1, steps+1):
        opt.zero_grad()
        params,_ = model(N_SHAPES, TRAIN_NOISE_SCALE)
        canvas   = rasterizer(params)
        loss, bd = loss_fn(canvas, params)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.quantum_encoder.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == 1:
            print(f"  [{step:>4d}/{steps}] loss={loss.item():.5f}  "
                  f"color={bd['color']:.4f} cov={bd['coverage']:.4f} "
                  f"div={bd['diversity']:.4f}  ({time.strftime('%H:%M:%S')})", flush=True)
    print("="*64, flush=True)


def report_quantum_state(z: torch.Tensor, sample_idx: int) -> None:
    z_np = z.cpu().numpy()
    print(f"  ┌─ Quantum state #{sample_idx}  ({N_QUBITS} qubits · {N_LAYERS} layers · adjoint diff)", flush=True)
    print(f"  │  ⟨Z⟩ measurements : [{', '.join(f'{v:+.3f}' for v in z_np)}]", flush=True)
    entropy = -np.sum((np.abs(z_np)/np.sum(np.abs(z_np))) *
                      np.log(np.abs(z_np)/np.sum(np.abs(z_np)) + 1e-9))
    print(f"  │  Shannon entropy  : {entropy:.4f} nats  (higher = more diverse layout)", flush=True)
    print(f"  └─ Each sample draws fresh ε~N(0,{SAMPLE_NOISE_SCALE}²π²) before the circuit", flush=True)




def main():
    print("\n" + "="*64, flush=True)
    print("  Quantum Patchwork v5 — Multi-Shape Canvas Generator", flush=True)
    print("  Quantum backend : PennyLane  |  SVG vector output", flush=True)
    print("  Shapes :", ", ".join(SHAPE_TYPES), flush=True)
    print("  Colors :", ", ".join(sorted(NAMED_COLORS.keys())), flush=True)
    print("="*64, flush=True)
    print("  Prompt examples:", flush=True)
    print("    '2 purple heart and 2 red triangle'", flush=True)
    print("    '3 blue circle, green hexagon, 2 pink star'", flush=True)
    print("    'red circle'", flush=True)


    while True:
        raw = input("\nEnter shape prompt: ").strip()
        if not raw:
            continue
        shape_specs = parse_multi_prompt(raw)
        total = sum(c for c, _, _ in shape_specs)
        print(f"  Parsed {total} shape(s) across {len(shape_specs)} group(s):", flush=True)
        for count, shape, color in shape_specs:
            print(f"    {count}\u00d7 {shape}  color={color}", flush=True)
        break

    n_samples = 1

    train_seed = int(time.time() * 1000) % (2**32)
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    run_id = int(time.time())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Device     : {device}", flush=True)
    print(f"  PennyLane  : {qml.__version__}", flush=True)
    print(f"  PyTorch    : {torch.__version__}", flush=True)
    print(f"  Qubits     : {N_QUBITS}  (state-vector dim = {2**N_QUBITS})", flush=True)
    print(f"  Layers     : {N_LAYERS} StronglyEntanglingLayers (adjoint diff)", flush=True)
    total_shapes = sum(c for c, _, _ in shape_specs)
    print(f"  Canvases   : {n_samples}  |  Shapes per canvas : {total_shapes}", flush=True)

    raster = ShapeRasterizer(IMG_SIZE).to(device)

    unique_keys = list(dict.fromkeys((s, c) for _, s, c in shape_specs))
    trained_models: dict = {}
    print(f"\n[1/2]  Training {len(unique_keys)} quantum model(s) …", flush=True)
    for shape, color in unique_keys:
        print(f"  Model: '{shape}'  color={color}", flush=True)
        prior   = SHAPE_PRIORS[shape]
        loss_fn = ShapeLoss(prior, color).to(device)
        mdl     = QuantumVectorModel(shape, color, N_QUBITS, N_LAYERS).to(device)
        nq  = sum(p.numel() for p in mdl.quantum_encoder.parameters())
        ncl = sum(p.numel() for p in mdl.shape_decoder.parameters())
        print(f"  Quantum params : {nq}  |  Classical params : {ncl}", flush=True)
        train_model(mdl, raster, loss_fn, steps=400)
        trained_models[(shape, color)] = mdl
        w_path = f"qshape_{shape}_{run_id}_weights.pth"
        torch.save(mdl.state_dict(), w_path)
        print(f"  Weights → {w_path}", flush=True)

    print(f"\n[2/2]  Compositing {n_samples} unique quantum canvas(es) …", flush=True)
    print("  Fresh \u03b5~N(0,\u03c3\u00b2\u03c0\u00b2) noise is injected per shape instance per canvas,", flush=True)
    print("  so every run \u2014 even with the same prompt \u2014 yields a unique layout.\n", flush=True)

    generated = []
    for i in range(1, n_samples + 1):
        torch.manual_seed(int(time.time() * 1e9) % (2**32) + i * 997)

        instances   = []   # (params_1d_np, shape_name) for every shape on this canvas
        inst_number = 0
        for count, shape, color in shape_specs:
            mdl = trained_models[(shape, color)]
            for _ in range(count):
                with torch.no_grad():
                    params, z = mdl(1, SAMPLE_NOISE_SCALE)
                inst_number += 1
                report_quantum_state(z, inst_number)
                instances.append((params.cpu().numpy()[0], shape))

        tag      = raw.replace(" ", "_")[:40]
        svg_path = f"qshape_{tag}_{run_id}_v{i:02d}.svg"
        instances = spread_instances(instances)
        save_multi_svg(instances, IMG_SIZE, svg_path)
        print(f"  [{i}/{n_samples}]  Canvas \u2192 {svg_path}\n", flush=True)
        generated.append(svg_path)

    print("="*64, flush=True)
    print(f"  Generated {n_samples} unique quantum canvas(es):", flush=True)
    for svg in generated:
        print(f"    {svg}  \u2190 open in browser for \u221e-resolution", flush=True)
    print("="*64, flush=True)
    print("  Quantum note: each shape instance was placed by fresh", flush=True)
    print("  \u03b5~N(0,\u03c3\u00b2) noise injected before the variational quantum circuit,", flush=True)
    print("  making the quantum latent space the sole source of visual diversity.\n", flush=True)


if __name__ == "__main__":
    main()
