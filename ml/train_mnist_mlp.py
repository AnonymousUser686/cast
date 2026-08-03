"""Train a small MNIST MLP and export integer weights for the CaST systolic array.

The network is deliberately shaped so every dimension is a multiple of 5, which
is the tile size of the 5x5 systolic array:

    input 10x10 = 100  ->  hidden 20  ->  output 10

Everything is quantised to integers that the CaST hardware can actually hold:

    pixels    uint8   0..255
    weights   int8    -127..127   (symmetric, per tensor)
    bias      int32   in accumulator units
    accum     int32

The integer forward pass implemented in ``int_forward`` is the *specification*
the CaST program must reproduce bit for bit. It uses only operations the
compiler supports on unsigned wires:

    * multiply / add          - two's complement, correct for negatives
    * ReLU via a sign-bit test - ``acc >= 2**31`` detects a negative int32
    * right shift              - only applied AFTER ReLU, so the value is
                                 non-negative and a logical shift is correct

Run:  python ml/train_mnist_mlp.py
Out:  ml/mnist_model.json
"""

import json
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "mnist_model.json"

GRID = 10          # downsample 28x28 -> GRID x GRID
IN_DIM = GRID * GRID
HIDDEN = 20
CLASSES = 10
TILE = 5           # systolic array size
EPOCHS = 6
BATCH = 128
SEED = 0


# ---------------------------------------------------------------- data


def downsample(x):
    """28x28 float in [0,1] -> GRID*GRID float in [0,1]."""
    return F.adaptive_avg_pool2d(x, (GRID, GRID)).reshape(x.shape[0], -1)


def load_split(train):
    ds = datasets.MNIST(DATA, train=train, download=True,
                        transform=transforms.ToTensor())
    x = ds.data.float().unsqueeze(1) / 255.0
    y = ds.targets.clone()
    return downsample(x), y


# ---------------------------------------------------------------- model


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, CLASSES)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def train(xtr, ytr, xte, yte):
    torch.manual_seed(SEED)
    net = MLP()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    n = xtr.shape[0]
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(net(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = (net(xte).argmax(1) == yte).float().mean().item()
        print(f"  epoch {epoch + 1}/{EPOCHS}  test acc {acc * 100:.2f}%")
    return net


# ---------------------------------------------------------- quantisation


def sym_quantize(w):
    """Symmetric int8 quantisation of a weight tensor. Returns (int8, scale)."""
    scale = float(np.abs(w).max()) / 127.0
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int64)
    return q, scale


def int_forward(x_u8, p):
    """Exact integer inference. x_u8: (B, IN_DIM) ints 0..255.

    This is what the CaST program must reproduce. Every intermediate is an
    int32; nothing here relies on signed comparison or signed shifting.
    """
    w1, b1 = np.array(p["w1"]), np.array(p["b1"])
    w2, b2 = np.array(p["w2"]), np.array(p["b2"])
    shift = p["hidden_shift"]

    acc1 = x_u8 @ w1.T + b1                 # int32, may be negative
    acc1 = np.maximum(acc1, 0)              # ReLU  (sign-bit test in CaST)
    h = np.minimum(acc1 >> shift, 255)      # requantise, safe: acc1 >= 0
    acc2 = h @ w2.T + b2                    # int32 logits
    return acc1, h, acc2


def pick_shift(acc1):
    """Smallest shift that keeps every post-ReLU activation inside uint8."""
    peak = int(np.maximum(acc1, 0).max())
    s = 0
    while (peak >> s) > 255:
        s += 1
    return s


# ---------------------------------------------------------------- main


def main():
    os.makedirs(DATA, exist_ok=True)
    print("loading MNIST ...")
    xtr, ytr = load_split(True)
    xte, yte = load_split(False)
    print(f"  train {tuple(xtr.shape)}  test {tuple(xte.shape)}")

    print("training ...")
    net = train(xtr, ytr, xte, yte)

    with torch.no_grad():
        float_acc = (net(xte).argmax(1) == yte).float().mean().item()
        W1 = net.fc1.weight.numpy().astype(np.float64)
        B1 = net.fc1.bias.numpy().astype(np.float64)
        W2 = net.fc2.weight.numpy().astype(np.float64)
        B2 = net.fc2.bias.numpy().astype(np.float64)

    print("quantising ...")
    # Pixels become plain 0..255 integers, so the activation scale is 1/255.
    s_x = 1.0 / 255.0
    w1q, s_w1 = sym_quantize(W1)
    b1q = np.rint(B1 / (s_x * s_w1)).astype(np.int64)

    xte_u8 = np.rint(xte.numpy() * 255.0).astype(np.int64)
    xtr_u8 = np.rint(xtr.numpy() * 255.0).astype(np.int64)

    # Choose the requantisation shift from the training set activations.
    acc1_tr = xtr_u8 @ w1q.T + b1q
    shift = pick_shift(acc1_tr)

    # Hidden activation scale follows from the shift.
    s_h = s_x * s_w1 * (2 ** shift)
    w2q, s_w2 = sym_quantize(W2)
    b2q = np.rint(B2 / (s_h * s_w2)).astype(np.int64)

    params = {
        "w1": w1q.tolist(), "b1": b1q.tolist(),
        "w2": w2q.tolist(), "b2": b2q.tolist(),
        "hidden_shift": shift,
    }

    _, _, logits = int_forward(xte_u8, params)
    int_acc = float((logits.argmax(1) == yte.numpy()).mean())
    print(f"  float test acc  {float_acc * 100:.2f}%")
    print(f"  int   test acc  {int_acc * 100:.2f}%")
    print(f"  hidden shift    {shift}")
    print(f"  |acc1| max      {int(np.abs(acc1_tr).max())}  (int32 headroom ok)")

    # ---- reference batch: 5 images, the natural systolic batch size -------
    ref_idx = list(range(TILE))
    ref_x = xte_u8[ref_idx]
    ref_acc1, ref_h, ref_logits = int_forward(ref_x, params)

    # ---- one 5x5 tile of layer 1 ----------------------------------------
    # C = X_tile @ W_tile  where
    #   X_tile[i][k] = pixel k of image i          (5 images x 5 pixels)
    #   W_tile[k][j] = weight from pixel k to hidden unit j
    # This is exactly one pass of the 5x5 output-stationary array.
    #
    # The corner of an MNIST image is all background, so a tile taken from
    # feature 0 would be an all-zero matrix and prove nothing. Pick the
    # 5-feature block carrying the most signal in the reference batch.
    block_energy = ref_x.reshape(TILE, -1, TILE).sum(axis=(0, 2))
    kblock = int(block_energy.argmax())
    k0 = kblock * TILE
    x_tile = ref_x[:TILE, k0:k0 + TILE]
    w_tile = w1q[:TILE, k0:k0 + TILE].T   # (in, out) orientation
    c_tile = x_tile @ w_tile

    doc = {
        "arch": {"grid": GRID, "in_dim": IN_DIM, "hidden": HIDDEN,
                 "classes": CLASSES, "tile": TILE},
        "scales": {"s_x": s_x, "s_w1": s_w1, "s_h": s_h, "s_w2": s_w2},
        "accuracy": {"float": float_acc, "int": int_acc},
        "params": params,
        "reference": {
            "labels": yte.numpy()[ref_idx].tolist(),
            "x": ref_x.tolist(),
            "acc1": ref_acc1.tolist(),
            "hidden": ref_h.tolist(),
            "logits": ref_logits.tolist(),
            "pred": ref_logits.argmax(1).tolist(),
        },
        "tile": {
            "feature_offset": k0,
            "hidden_offset": 0,
            "x": x_tile.tolist(),
            "w": w_tile.tolist(),
            "c": c_tile.tolist(),
        },
    }
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"wrote {OUT}")
    print("\n5x5 tile (X @ W) the CaST array must reproduce:")
    for row in c_tile:
        print("   ", "  ".join(f"{v:8d}" for v in row))


if __name__ == "__main__":
    main()
