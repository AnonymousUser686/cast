"""Train a small MNIST MLP and export integer weights for the CaST hex array.

The network is deliberately shaped so every dimension is a multiple of 5, the
tile size of the systolic array:

    28x28 -> crop to the 20x20 digit box -> pool to 5x5 = 25 pixels
          -> 20 hidden (ReLU) -> 10 classes

Everything is quantised to integers the CaST hardware can hold:

    pixels    uint8   0..255
    weights   int8    -127..127   (symmetric, per tensor)
    bias      int32   in accumulator units
    accum     int32

The integer forward pass in ``int_forward`` is the *specification* the CaST
program must reproduce bit for bit. It uses only operations that are correct
on CaST's unsigned wires:

    * multiply / add  - two's complement, identical bits signed or unsigned
    * ReLU via a sign-bit test rather than ``< 0``, which is unsigned
    * right shift applied only AFTER ReLU, where the value cannot be negative
      (CaST's ``>>`` is logical, so it would be wrong on a negative)

Scaling up: GRID=10 with HIDDEN=20 reaches 94.6%, at 2200 weight registers and
88 array passes instead of 700 and 28. Raise GRID here and regenerate if the
larger design compiles comfortably.

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

GRID = 5           # pooled image is GRID x GRID
CROP = True        # crop to the 20x20 box the digits actually occupy
IN_DIM = GRID * GRID
HIDDEN = 20
CLASSES = 10
TILE = 5           # systolic array size, and images per array pass
N_IMAGES = 100     # test images exported for the CaST program (multiple of TILE)
EPOCHS = 8
BATCH = 128
SEED = 0


# ---------------------------------------------------------------- data


def preprocess(raw):
    """(N,28,28) uint8 -> (N, IN_DIM) float in [0,1]."""
    x = raw.float().unsqueeze(1) / 255.0
    if CROP:
        x = x[:, :, 4:24, 4:24]
    return F.adaptive_avg_pool2d(x, (GRID, GRID)).reshape(x.shape[0], -1)


def load_split(train):
    ds = datasets.MNIST(DATA, train=train, download=True,
                        transform=transforms.ToTensor())
    return preprocess(ds.data), ds.targets.clone()


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
            F.cross_entropy(net(xtr[idx]), ytr[idx]).backward()
            opt.step()
        with torch.no_grad():
            acc = (net(xte).argmax(1) == yte).float().mean().item()
        print(f"  epoch {epoch + 1}/{EPOCHS}  test acc {acc * 100:.2f}%")
    return net


# ---------------------------------------------------------- quantisation


def sym_quantize(w):
    """Symmetric int8 quantisation. Returns (int weights, scale)."""
    scale = float(np.abs(w).max()) / 127.0
    return np.clip(np.rint(w / scale), -127, 127).astype(np.int64), scale


def int_forward(x_u8, p):
    """Exact integer inference; the reference the CaST program must match."""
    w1, b1 = np.array(p["w1"]), np.array(p["b1"])
    w2, b2 = np.array(p["w2"]), np.array(p["b2"])
    shift = p["shift"]

    acc1 = x_u8 @ w1.T + b1                 # int32, may be negative
    acc1 = np.maximum(acc1, 0)              # ReLU (sign-bit test in CaST)
    h = np.minimum(acc1 >> shift, 255)      # requantise; safe, acc1 >= 0
    acc2 = h @ w2.T + b2                    # int32 logits
    return acc1, h, acc2


def pick_shift(acc1):
    """Smallest shift keeping every post-ReLU activation inside uint8."""
    s, peak = 0, int(np.maximum(acc1, 0).max())
    while (peak >> s) > 255:
        s += 1
    return s


# ------------------------------------------------------------- tiling


def tile_operand(m, kt_dim, jt_dim):
    """Split an (in, out) matrix into TILE x TILE blocks, k-major.

    Result index is ``kt * jt_dim + jt`` so the CaST program can address a
    block with a single runtime multiply-add.
    """
    return [m[kt * TILE:(kt + 1) * TILE, jt * TILE:(jt + 1) * TILE].tolist()
            for kt in range(kt_dim) for jt in range(jt_dim)]


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
        W1 = net.fc1.weight.numpy().astype(np.float64)   # (hidden, in)
        B1 = net.fc1.bias.numpy().astype(np.float64)
        W2 = net.fc2.weight.numpy().astype(np.float64)   # (classes, hidden)
        B2 = net.fc2.bias.numpy().astype(np.float64)

    print("quantising ...")
    s_x = 1.0 / 255.0                        # pixels are plain 0..255 ints
    w1q, s_w1 = sym_quantize(W1)
    b1q = np.rint(B1 / (s_x * s_w1)).astype(np.int64)

    xtr_u8 = np.rint(xtr.numpy() * 255.0).astype(np.int64)
    xte_u8 = np.rint(xte.numpy() * 255.0).astype(np.int64)

    shift = pick_shift(xtr_u8 @ w1q.T + b1q)
    s_h = s_x * s_w1 * (2 ** shift)          # hidden scale follows the shift
    w2q, s_w2 = sym_quantize(W2)
    b2q = np.rint(B2 / (s_h * s_w2)).astype(np.int64)

    params = {"w1": w1q.tolist(), "b1": b1q.tolist(),
              "w2": w2q.tolist(), "b2": b2q.tolist(), "shift": shift}

    acc1_all, _, logits_all = int_forward(xte_u8, params)
    int_acc = float((logits_all.argmax(1) == yte.numpy()).mean())
    print(f"  float test acc  {float_acc * 100:.2f}%")
    print(f"  int   test acc  {int_acc * 100:.2f}%")
    print(f"  hidden shift    {shift}")
    print(f"  |acc1| max      {int(np.abs(acc1_all).max())} (int32 headroom ok)")
    print(f"  |acc2| max      {int(np.abs(logits_all).max())}")

    # ---- reference set: N_IMAGES images, NB array-widths of work ---------
    ref = list(range(N_IMAGES))
    ref_x = xte_u8[ref]
    ref_acc1, ref_h, ref_logits = int_forward(ref_x, params)
    ref_pred = ref_logits.argmax(1)
    ref_labels = yte.numpy()[ref]
    ref_correct = int((ref_pred == ref_labels).sum())

    kt1, jt1 = IN_DIM // TILE, HIDDEN // TILE
    kt2, jt2 = HIDDEN // TILE, CLASSES // TILE
    nb = N_IMAGES // TILE                        # image batches of TILE each

    # ---- one 5x5 tile of layer 1, for the standalone tile demo -----------
    # An image corner is all background, so pick the pixel block carrying the
    # most signal -- an all-zero tile would prove nothing. The tile demo uses
    # just the first array-width of images.
    tile_x = ref_x[:TILE]
    kblock = int(tile_x.reshape(TILE, -1, TILE).sum(axis=(0, 2)).argmax())
    k0 = kblock * TILE
    x_tile = tile_x[:, k0:k0 + TILE]
    w_tile = w1q[:TILE, k0:k0 + TILE].T          # (in, out)

    doc = {
        "arch": {"grid": GRID, "crop": CROP, "in_dim": IN_DIM,
                 "hidden": HIDDEN, "classes": CLASSES, "tile": TILE,
                 "images": N_IMAGES, "nb": nb,
                 "kt1": kt1, "jt1": jt1, "kt2": kt2, "jt2": jt2},
        "shift": shift,
        "accuracy": {"float": float_acc, "int": int_acc},
        "params": params,
        "tiles": {
            # xt[ib * kt1 + kt][image within batch][pixel]
            "x": [ref_x[ib * TILE:(ib + 1) * TILE,
                        kt * TILE:(kt + 1) * TILE].tolist()
                  for ib in range(nb) for kt in range(kt1)],
            "w1": tile_operand(w1q.T, kt1, jt1),
            "b1": [b1q[jt * TILE:(jt + 1) * TILE].tolist()
                   for jt in range(jt1)],
            "w2": tile_operand(w2q.T, kt2, jt2),
            "b2": [b2q[jt * TILE:(jt + 1) * TILE].tolist()
                   for jt in range(jt2)],
        },
        "reference": {
            "labels": ref_labels.tolist(),
            "acc1": ref_acc1.tolist(),
            "hidden": ref_h.tolist(),
            "logits": ref_logits.tolist(),
            "pred": ref_pred.tolist(),
            "correct": ref_correct,
        },
        "tile": {
            "feature_offset": k0,
            "x": x_tile.tolist(),
            "w": w_tile.tolist(),
            "c": (x_tile @ w_tile).tolist(),
        },
    }
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"wrote {OUT}")
    print(f"\nreference set: {ref_correct} of {N_IMAGES} correct "
          f"({ref_correct * 100.0 / N_IMAGES:.0f}%)")


if __name__ == "__main__":
    main()
