# MNIST MLP on the CaST systolic array

A quantised PyTorch MLP whose matrix multiplies are reproduced, exactly, by the
5x5 systolic array in CaST.

```
ml/train_mnist_mlp.py   train + quantise + export  ->  ml/mnist_model.json
ml/gen_cast_tile.py     json -> examples/mnist_tile.cast (+ expected output)
```

## The network

Every dimension is a multiple of 5 so it tiles onto the array cleanly.

```
28x28 image --avgpool--> 10x10 = 100 pixels --> 20 hidden (ReLU) --> 10 classes
```

| | |
|---|---|
| float test accuracy | 92.85% |
| integer test accuracy | 92.80% |
| pixels | `uint8`, 0..255 |
| weights | `int8`, symmetric per tensor |
| accumulator | `int32` (peak seen: 140619, ample headroom) |
| hidden requantise | `>> 10`, then clamp to 255 |

## The integer forward pass

`int_forward` in the training script is the specification. It is written to use
only operations that behave correctly on CaST's unsigned wires:

```python
acc1 = x @ W1.T + b1        # multiply/add: two's complement, signed-correct
acc1 = maximum(acc1, 0)     # ReLU
h    = minimum(acc1 >> 10, 255)
acc2 = h @ W2.T + b2        # int32 logits
```

Two subtleties, both of which the generated CaST respects:

- **ReLU** cannot be written `if (acc < 0)` — `<` is an unsigned compare, so
  every negative looks *larger* than every positive. Use the sign-bit test
  `if (acc >= 2147483648)`.
- **The shift is applied only after ReLU**, where the value is guaranteed
  non-negative. CaST's `>>` is a logical shift; on a negative accumulator it
  would be wrong, but here it never sees one.

For the same reason, `print` formats wires as unsigned, so results are printed
in sign-magnitude form.

## Running the tile

`examples/mnist_tile.cast` computes one 5x5 block of layer 1:

```
C[i][j] = sum_k  X[i][k] * W[k][j]
```

for 5 test images and 5 hidden units. The weight tile contains negative
values, so a correct result is also a proof that signed multiplication works.

```bash
# on a node
./simulate.sh examples/mnist_tile.cast --duration=1000
diff <(./simulate.sh examples/mnist_tile.cast --duration=1000 | tail -27) \
     tests/expected/mnist_tile.expected
```

Expected first lines:

```
MnistTile: 5x5 layer-1 tile, signed weights
C = X * W:
  C[0][0] = -1473
  C[0][1] = 3617
  ...
```

The run takes 41 cycles (410 ns), just inside the 500 ns default — pass
`--duration=1000` so a small change does not silently truncate the output.

Without a node, check it with the behavioural simulator:

```bash
tools/cast_sim examples/mnist_tile.cast
```

## Regenerating

`ml/mnist_model.json` is committed, so `gen_cast_tile.py` runs on its own.
Re-running `train_mnist_mlp.py` downloads MNIST (~64 MB into `ml/data/`, which
is gitignored), retrains, and overwrites the json — the tile values will change
and `tests/expected/mnist_tile.expected` is regenerated to match.

## Where this goes next

The tile is one pass of the array. A full layer-1 evaluation for a batch of 5
images is 20 tiles of `X` against 4 tiles of `W`, accumulated in groups of 20 —
80 array passes. That needs a runtime `goto` loop stepping a tile index, not
more unrolling, since 100x20 multipliers will not fit.
