# MNIST inference on a hexagonal systolic array

A quantised PyTorch MLP whose matrix multiplies are reproduced, exactly, by a
hexagonal systolic array written in CaST.

```
ml/train_mnist_mlp.py   train + quantise + export  ->  ml/mnist_model.json
ml/gen_cast_tile.py     json -> examples/mnist_tile.cast      (one tile)
ml/gen_cast_mlp.py      json -> examples/mnist_mlp_hex.cast   (full inference)
```

Both generators also write the expected output into `tests/expected/`, taken
straight from PyTorch's integer forward pass. The CaST programs match it line
for line.

## The network

Every dimension is a multiple of 5, the array's tile size.

```
28x28 -> crop to the 20x20 digit box -> pool to 10x10 = 100 pixels
      -> 40 hidden (ReLU) -> 10 classes
```

| | |
|---|---|
| float test accuracy | 96.01% |
| **integer test accuracy** | **95.97%** |
| pixels | `uint8`, 0..255 |
| weights | `int8`, symmetric per tensor |
| accumulator | `int32` (peak 237831 — ample headroom) |
| hidden requantise | `>> 10`, clamp to 255 |
| weight registers | 4400 |
| images classified | 100, in 20 batches of 5 |
| array passes | 176 per batch (160 for layer 1, 16 for layer 2); 3520 total |
| run length | 52983 cycles |

## Scaling

Measured on this pipeline, same quantisation throughout:

| grid | inputs | hidden | int accuracy | weight registers | passes/batch |
|---|---|---|---|---|---|
| 5 | 25 | 20 | 88.8% | 700 | 28 |
| **10** | **100** | **40** | **96.0%** | **4400** | **176** |
| 15 | 225 | 60 | 97.4% | 14100 | 564 |
| 20 | 400 | 100 | 97.8% | 41000 | 1640 |

Weight registers are what bite first: every weight is a `seq.compreg`, so the
generated Verilog grows with the model, and grid 20 is 41000 of them.

**Roughly 97.8% is the ceiling for a two-layer MLP**, at any resolution — the
last two rows buy 0.4 points for 3× the hardware. Reaching 99% on MNIST needs
convolutions, which is a different dataflow: weight reuse across spatial
positions rather than a single dense tile stream, so the array would have to
be re-scheduled rather than just resized.

Raise `GRID`/`HIDDEN` in the training script and rerun both generators to move
along the table.

## The hexagonal array

`C[i][j] += A[i][k]*B[k][j]` has a 5×5×5 iteration space. Project that cube
along its **(1,1,1) diagonal** and schedule iteration `(i,j,k)` at time
`t = i+j+k`. Each operand then travels along one of three axes 120° apart, one
PE per cycle:

```
PE(U,V) = (i-k+4, j-k+4)          U,V in 0..8

    A[i][k] -> +V      B[k][j] -> +U      C[i][j] -> (-1,-1)
```

**Nothing is stationary.** That is the difference from the square array in
[examples/systolic_matmul.cast](../examples/systolic_matmul.cast), where `C`
sits still and only the operands move. Here the partial sums stream through
too, which is what makes the connectivity hexagonal — six neighbours per PE.

The live sites form a hexagon inside the 9×9 bounding box (61 of 81); the
corners stay zero. Because the projection direction equals the schedule
vector, a PE holds a matching `(i,j,k)` triple only every third cycle; on the
other two it passes data through, and the partial sums riding those slots are
never captured.

Entry and exit are the only fiddly part, and both turn out to be free:

| | where | when |
|---|---|---|
| `A[i][k]` enters (at `j == 0`) | `PE(i+V, V)`, `V = 4-k` | `t = i-V+4` |
| `B[k][j]` enters (at `i == 0`) | `PE(U, U+j)`, `U = 4-k` | `t = j-U+4` |
| `C[i][j]` starts at zero (`k == 0`) | `PE(i+4, j+4)` | `t = i+j` |
| `C[i][j]` is finished (`k == 4`) | `PE(i,j)` | `t = i+j+4` |

Every site and time is a compile-time constant, so each edge costs one
comparison against `t` — no address arithmetic, no division. The whole array
is 61 lines of CaST in the `step` state.

## Signedness

Weights are negative roughly half the time. Multiply and add are sign-agnostic
in two's complement, so the MAC itself needs nothing special. Three places do:

1. **ReLU** — `if (x < 0)` is an *unsigned* compare and never fires. Test the
   sign bit: `if ((x >> 31) == 1)`.
2. **Requantisation** — `>>` is a logical shift. It is only correct here
   because ReLU has just guaranteed a non-negative value.
3. **Argmax** — adding `1 << 31` flips the sign bit, which maps int32 ordering
   onto uint32 ordering exactly, so unsigned compares rank logits correctly.

Printing is affected too: `print` formats wires as unsigned, so
`mnist_tile.cast` prints in sign-magnitude form.

See [../examples/signed_arithmetic.cast](../examples/signed_arithmetic.cast)
for the minimal test of all this.

## Batching

The array is 5 wide in the image dimension, so **5 images go through
together** — one pass computes a 5×5 block of (images × units). 100 images is
20 such batches, run back to back through the same hardware.

Only the pixels (10000 registers) and labels are stored per image. The
accumulators are cleared and reused each batch, and the score is a single
running counter rather than 100 stored predictions.

## Running

```bash
./simulate.sh examples/mnist_mlp_hex.cast --duration=600000
```

Expected — one line per image, then the score:

```
MnistMLP: 100 images, 100 pixels -> 40 hidden -> 10 classes
image 0  label 7  guess 7  ok
image 1  label 2  guess 2  ok
...
image 99  label 9  guess 9  ok
correct: 98 of 100
accuracy: 98%
```

**98 of these particular 100 images is a lucky draw, not the model's
accuracy** — the honest number is 95.97% over the full 10,000-image test set.
A 100-image sample has a standard error of about 2 points, so quote 96%.
Since the run is exactly 100 images the count *is* the percentage, which
saves synthesising a divider.

Diff it against `tests/expected/mnist_mlp_hex.expected`; all 103 lines,
including which 2 images are wrong, come from PyTorch. The run is 52983
cycles (529830 ns), so `--duration=600000` leaves room.

`examples/mnist_tile.cast` is the smaller companion: one 5×5 layer-1 tile on
the *square* array, printed in sign-magnitude. Useful to confirm signed
multiply on its own before running the whole network.

Without a node, both check out under the behavioural simulator:

```bash
tools/cast_sim --max-cycles=60000 examples/mnist_mlp_hex.cast   # ~45 s
tools/cast_sim examples/mnist_tile.cast
```

## Regenerating

`ml/mnist_model.json` is committed, so both generators run on their own.
Re-running `train_mnist_mlp.py` downloads MNIST (~64 MB into `ml/data/`,
gitignored), retrains, and overwrites the json — rerun both generators
afterwards so the expected-output files match.
