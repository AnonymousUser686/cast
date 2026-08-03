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
28x28 -> crop to the 20x20 digit box -> pool to 5x5 = 25 pixels
      -> 20 hidden (ReLU) -> 10 classes
```

| | |
|---|---|
| float test accuracy | 89.87% |
| **integer test accuracy** | **89.78%** |
| pixels | `uint8`, 0..255 |
| weights | `int8`, symmetric per tensor |
| accumulator | `int32` (peak 82664 — ample headroom) |
| hidden requantise | `>> 9`, clamp to 255 |
| weight registers | 700 |
| array passes | 28 (20 for layer 1, 8 for layer 2) |
| run length | 449 cycles |

Scaling up: `GRID = 10` in the training script reaches **94.6%**, at 2200
weight registers and 88 passes. Raise it and regenerate if the larger design
compiles comfortably on the node.

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

## Running

```bash
./simulate.sh examples/mnist_mlp_hex.cast --duration=8000
```

Expected (20 hidden activations for image 0, then the five guesses):

```
MnistMLP: 5 images, 25 pixels -> 20 hidden -> 10 classes
h[0] = 0
h[1] = 12
...
image 0  label 7  guess 7
image 1  label 2  guess 2
image 2  label 1  guess 1
image 3  label 0  guess 0
image 4  label 4  guess 4
```

Diff it against `tests/expected/mnist_mlp_hex.expected`. The run is 449
cycles (4490 ns), so `--duration=8000` leaves room.

`examples/mnist_tile.cast` is the smaller companion: one 5×5 layer-1 tile on
the *square* array, printed in sign-magnitude. Useful to confirm signed
multiply on its own before running the whole network.

Without a node, both check out under the behavioural simulator:

```bash
tools/cast_sim --max-cycles=1000 examples/mnist_mlp_hex.cast
tools/cast_sim examples/mnist_tile.cast
```

## Regenerating

`ml/mnist_model.json` is committed, so both generators run on their own.
Re-running `train_mnist_mlp.py` downloads MNIST (~64 MB into `ml/data/`,
gitignored), retrains, and overwrites the json — rerun both generators
afterwards so the expected-output files match.
