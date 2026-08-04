# MNIST on a hexagonal systolic array

A handwritten-digit classifier that runs as **hardware**. A small neural
network is trained in PyTorch, quantised to integers, and compiled into a
CaST state machine whose matrix multiplies are performed by a hexagonal
systolic array. That state machine becomes SystemVerilog, and the Verilog
simulation classifies 100 MNIST test images and prints its own score.

The interesting claim is not that it works — it is that the hardware's answers
match PyTorch's **exactly, on every image**, including the two it gets wrong.

```
image 0  label 7  guess 7  ok
image 1  label 2  guess 2  ok
...
image 8  label 5  guess 6  WRONG
...
correct: 98 of 100
accuracy: 98%
cycles: 52981
cycles per image: 529.8
MACs: 440000 (8.3 per cycle)
```

| | |
|---|---|
| accuracy (full 10,000-image test set) | **95.97%** integer, 96.01% float |
| score on the 100 images it runs | 98 / 100 |
| network | 100 pixels → 40 hidden (ReLU) → 10 classes |
| arithmetic | integer only — `uint8` pixels, `int8` weights, `int32` accumulators |
| program | 3,310 lines of CaST, ~15,200 registers |
| classification | **52,981 cycles** for 100 images, 440,000 MACs at 8.3/cycle |

> **Quote 95.97%, not 98%.** The program scores the first 100 test images and
> happens to get 98 of them. A 100-image sample has a standard error of about
> 2 points, so 98% is a lucky draw, not the model's accuracy. The honest
> figure is 95.97% over all 10,000 test images. (The run is exactly 100 images
> so the count *is* the percentage, which conveniently avoids synthesising a
> divider.)

---

## Contents

1. [Run it](#1-run-it)
2. [The pipeline](#2-the-pipeline)
3. [Why a neural net is awkward on this hardware](#3-why-a-neural-net-is-awkward-on-this-hardware)
4. [The network, and turning it into integers](#4-the-network-and-turning-it-into-integers)
5. [The hexagonal array](#5-the-hexagonal-array)
6. [How the program is scheduled](#6-how-the-program-is-scheduled)
7. [Signedness: three places it bites](#7-signedness-three-places-it-bites)
8. [What it costs, and why not 99%](#8-what-it-costs-and-why-not-99)
9. [How we know it is correct](#9-how-we-know-it-is-correct)
10. [Changing the model](#10-changing-the-model)
11. [File map](#11-file-map)

---

## 1. Run it

On a node with `castc` (see [../GETTING_STARTED.md](../GETTING_STARTED.md)):

```bash
./simulate.sh examples/mnist_mlp_hex.cast --duration=600000
```

The long `--duration` is required — the run is 529,860 ns and the default is
500 ns. This is the largest program in the repo, so **expect `castc` and
especially `iverilog` to take noticeably longer than the other examples.**

Without a node, the behavioural simulator needs only a C++ compiler:

```bash
c++ -std=c++20 -O2 -I src/Frontend tools/cast_sim.cpp \
    src/Frontend/CastParser.cpp -o tools/cast_sim

tools/cast_sim --max-cycles=60000 examples/mnist_mlp_hex.cast   # ~45 s
tools/cast_sim examples/mnist_tile.cast                         # instant
```

Either way, the output should match `tests/expected/mnist_mlp_hex.expected`
line for line — all 106 lines of it.

---

## 2. The pipeline

Three stages, each a separate script, connected by one JSON file:

```
   ml/train_mnist_mlp.py          ml/gen_cast_mlp.py           castc          iverilog
        │                              │                         │               │
  MNIST ┴─► train, quantise ─► mnist_model.json ─► .cast ────► .sv ──────────► a.out ─► output
              (PyTorch)          (weights +        (hardware)  (Verilog)      (vvp)
                                  reference
                                  answers)
```

**`ml/train_mnist_mlp.py`** trains the network, quantises it to integers, and
writes `ml/mnist_model.json`. That file holds the integer weights *already
split into 5×5 tiles* in the order the hardware wants them, plus the exact
answers PyTorch's integer forward pass produces for the 100 test images.

**`ml/gen_cast_mlp.py`** reads the JSON and writes two files: the CaST program
`examples/mnist_mlp_hex.cast`, and `tests/expected/mnist_mlp_hex.expected`,
which is what the program must print. Only the constant tables are generated —
the state machine itself is written out literally, because the array is the
point of the exercise and should read as hardware, not as generated text.

**`castc` → `iverilog`** is the normal CaST flow: state machine → MLIR/CIRCT →
synthesisable SystemVerilog → simulation.

There is also **`ml/gen_cast_tile.py`**, which emits
`examples/mnist_tile.cast`: a single 5×5 tile of layer 1 on the simpler
*square* array. It exists to confirm signed multiplication in isolation before
you go looking for bugs in the big program.

The JSON is committed, so you can regenerate the CaST without installing
PyTorch or downloading MNIST.

---

## 3. Why a neural net is awkward on this hardware

CaST describes synthesisable hardware, and four of its properties shape
everything below. None of them are bugs; they are what "this is hardware"
means.

**One state per clock cycle.** Everything inside a state body happens in that
one cycle, in parallel. Register reads return the value from the *start* of
the cycle. So this is wrong:

```cast
compute: { total = a + b; print("total = ", total); goto next; }   // prints the OLD total
```

Values must be used in the *next* state. The exception is inside a for-loop
body, where reads do see writes made earlier in the same body — which is what
lets the epilogue chain bias → ReLU → shift → clamp in a single cycle.

**For loops fully unroll.** A loop is replicated hardware, not iteration, and
its bound must be a compile-time constant. 25 multiplies in a loop means 25
multipliers. Anything that must genuinely repeat at runtime — like the 176
array passes — has to be a state that jumps back to itself with `goto`.

**No I/O.** There is no file system, no input port, no way to load data at
runtime. Every pixel and every weight must be baked into the program as a
register written once during reset. This is the single biggest constraint on
the design: **model size is limited by how many registers you are willing to
generate**, not by compute.

**No floats, and no signed types.** Every register is a `uint32`. Floating
point does not exist. Negative numbers exist only as two's complement bit
patterns that *you* have to interpret correctly — see
[§7](#7-signedness-three-places-it-bites).

Together these force the shape of the whole thing: a small, integer-only,
fully-unrolled-where-possible network with its data compiled in.

---

## 4. The network, and turning it into integers

Every dimension is a multiple of 5, the array's tile size — otherwise the last
tile of each row would be a partial one, and the array would need masking
logic it does not have.

```
28×28  ──crop──►  20×20     the box the digits actually occupy
       ──pool──►  10×10     = 100 pixels
       ──fc1──►   40 hidden  (ReLU)
       ──fc2──►   10 classes
```

Cropping first matters more than it looks: MNIST digits sit inside a 20×20 box
with a 4-pixel border of guaranteed background. Pooling 20×20 → 10×10 spends
every one of the 100 inputs on the digit rather than on empty margin.

Training is 12 epochs of Adam with cosine decay, on CPU.

### Quantisation

The trained floats are converted to integers the hardware can hold:

| | type | scheme |
|---|---|---|
| pixels | `uint8` 0..255 | the natural pixel range; no scale needed |
| weights | `int8` −127..127 | symmetric, one scale per tensor |
| biases | `int32` | pre-scaled into accumulator units |
| accumulators | `int32` | |

Symmetric means the quantised value is just `round(w / scale)` with
`scale = max|w| / 127` — no zero-point offset, so the hardware never has to
subtract one. Biases are folded into accumulator units in advance, so the
hardware adds them directly with no rescaling.

Between the layers, the `int32` accumulator has to come back down to something
`int8` weights can multiply against. That is a single right shift by 10,
chosen at export time as the smallest shift keeping every activation inside
`uint8`, followed by a clamp to 255. Choosing it from the *training* set means
it is not tuned to the test data.

The accumulator swings between **−371,327 and +237,831** across the test set,
against an `int32` range of ±2.1 billion — about 5,700× headroom, so overflow
is not a concern here.

Quantisation costs **0.04 percentage points**: 96.01% float → 95.97% integer.

The function `int_forward()` in the training script is the *specification*.
It is written using only operations that are correct on unsigned wires, and
the CaST program must reproduce it bit for bit.

---

## 5. The hexagonal array

This is the part worth reading.

A matrix multiply `C[i][j] += A[i][k] * B[k][j]` over 5×5×5 has a
three-dimensional iteration space — a cube of 125 multiply-accumulates. A
systolic array is what you get when you *project* that cube onto a plane of
processing elements and assign each point a time.

The usual square array projects along `k`: `C` sits still in a 5×5 grid while
`A` and `B` stream past it. That is `examples/systolic_matmul.cast`.

This one projects along the cube's **(1,1,1) diagonal** and schedules
iteration `(i,j,k)` at time `t = i + j + k`:

```
PE(U,V) = (i − k + 4,  j − k + 4)          U, V in 0..8

     A[i][k] moves +V        B[k][j] moves +U        C[i][j] moves (−1,−1)
```

**Nothing is stationary.** All three operands flow, each along one of three
axes 120° apart, one PE per cycle. Partial sums stream through the array
alongside the data that feeds them. That is exactly what makes the
connectivity hexagonal: each PE talks to six neighbours (±U, ±V, ±(1,1))
rather than four. It is the classic Kung–Leiserson hex array.

Because `U − V = i − j` and both are in 0..4, the live PEs are the band
`|U − V| ≤ 4` inside the 9×9 bounding box — a hexagon, 61 sites of 81:

```
      V: 0 1 2 3 4 5 6 7 8
 U=0     # # # # # . . . .
 U=1     # # # # # # . . .
 U=2     # # # # # # # . .
 U=3     # # # # # # # # .
 U=4     # # # # # # # # #
 U=5     . # # # # # # # #
 U=6     . . # # # # # # #
 U=7     . . . # # # # # #
 U=8     . . . . # # # # #
```

The 20 sites in the two cut corners are never reached by data. The program
still runs its multiply-accumulate over the whole 9×9 grid — those PEs simply
compute `0 + 0*0` — because feeding a rectangle and letting the corners idle
is cheaper than special-casing the boundary.

### Every third cycle

Solving the schedule for `k` gives `k = (t − U − V + 8) / 3`. That is an
integer only when `t ≡ U + V − 8 (mod 3)`, so **a PE holds a matching
`(i,j,k)` triple only on one cycle in three.** On the other two it just passes
data through, and the partial sums riding those slots are never captured.

This is the price of projecting along the schedule direction, and it is why
the array is only about a third utilised. At its widest — `t = 6` — 19 PEs are
multiplying at once; averaged over the pass it is 125 MACs in 13 cycles, about
9.6 per cycle.

### Entry and exit are free

The fiddly part of any systolic array is getting data in at the right place at
the right time. Here every site and time works out to a compile-time constant:

| | where | when |
|---|---|---|
| `A[i][k]` enters (at `j == 0`) | `PE(i+V, V)`, `V = 4−k` | `t = i − V + 4` |
| `B[k][j]` enters (at `i == 0`) | `PE(U, U+j)`, `U = 4−k` | `t = j − U + 4` |
| `C[i][j]` starts at zero (`k == 0`) | `PE(i+4, j+4)` | `t = i + j` |
| `C[i][j]` is finished (`k == 4`) | `PE(i, j)` | `t = i + j + 4` |

So each hexagon edge costs a single comparison against `t` — no address
arithmetic, no division, no boundary FSM. The whole array is one state,
`step`, of about forty lines: one loop for the MAC, one shift loop per
operand direction, and one guarded loop per edge of the table above.

---

## 6. How the program is scheduled

### The batch dimension is the image dimension

One array pass multiplies a 5×5 block of `A` by a 5×5 block of `B`. For layer
1 that is *5 images × 5 pixels* against *5 pixels × 5 hidden units* — so
**five images go through the array together, for free.** The batch dimension
was already there; classifying 100 images is 20 such batches run back to back
through the same hardware, not a bigger array.

Per batch:

* **layer 1** = 20 pixel-tiles × 8 hidden-tiles = **160 passes**
* **layer 2** = 8 hidden-tiles × 2 class-tiles = **16 passes**
* total **176 passes per batch**, 3,520 over the whole run

Layer 2 needs no repacking: layer 1's output tiles are partitioned by hidden
unit, which is exactly how layer 2 wants its inputs. The accumulators are
rewritten in place and re-read as the next layer's activations.

### States

```
setup                                    once: write every constant
  │
  ├─► batch                              once per group of 5 images  ─┐
  │     │                                                             │
  │     ├─► stage ─► step ─► accum   ×160   layer 1                    │
  │     ├─► epilogue1                       bias, ReLU, requantise     │  ×20
  │     ├─► stage ─► step ─► accum   ×16    layer 2                    │
  │     ├─► epilogue2                       bias, sign-flip            │
  │     ├─► argmax                          5 predictions              │
  │     └─► report                   ×5     print and score           ─┘
  │
  └─► summary… ─► halt                   score, then the timing figures
```

| state | cycles | what it does |
|---|---|---|
| `setup` | 1 | writes all 14,550 constants — pixels, weights, biases, labels |
| `batch` | 1 | clears the accumulators for a new group of 5 images |
| `stage` | 1 | loads the two 5×5 operand tiles, clears the array, injects the first cell |
| `step` | 13 | one array cycle each, `t = 0..12`; the hexagon in motion |
| `accum` | 1 | adds the finished 5×5 product into the accumulator, advances the tile counters |
| `epilogue1` | 1 | bias + ReLU + `>> 10` + clamp, all 200 activations in one cycle |
| `epilogue2` | 1 | bias, then offset the logits for the argmax compare |
| `argmax` | 1 | picks the largest of 10 logits for each of 5 images |
| `report` | 5 | one image printed per cycle, scoring as it goes |
| `summary*` | 5 | score, percentage, and the three timing lines |

`t = 0..12` inclusive is 13 cycles because `LAST_T = 3 × (5−1) = 12` is the
largest value of `i + j + k`. So a pass is `1 + 13 + 1 = 15` cycles, a batch
is `1 + 176×15 + 1 + 1 + 1 + 5 = 2,649`, and the classification is
`1 + 20×2,649 = 52,981` cycles. The five summary states bring the whole
program to 52,986.

The generator computes those numbers rather than hardcoding them, so the
`--duration` in the header never goes stale when the model changes.

### The cycle counter

The `cycles:` line the program prints is **measured, not asserted**. A
register `cyc` is incremented once in every state that does classification
work — `setup`, `batch`, `stage`, `step`, `accum`, both epilogues, `argmax`
and `report` — which is exactly once per clock cycle. The summary states
deliberately leave it alone, so it stops at the cost of the work rather than
counting the cycles spent printing the report.

That makes the expected-output file a real prediction: `cycles: 52981` is
computed by the generator from the schedule, and the program independently
arrives at the same number by counting. If a future change to the state
machine altered the schedule, the diff would catch it.

The two derived lines below it — `cycles per image` and `MACs per cycle` — are
compile-time constants rather than runtime arithmetic. A 32-bit divider is a
lot of hardware for a line of text, and the measured count above them is what
validates them.

For comparing against other hardware, the useful shape of it is:

| | |
|---|---|
| useful MACs | 440,000 = 100 images × (100×40 + 40×10) |
| cycles | 52,981 |
| **throughput** | **8.3 MACs/cycle**, 529.8 cycles/image |
| multipliers instantiated | 81 (the full 9×9 grid) |
| peak achievable | 19 MACs in one cycle, at `t = 6` |

The gap between 8.3 and 81 is worth understanding before quoting it as an
efficiency figure, because it comes from two different places:

* **81 → ~20 is inherent to this array.** Twenty of the 81 sites are dead
  corners, and as [§5](#5-the-hexagonal-array) explains, a live PE only holds
  a matching `(i,j,k)` triple one cycle in three. So even perfectly fed, the
  hexagon tops out near 61/3 ≈ 20 MACs/cycle.
* **~20 → 8.3 is the simplification noted below.** The array is drained and
  reloaded between passes rather than pipelined, so each pass pays a ramp-up
  and ramp-down (125 MACs spread over 13 `step` cycles = 9.6/cycle), and the
  `stage`/`accum` cycles around it add no MACs at all.

The second gap is recoverable — streaming the next tile in behind the current
one would roughly double throughput. The first is the price of the (1,1,1)
projection that makes the array hexagonal in the first place.

Two deliberate simplifications: the array is cleared and reloaded for every
pass rather than pipelined across passes, and `report` prints one image per
cycle rather than five in one state — prints issued in the same state share a
clock edge, and Verilog leaves their order undefined.

### Why the loop bounds look off by one

Throughout the program you will see `if (kt < 19)` where 20 tiles are being
counted, and `if (pi < 4)` for 5 images. This is not a bug. `kt++` and the
comparison are in the same state, so the comparison reads `kt`'s
*start-of-cycle* value — it must test against the **last index**, not the
count.

---

## 7. Signedness: three places it bites

About half the weights are negative, but every CaST register is a `uint32`.
Multiplication and addition are sign-agnostic in two's complement — the bits
are identical whether you call the operands signed or unsigned — so the MAC
itself needs nothing special. Three operations are *not* sign-agnostic:

**1. ReLU.** `if (x < 0)` compiles to an unsigned compare and can never fire.
Test the sign bit directly:

```cast
if ((acc1[p][i][j] >> 31) == 1) { acc1[p][i][j] = 0; }
```

**2. The requantising shift.** `>>` is a *logical* shift, so on a negative
value it would shift in zeros and produce a large positive number. It is
correct here only because ReLU has just guaranteed the value is non-negative —
the ordering of those two lines is load-bearing.

**3. Argmax.** Logits are genuinely signed and unsigned compares would rank
them wrongly. Adding `1 << 31` flips the sign bit, which maps `int32` ordering
onto `uint32` ordering exactly, so the unsigned compares then rank correctly:

```cast
acc2[p][i][c] = (acc2[p][i][c] + b2t[p][c]) + (1 << 31);
```

Printing is affected too — `print` formats wires as unsigned, which is why
`mnist_tile.cast` prints its results in sign-magnitude form. It does not
affect the main program, which only ever prints small non-negative numbers.

[`../examples/signed_arithmetic.cast`](../examples/signed_arithmetic.cast) is
the minimal test of all three.

---

## 8. What it costs, and why not 99%

Every constant is a `seq.compreg`, so the generated Verilog grows with the
model:

| | registers |
|---|---|
| pixels (100 images × 100) | 10,000 |
| layer 1 weights | 4,000 |
| layer 2 weights | 400 |
| biases, labels | 150 |
| the hexagonal array (`ha`/`hb`/`hc`/`hr`, 4 × 81) | 324 |
| accumulators, staging, counters (including `cyc`) | 343 |
| **total** | **≈ 15,200** |

Note that the *pixels dominate* — two thirds of the design is baked-in test
data, a direct consequence of having no runtime I/O.

### Scaling

Measured on this exact pipeline, same quantisation throughout:

| grid | inputs | hidden | int accuracy | weight regs | passes/batch |
|---|---|---|---|---|---|
| 5 | 25 | 20 | 88.8% | 700 | 28 |
| **10** | **100** | **40** | **96.0%** | **4,400** | **176** |
| 15 | 225 | 60 | 97.4% | 14,100 | 564 |
| 20 | 400 | 100 | 97.8% | 41,000 | 1,640 |

Grid 10 is the knee of the curve: it cuts the error rate from 11.2% to 4.0%
for 6× the weights, whereas grid 20 buys only 1.8 points more for 9× again.

### The 99% question

**MNIST is a 99%+ problem, but not for this network.** Grid 20 is *full
resolution* — no pooling loss at all — and it still stops around 97.8%. That
is the ceiling for a two-layer MLP, and it is a property of the model, not of
the hardware or the quantisation.

Getting to 99% needs **convolutions**, and that is not a bigger array — it is a
different dataflow. A convolution reuses each weight across every spatial
position, so instead of streaming dense independent tiles you slide a small
filter over a window and hold the filter still. The hexagonal projection still
applies, but the injection geometry in [§5](#5-the-hexagonal-array) would have
to be rederived, and weight reuse changes what is worth keeping in registers.
Real work, and a genuinely interesting follow-on.

---

## 9. How we know it is correct

The claim is *bit-exact agreement with PyTorch*, and it is checked rather than
asserted:

1. `int_forward()` in the training script is the integer specification.
2. The generator runs it on the 100 exported images and writes the results —
   every prediction, including the wrong ones — into
   `tests/expected/mnist_mlp_hex.expected`, and asserts its own score matches
   the one recorded at training time.
3. The CaST program's output is diffed against that file. All 106 lines match,
   including `image 8  label 5  guess 6  WRONG` and
   `image 33  label 4  guess 6  WRONG`.

Agreeing on which images are *wrong* is the strong part. A rounding error
anywhere in 3,520 array passes would change a logit, and near-ties like those
two are exactly where it would show.

The `cycles: 52981` line is checked the same way, from the other direction:
the generator predicts it from the schedule, the program counts it at runtime,
and the diff compares the two. So the timing figure is verified rather than
documented, and any change to the state machine that costs a cycle shows up
immediately.

If the diff ever fails, work up the chain: run `mnist_tile.cast` first (one
tile, prints raw accumulator values) to isolate signed multiply from the
scheduling around it.

> **Note on Windows:** redirecting simulator output in PowerShell writes
> UTF-16, and `diff` will report "Binary files differ" with no useful detail.
> Redirect from a POSIX shell, or use `Out-File -Encoding utf8`.

These two examples are **not** in `ninja check`. The test harness hardcodes
`--duration=100`, which is far too short, and adding a per-test duration to
`tests/run_cast_test.sh` is the fix if that becomes worth doing.

---

## 10. Changing the model

The JSON is committed, so regenerating the CaST needs nothing but Python:

```bash
python ml/gen_cast_mlp.py       # -> examples/mnist_mlp_hex.cast + expected
python ml/gen_cast_tile.py      # -> examples/mnist_tile.cast    + expected
```

To change the network, edit the constants at the top of
`ml/train_mnist_mlp.py`:

```python
GRID   = 10     # pooled image is GRID x GRID
HIDDEN = 40     # hidden units
N_IMAGES = 100  # test images baked into the program
```

`GRID*GRID`, `HIDDEN`, `CLASSES` and `N_IMAGES` must all be multiples of
`TILE` (5). Then:

```bash
python ml/train_mnist_mlp.py    # downloads MNIST (~64 MB into ml/data/, gitignored)
python ml/gen_cast_mlp.py
python ml/gen_cast_tile.py
```

**Rerun both generators after retraining**, or the expected-output files will
describe the old weights. The generator recomputes the cycle count and the
`--duration` in the program header, so those stay correct automatically.

If `N_IMAGES` is not 100 the generator emits a division to compute the
percentage, which synthesises a divider — cheap enough, but 100 avoids it.

---

## 11. File map

| file | what it is |
|---|---|
| [train_mnist_mlp.py](train_mnist_mlp.py) | train, quantise, export — the integer forward pass here is the spec |
| [gen_cast_mlp.py](gen_cast_mlp.py) | JSON → `examples/mnist_mlp_hex.cast` + expected output |
| [gen_cast_tile.py](gen_cast_tile.py) | JSON → `examples/mnist_tile.cast` + expected output |
| `mnist_model.json` | committed weights, tiled, plus PyTorch's reference answers |
| [../examples/mnist_mlp_hex.cast](../examples/mnist_mlp_hex.cast) | the full classifier — generated, do not hand-edit |
| [../examples/mnist_tile.cast](../examples/mnist_tile.cast) | one 5×5 tile on the square array; the signedness proof |
| [../examples/systolic_matmul.cast](../examples/systolic_matmul.cast) | the square, output-stationary array, for contrast |
| [../examples/signed_arithmetic.cast](../examples/signed_arithmetic.cast) | minimal test of the three signedness workarounds |

Language reference: [../readme.md](../readme.md). Running things on a node:
[../GETTING_STARTED.md](../GETTING_STARTED.md).
