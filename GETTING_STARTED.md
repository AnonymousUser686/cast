# Getting started with cast on the COSMOS nodes

A practical guide for someone handed this repo and node access. It covers
getting a working compiler, running programs on a node, writing correct cast
code, and the handful of behaviours that will otherwise waste your afternoon.

For the full language reference see [readme.md](readme.md). This document is
the "how do I actually run something" companion.

**Contents**

1. [What cast is](#1-what-cast-is)
2. [Get the compiler (no building)](#2-get-the-compiler-no-building)
3. [Install it on a node](#3-install-it-on-a-node)
4. [Run your first program](#4-run-your-first-program)
5. [Writing cast programs](#5-writing-cast-programs)
6. [Gotchas that will bite you](#6-gotchas-that-will-bite-you)
7. [Trying things without a node](#7-trying-things-without-a-node)
8. [Troubleshooting](#8-troubleshooting)
9. [Rebuilding the compiler](#9-rebuilding-the-compiler)

---

## 1. What cast is

cast is a language for describing hardware state machines. A program declares
one or more **machines** (finite state machines with registers and channel
I/O); the compiler `castc` turns them into synthesisable SystemVerilog via
MLIR/CIRCT. You then simulate that Verilog with `iverilog` + `vvp`.

```
your_program.cast ──castc──> program.sv ──iverilog──> a.out ──vvp──> printed output
```

`simulate.sh` runs all three steps for you.

---

## 2. Get the compiler (no building)

**Do not build LLVM/MLIR/CIRCT from source.** The readme describes that path;
it takes hours and ~50 GB. A GitHub Actions workflow already builds a
ready-to-run Linux binary.

1. Go to the repo's **Actions** tab.
2. Click the most recent green ✓ run of **build castc**.
3. Scroll to **Artifacts** at the bottom and download **`castc-linux-x64`**.
4. Unzip it. You get a single ~16 MB file named `castc`.

That binary is **statically linked**, so it runs on the nodes regardless of
their glibc version. (This matters: the nodes are Ubuntu 22.04 / glibc 2.35,
while the binary is built on 24.04. A normal dynamic build would refuse to
start.)

> No green run available, or artifacts expired (they age out after ~90 days)?
> Open **Actions → build castc → Run workflow** to produce a fresh one.
> It takes about 15 minutes.

---

## 3. Install it on a node

Reserve and image a node as usual (`omf load ...`), then copy the binary in
two hops — your machine to the console, console to the node.

```bash
# 1. from your laptop  (use YOUR sandbox console hostname)
scp castc yourusername@console.sb6.cosmos-lab.org:~/

# 2. console -> node   (use YOUR node name)
ssh yourusername@console.sb6.cosmos-lab.org
scp ~/castc root@node1-1-d1:/root/

# 3. on the node
ssh root@node1-1-d1
cp ~/cast/build/castc ~/cast/build/castc.old     # keep the image's compiler
mv ~/castc ~/cast/build/castc
chmod +x ~/cast/build/castc
```

**`chmod +x` is required** — zip archives do not preserve the Unix executable
bit, so without it you get "Permission denied".

Verify:

```bash
~/cast/build/castc
```

It should print a usage message. If it says `version 'GLIBC_2.38' not found`,
you downloaded a dynamically linked build — see [Troubleshooting](#8-troubleshooting).

Also get the current examples onto the node (the copy in the node image is
older):

```bash
# on the console
git clone <this-repo-url> ~/cast-new
scp ~/cast-new/examples/*.cast root@node1-1-d1:/root/cast/examples/
```

> **`omf load` wipes the node's disk.** Anything you install there is gone
> after re-imaging. Keep `castc` on the console (which persists) and repeat
> the copy + `chmod` after each load. Your source belongs in git, not on the
> node.

---

## 4. Run your first program

```bash
cd ~/cast
./simulate.sh examples/array_sum.cast
```

Expected:

```
Done init
Finished sum: c[9] = 19, total = 100
c[0] = 1
c[1] = 3
...
c[9] = 19
```

Four more worth running, because between them they exercise every major
feature:

| Command | What it demonstrates | Expect |
|---|---|---|
| `./simulate.sh examples/systolic_matmul.cast` | 2-D arrays, nested loops, systolic dataflow | `[ 14 28 42 ]` / `[ 32 64 96 ]` / `[ 50 100 150 ]` |
| `./simulate.sh examples/pipeline.cast` | machine-to-machine channels | `got 0, 2, 4, 6, 8` |
| `./simulate.sh examples/systolic_matmul_5x5_scalar.cast` | 5×5 systolic array | rows `55 110 165 220 275` … `355 710 1065 1420 1775` |
| `./simulate.sh examples/mnist_mlp_hex.cast --duration=600000` | hexagonal array, negative weights, runtime `goto` loops | 100 `image N  label L  guess G  ok` lines, then `accuracy: 98%` |

The last one is by far the biggest program in the repo — a quantised MNIST
network (95.97% on the full test set) classifying 100 test images, whose
output matches PyTorch exactly. It is 3271 lines of cast holding 4400 weights,
and runs for 52983 cycles, so give it the long `--duration` and expect castc
and iverilog to take noticeably longer than the other examples.
See [ml/README.md](ml/README.md).

Useful flags (passed through to `castc`):

```bash
./simulate.sh myprog.cast --duration=2000      # simulate longer (ns, default 500)
./simulate.sh myprog.cast --vcd=/tmp/my.vcd    # waveform for gtkwave
```

Ignore the `$fwrite ... cannot be synthesized in an always_ff process`
warnings from iverilog — `print` is simulation-only by design.

---

## 5. Writing cast programs

Minimal complete program:

```cast
machine Counter {
    interface {}                 // I/O channels (none here)
    shared {
        var uint32 n;            // registers, persist across states
    }
    states {
        count: {                 // FIRST state listed is the reset state
            print("n = ", n);
            n++;
            goto count;          // every path must end in a goto
        }
    }
}

instantiate {
    c = Counter();               // create an instance
}
```

**Mental model:** one state runs per clock cycle. Everything inside a state
body happens *in that single cycle, in parallel*. `goto` chooses the next
state.

### Arrays

```cast
shared {
    var uint32 a[10];            // 10 registers
    var uint32 m[3][3];          // 9 registers
}
...
a[0] = 5;                        // constant index — free
a[i] = i + 1;                    // loop variable — also resolved at compile time
x = a[idx];                      // register index — becomes a mux (costs logic)
```

### For loops

Loops are **fully unrolled at compile time** — they are replicated hardware,
not iteration. The bound must be a compile-time constant:

```cast
for (var int i = 0; i < 10; i++) {
    sum += a[i];                 // 10 adders, all in one cycle
}
```

A runtime bound (`i < n` where `n` is a register) is a compile error. For
runtime iteration, use a state that loops via `goto`:

```cast
drain: {
    if (idx < 10) {
        print("c = ", c[idx]);
        idx++;
        goto drain;              // one element per cycle
    } else {
        goto done;
    }
}
```

### Channels between machines

```cast
machine Producer { interface { output uint16 out; } ... out <- value; ... }
machine Consumer { interface { input  uint16 in;  }
    states {
        run: in -> x {           // body runs ONLY when data is available
            print("got ", x);
            goto run;
        }
    }
}

instantiate {
    p = Producer();
    c = Consumer();
    c.in <- p.out;               // wire them together
    c.other <- 42;               // or feed a constant (always-valid stream)
}
```

Connections are point-to-point (one producer, one consumer), buffered by a
5-deep FIFO, and consumers block until data arrives — so pipelines
self-synchronise without handshaking code.

---

## 6. Gotchas that will bite you

These are not bugs; they are how the hardware works. Each one has cost
someone an afternoon.

### Reads see start-of-cycle values

Within a state, register reads return the value from the *start* of the
cycle. Writes all commit together at the end.

```cast
compute: {
    total = a + b;
    print("total = ", total);    // prints the OLD total, not a+b!
    goto next;
}
```

Fix: print or use the value in the **next** state.

```cast
compute: { total = a + b; goto show; }
show:    { print("total = ", total); goto next; }   // correct
```

The exception is **inside a for-loop body**, where reads do see writes made
earlier in the same cycle — that is what makes `sum += a[i]` accumulate
correctly when unrolled.

### Print order within a state is unspecified

Multiple `print`s in one state all fire on the same clock edge, and Verilog
does not define their order — you may see lines shuffled or reversed.

```cast
done: {                          // rows may print in ANY order
    print("row 0: ", c[0]);
    print("row 1: ", c[1]);
}
```

Fix: one print per state, chained with `goto`. See the `row0`/`row1`/`row2`
states in [examples/systolic_matmul.cast](examples/systolic_matmul.cast).

### Loops unroll — watch the area

A loop with 25 multiplies synthesises 25 multipliers, and an unrolled
reduction becomes a deep combinational chain in one cycle. That is fine for
small arrays and is exactly what you want for a systolic PE grid, but spread
big work across states with `goto` rather than unrolling it.

### The compiler in the node image is old and has a precedence bug

If you use the `castc` that came with the node image rather than the one from
step 2, it parses `c + a * b` as `(c + a) * b` — silently wrong arithmetic.
It also has no arrays, no working loops, and no machine wiring. Use the
Actions build. (If you are ever stuck on the old one, parenthesise every
mixed-operator expression: `c + (a * b)`.)

### Editing files on the node

`vim` auto-indent mangles pasted code. Before pasting:

```
:set paste
```
then `i`, paste, `Esc`, `:wq`. Or write files locally and `scp` them.

---

## 7. Trying things without a node

`tools/cast_sim` is a behavioural simulator that needs **only a C++
compiler** — no LLVM, no CIRCT, no node. It reuses the real parser and models
the same semantics, so it predicts a program's `print` output.

```bash
c++ -std=c++20 -O2 -I src/Frontend \
    tools/cast_sim.cpp src/Frontend/CastParser.cpp -o tools/cast_sim

tools/cast_sim examples/array_sum.cast
tools/cast_sim --trace examples/array_sum.cast     # cycle-by-cycle states
```

Good for checking logic while writing. It is **not** the real compiler: it
does not emit Verilog, and it will not reproduce quirks of an older prebuilt
`castc`. Confirm anything important on a node.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Permission denied` running castc | Missing `chmod +x` (zip drops the executable bit) |
| `version 'GLIBC_2.38' not found` | Dynamically linked build; re-run the workflow and confirm the run's "Report binary portability" step says `link mode: static` |
| `cannot execute binary file` | Wrong architecture — that artifact is Linux x86-64, it will not run on macOS |
| Program prints nothing | A state is blocked on a channel receive that never gets data — check the `instantiate` wiring |
| Prints appear out of order | Multiple prints in one state; give each its own state |
| A value is one cycle "behind" | Reading in the same state it was written; use the next state |
| `for loop bound must be a compile-time constant` | Runtime bound — use the `goto` loop pattern instead |
| Simulation ends too early | `./simulate.sh prog.cast --duration=5000` |
| Compile error mentioning something you did not write | Check for `uint 16` (space) and similar typos — `uint16` is one token |
| `scp` created a local file named `user@host` | Missing the `:` — it is `scp file user@host:~/` |

---

## 9. Rebuilding the compiler

Only needed if you change the compiler itself (`src/Frontend/*`), not for
writing `.cast` programs.

**Easiest:** push your change; the workflow rebuilds automatically and a new
`castc-linux-x64` artifact appears under Actions. Editing
`.github/workflows/*` requires a GitHub token with the `workflow` scope.

**On a node** (only if it already has LLVM/MLIR/CIRCT installed):

```bash
cd ~/cast && mkdir -p build && cd build
MLIR=$(dirname "$(find / -name MLIRConfig.cmake 2>/dev/null | head -1)")
CIRCT=$(dirname "$(find / -name CIRCTConfig.cmake 2>/dev/null | head -1)")
cmake .. -G Ninja -DCMAKE_PREFIX_PATH="${MLIR%/lib/cmake/mlir};${CIRCT%/lib/cmake/circt}"
ninja
```

Run the test suite with `ninja -C build check` (compiles every example,
elaborates the Verilog, and diffs simulation output against
`tests/expected/`).
