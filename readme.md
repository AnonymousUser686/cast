# cast

cast is a language for describing hardware state machines. A cast program defines
one or more *machines* — finite state automata with typed registers, channel-based
I/O, and explicit control flow — and compiles them to synthesisable SystemVerilog via
MLIR and the CIRCT compiler infrastructure.

```
cast source (.cast)
       │
       ▼
  Handwritten parser
       │  AST
       ▼
  MLIR/CIRCT lowering
  (ESI channels → HW/Seq/SV dialects)
       │  MLIR module
       ▼
  exportVerilog
       │  SystemVerilog (.sv)
       ▼
  iverilog / synthesis tool
```

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Building](#building)
3. [Quick start](#quick-start)
4. [Compiler flags](#compiler-flags)
5. [simulate.sh](#simulatesh)
6. [Language reference](#language-reference)
7. [Examples](#examples)
8. [Testing](#testing)
9. [run_cast_test.sh](#run_cast_testsh)
10. [How it works](#how-it-works)

---

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| clang++ (≥ 17) | C++23 compiler | `brew install llvm` |
| CMake ≥ 3.14, Ninja | Build system | `brew install cmake ninja` |
| LLVM + MLIR | IR infrastructure | build from source (see below) |
| CIRCT | HW/Seq/SV dialects | build from source (see below) |
| iverilog | Simulation | `brew install icarus-verilog` |
| gtkwave *(optional)* | Waveform viewer | `brew install gtkwave` |

### Building LLVM + MLIR

```bash
git clone https://github.com/llvm/llvm-project
cd llvm-project
git checkout llvmorg-22.1.1
mkdir build && cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_CCACHE_BUILD=ON
ninja
```

### Building CIRCT

```bash
git clone https://github.com/llvm/circt
cd circt
git submodule init && git submodule update
git checkout 7f76cbc26
mkdir build && cd build
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON
ninja
```

---

## Building

```bash
git clone https://github.com/jelvani/cast cast && cd cast
mkdir build && cd build
cmake .. -G Ninja \
  -DCMAKE_PREFIX_PATH="/path/to/llvm-project/build;/path/to/circt/build"
ninja
```

The build produces a single binary: `build/castc`.

### Prebuilt binary via GitHub Actions (no LLVM build needed)

The `build castc` workflow (`.github/workflows/build-castc.yml`) builds a
Linux x86_64 `castc` against the prebuilt `circt-full-static` release package
instead of a from-source LLVM/CIRCT build, then runs the full example suite
with iverilog. Trigger it from the Actions tab and download the
`castc-linux-x64` artifact from the run page; place the binary at
`<repo>/build/castc` on the target machine and `simulate.sh` works as-is.
Useful when the target machine (or your workstation) cannot host the ~50 GB
LLVM + CIRCT source build.

---

## Quick start

Compile and simulate any `.cast` file in one command:

```bash
./simulate.sh examples/counter.cast
```

Expected output:

```
==> compiling examples/counter.cast
    verilog written to /tmp/counter.sv
==> elaborating
==> simulating
count: 0
count: 1
count: 2
...
count: 9
--- wrap: resetting to 0 ---
count: 0
...
```

---

## Compiler flags

```
castc [options] <file.cast>
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tb` | off | Append an embedded `module tb` testbench to the output |
| `--duration=<ns>` | `500` | Simulation run length in nanoseconds (used by `--tb`) |
| `--vcd=<path>` | `/tmp/<basename>.vcd` | VCD waveform dump path (used by `--tb`) |

Without `--tb` the output is pure synthesisable SystemVerilog with no simulation
constructs.

### Manual compilation steps

```bash
# 1. Compile to SystemVerilog (with embedded testbench)
./build/castc --tb examples/counter.cast > /tmp/counter.sv

# 2. Elaborate
iverilog -g2012 -gno-assertions -o /tmp/counter_sim /tmp/counter.sv

# 3. Simulate
vvp /tmp/counter_sim
```

### Viewing waveforms

```bash
./build/castc --tb --vcd=/tmp/counter.vcd examples/counter.cast > /tmp/counter.sv
iverilog -g2012 -gno-assertions -o /tmp/counter_sim /tmp/counter.sv
vvp /tmp/counter_sim
gtkwave /tmp/counter.vcd
```

---

## simulate.sh

The helper script wraps the three manual steps above:

```
./simulate.sh <file.cast> [--duration=<ns>] [--vcd=<path>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--duration=<ns>` | `500` | How long the simulation runs |
| `--vcd=<path>` | `/tmp/<basename>.vcd` | Where the VCD waveform is written |

Options after the file are forwarded directly to `castc`.

---

## Language reference

### Program structure

A cast program contains one or more `machine` definitions followed by a single
`instantiate` block.

```cast
machine <Name> {
    interface { ... }   // I/O channels
    shared    { ... }   // persistent registers
    states    { ... }   // state definitions
}

instantiate {
    x = <Name>();       // create an instance
    x.port <- value;    // feed a channel
}
```

### Machines

A machine is a named state automaton. Each machine compiles to a hardware module
with `clk` and `rst` ports plus one port per declared channel.

### interface

Declares the machine's I/O channels. Every channel has a direction, a type, and a
name.

```cast
interface {
    input  uint16 data_in;
    output bool   ready;
}
```

| Direction | Meaning |
|-----------|---------|
| `input`   | The machine receives values on this channel |
| `output`  | The machine sends values on this channel |

### shared

Declares registers that persist across states. These become synchronous registers
(`seq.compreg`) in the generated Verilog. A constant initializer becomes the
register's reset value (registers without one reset to 0).

```cast
shared {
    var uint32 counter;
    var uint32 threshold = 42;
    var bool   flag;
    var uint32 window[8];
}
```

### states

A states block contains one or more named states. The **first declared state** is
always the reset/start state.

```cast
states {
    idle: channel_name -> local_var {   // header: block on channel receive
        // body: runs once per received value
        goto next_state;
    }
    next_state: {
        // body: runs every cycle
        goto idle;
    }
}
```

#### State headers (channel receive)

A state can begin with one or more channel-receive bindings before the body block:

```
state_name: channel -> local_name { ... }
state_name: ch1 -> a, ch2 -> b { ... }
```

The body only executes in cycles where all listed channels have valid data.
`local_name` is a combinational alias for the received value — it is visible to
all statements in the body including `print`.

#### goto

Every execution path through a state body must end with a `goto`:

```cast
goto other_state;
```

### Types

| cast type | Width | Notes |
|-----------|-------|-------|
| `bool`    | 1-bit | |
| `byte`, `uint8`, `int8` | 8-bit | |
| `uint16`, `int16` | 16-bit | |
| `uint32`, `int32`, `int` | 32-bit | |
| `uint64`, `int64` | 64-bit | |

Arithmetic and comparisons are currently unsigned regardless of the declared
signedness.

### Arrays

Arrays are declared with a C-style suffix or a Go-style prefix — both are
equivalent:

```cast
shared {
    var uint32 a[10];        // C-style
    var [10]uint32 b;        // Go-style
    var uint32 m[3][3];      // multi-dimensional
}
```

An `[N]T` array lowers to N element registers (a register file). Indexing
works with any integer expression:

- **Constant indices** (literals, enum members, or for-loop variables) select
  the element register directly at compile time and cost no extra hardware.
  Out-of-range constant indices are a compile error.
- **Dynamic indices** (expressions involving runtime registers or channel
  values) lower to a mux tree on reads and per-element write-enables on
  writes. An out-of-range dynamic index reads 0 and writes nothing.

```cast
c[0] = 5;                 // constant index
c[i] = a[i] + b[i];       // i is a loop variable: resolved at unroll time
print("val = ", c[idx]);  // idx is a register: mux over all elements
arr[i] <- in_ch;          // channel receive into an element (state header)
```

Array initializers (`= {...}`) are not supported — assign elements in a
state. Arrays cannot be channel (interface) types.

### Operators

| Category | Operators |
|----------|-----------|
| Arithmetic | `+` `-` `*` `/` `%` (and unary `-`) |
| Bitwise | `&` `\|` `^` `<<` `>>` |
| Comparison | `==` `!=` `<` `<=` `>` `>=` |
| Logical | `&&` `\|\|` `!` |
| Compound assign | `+=` `-=` `*=` `/=` `^=` `<<=` `>>=` |
| Increment/decrement | `++` `--` |

### Channel operations

| Syntax | Meaning |
|--------|---------|
| `ch -> v` | Receive from channel `ch` into local binding `v` (state header) |
| `ch <- expr` | Send `expr` on output channel `ch` |
| `inst.port <- expr` | Feed a constant into instance `inst`'s input channel (always-valid stream) |
| `b.in <- a.out` | Connect instance `a`'s output channel to instance `b`'s input channel |

### Control flow

```cast
if expr { ... }
if expr { ... } else { ... }
if expr { ... } else if expr { ... } else { ... }

for (var uint16 i = 0; i < 10; i++) { ... }

goto state_name;
```

#### for loops

A `for` loop is **fully unrolled at compile time** into the single cycle of
the state that contains it — it describes replicated hardware, not sequential
iteration. Consequences:

- The bound must be a compile-time constant: a literal, an expression of
  literals, or an enclosing loop's variable (`for (var int j = i; ...)` nests
  fine). A runtime bound is a compile error — for runtime iteration, make a
  state that increments a shared register and re-enters itself with `goto`
  (see [array_sum.cast](examples/array_sum.cast)).
- The init may declare a fresh variable (`var uint16 i = 0`) or assign an
  existing shared one (`i = 0`; the register receives the final value when
  the loop ends). Updates may be `i++`, `i--`, or any compound assignment
  with a constant step (`i += 2`, `i >>= 1`, ...).
- The loop variable is a compile-time constant in the body: using it in an
  array index selects elements at zero hardware cost.

**Read-after-write semantics.** Outside loops, register reads return the
pre-update value — all updates in a state commit together at the end of the
cycle (this is what makes `b = a + b; a = b;` in fibonacci work). *Inside a
for-loop body*, reads see values written earlier in the same cycle, in
program order. This makes unrolled accumulations work as software intuition
expects:

```cast
sum = 0;
for (var int i = 0; i < 10; i++) {
    sum += data[i];        // single-cycle reduction tree over all 10 elements
}
```

The flip side: a shift written low-to-high (`for i: p[i+1] = p[i]`) would
propagate one value everywhere, exactly as it would in software. Iterate
high-to-low to shift, as [systolic_matmul.cast](examples/systolic_matmul.cast)
does.

### print

`print` lowers to `$display` in simulation (no-op in synthesis):

```cast
print("label: ", value);
print("a=", a, " b=", b);
```

### instantiate

The `instantiate` block creates machine instances, feeds their input channels,
and wires machines together.

```cast
instantiate {
    m = MyMachine();
    m.data <- 42;          // constant feed: an always-valid stream of 42s

    a = Producer();
    b = Consumer();
    b.in <- a.out;         // machine-to-machine connection
}
```

**Constant feeds** (`m.port <- 42`) drive the input with an always-valid
stream of that value — the machine sees it every cycle it asks.

**Machine-to-machine connections** (`b.in <- a.out`) wire one instance's
output channel into another's input channel:

- Channels are **point-to-point**: each output may feed exactly one input,
  and each input may have exactly one source (a constant feed or one
  connection, not both). Violations are compile errors.
- Both ports must be declared with the **same type**.
- Every connection is decoupled by a small FIFO, so a value sent on cycle T
  is receivable at cycle T+1. The consumer's header receive blocks until
  data arrives, so pipelines self-synchronise.
- Sends are fire-and-forget: if a connection's FIFO is full (consumer
  stalled for >5 cycles while the producer keeps sending), further sends
  are dropped. Design consumers to keep up, or throttle producers.
- Declare both instances before connecting them; feedback loops
  (`a.in <- b.out; b.in <- a.out;`) are allowed.

See [pipeline.cast](examples/pipeline.cast) for a three-stage
producer → doubler → printer chain.

---

## Examples

| File | Level | Description |
|------|-------|-------------|
| [counter.cast](examples/counter.cast) | Beginner | Counts 0–9 and wraps — the simplest self-contained state machine |
| [fibonacci.cast](examples/fibonacci.cast) | Beginner | Generates the Fibonacci sequence using two shared registers |
| [gcd.cast](examples/gcd.cast) | Beginner | GCD(48, 18) via repeated subtraction (Euclidean algorithm) |
| [alarm.cast](examples/alarm.cast) | Intermediate | Threshold alarm with edge detection — prints only on level change |
| [running_average.cast](examples/running_average.cast) | Intermediate | Accumulates 8 samples and prints their integer average |
| [traffic_light.cast](examples/traffic_light.cast) | Intermediate | Red/green/yellow cycle with per-phase countdown timers |
| [debounce.cast](examples/debounce.cast) | Intermediate | Counter-based digital debouncer — requires 16 stable cycles to latch |
| [checksum.cast](examples/checksum.cast) | Intermediate | Rolling XOR checksum over a 5-byte window |
| [pwm.cast](examples/pwm.cast) | Intermediate | Pulse-width modulator — duty cycle set via input channel |
| [shift_register.cast](examples/shift_register.cast) | Advanced | Serial-in parallel-out: assembles 8 bits into a byte |
| [binary_search.cast](examples/binary_search.cast) | Advanced | Binary search on a [0, 127] range with multi-cycle halving |
| [popcount.cast](examples/popcount.cast) | Advanced | Counts set bits in a 16-bit word (LSB-first iteration) |
| [uart_tx.cast](examples/uart_tx.cast) | Advanced | 8N1 UART transmitter — serialises a byte with start/stop framing |
| [test_loop.cast](examples/test_loop.cast) | Beginner | Minimal for-loop unrolling demo with per-iteration prints |
| [array_sum.cast](examples/array_sum.cast) | Intermediate | Arrays + loops: unrolled element-wise sum, single-cycle reduction, and a goto-based runtime drain loop |
| [matrix_multiply.cast](examples/matrix_multiply.cast) | Intermediate | 2x2 matrix multiply with scalar registers and explicit dot products |
| [matrix_multiply_loops.cast](examples/matrix_multiply_loops.cast) | Intermediate | 2x2 matrix multiply routed through unrolled loops |
| [systolic_matmul.cast](examples/systolic_matmul.cast) | Advanced | 3x3 output-stationary systolic array: 9 parallel MAC PEs with skewed operand injection |

Run any example:

```bash
./simulate.sh examples/fibonacci.cast
./simulate.sh examples/uart_tx.cast
```

---

## Testing

`ninja check` compiles every listed example through up to three stages and reports results:

```bash
ninja -C build check
```

| Stage | Tool | What it checks |
|-------|------|----------------|
| 1 | `castc --tb --duration=100` | cast source compiles without error |
| 2 | `iverilog -g2012 -gno-assertions` | generated Verilog is valid and elaborates cleanly |
| 3 | `vvp` + `diff` | simulation output matches `tests/expected/<name>.expected` |

Stage 3 only runs when a corresponding expected file exists. All 14 built-in examples
include one.

Output on a passing run:

```
Test project /path/to/cast/build
      Start  1: example.simple
 1/14 Test  #1: example.simple ...................   Passed
      Start  2: example.counter
 2/14 Test  #2: example.counter ..................   Passed
...
14/14 Test #14: example.uart_tx ..................   Passed

100% tests passed, 0 tests failed out of 14
```

On failure, the stderr from the failing stage is shown. For a Stage 3 failure the
diff is printed so it is immediately clear which lines changed:

```
  [3/3] simulate + verify: gcd.expected
--- expected vs actual (unified diff) ---
--- tests/expected/gcd.expected
+++ /tmp/cast_check_gcd_actual
@@ -6 +6 @@
-gcd(48, 18) = 6
+gcd(48, 18) = 7
FAIL [gcd]: simulation output does not match expected
```

To run a single test:

```bash
ctest --test-dir build -R example.fibonacci --output-on-failure
```

### Expected output files

The files in `tests/expected/` are the correctness specification for each example.
Each file contains exactly the `print` output the simulation should produce over a
100 ns run — VCD banner lines and the `$finish` timing line are stripped before
comparison. Edit these files when a behavioural change is intentional.

### Adding a new example to the test suite

1. Place the `.cast` file in `examples/`.
2. Add one line to `CMakeLists.txt`:

```cmake
add_cast_test(my_new_example)
```

3. *(Optional but recommended)* Create `tests/expected/my_new_example.expected`
   containing the expected `print` output for a 100 ns simulation. With this file
   present, Stage 3 runs automatically and `ninja check` enforces correctness.

---

## run_cast_test.sh

`tests/run_cast_test.sh` is the script CTest calls for every test, but it can also
be run directly to debug a single example without going through `ninja check`.

```
bash tests/run_cast_test.sh <castc> <file.cast> [<expected_file>]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `<castc>` | yes | Path to the compiled `castc` binary |
| `<file.cast>` | yes | The cast source file to test |
| `<expected_file>` | no | Path to an expected output file — enables Stage 3 |

**Compile and elaborate only (Stages 1–2):**

```bash
bash tests/run_cast_test.sh build/castc examples/counter.cast
```

**Compile, elaborate, and verify correctness (Stages 1–3):**

```bash
bash tests/run_cast_test.sh build/castc examples/gcd.cast tests/expected/gcd.expected
```

**Check a file that has no expected output yet (generate one):**

```bash
# Run the three stages manually to see what the simulation produces
build/castc --tb --duration=100 examples/my_new.cast > /tmp/my_new.sv
iverilog -g2012 -gno-assertions -o /tmp/my_new_sim /tmp/my_new.sv
vvp /tmp/my_new_sim | grep -v '^VCD info:' | grep -v '\$finish called'
# Review the output, then save it as the expected file
vvp /tmp/my_new_sim | grep -v '^VCD info:' | grep -v '\$finish called' \
    > tests/expected/my_new.expected
```

Intermediate files (`/tmp/cast_check_<name>.sv`, `/tmp/cast_check_<name>_sim`,
`/tmp/cast_check_<name>_actual`) are left in `/tmp` after a run, so you can inspect
the generated Verilog or replay the simulation without recompiling.

---

## How it works

### Frontend

The handwritten lexer and parser ([CastParser.cpp](src/Frontend/CastParser.cpp)) parse Cast source into an AST defined in [CastAST.hpp](src/Frontend/CastAST.hpp). The visitor in [castLower.cpp](src/Frontend/castLower.cpp) walks this AST and emits MLIR operations directly.

### MLIR/CIRCT lowering pipeline

Each machine becomes an `esi.pure_module` containing:

- **ESI channels** (`esi.channel<T>`) for every declared interface port
- **`seq.compreg`** for every `shared` variable and FIFO stub register
- **`hw.module`** with `clk` and `rst` ports generated by the ESI lowering passes

The passes run in this order:

```
ESIPhysical → ESIBundle → ESIPort → ESIType →
ESItoHW → SeqFIFO → SeqHLMem → VerifToSV →
LowerSeqToSV → exportVerilog
```

### Testbench generation

When `--tb` is passed, `castc` appends a `module tb` after `exportVerilog` output.
The testbench instantiates the generated `Main` module, drives `clk` and `rst`,
opens a VCD dump, and calls `$finish` after `--duration` nanoseconds.
