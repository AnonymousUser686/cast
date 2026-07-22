# tools/

## cast_sim — behavioral simulator

`cast_sim` is a small, dependency-free simulator for cast programs. It is **not**
the real compiler: it reuses the real cast parser (`src/Frontend/CastParser.cpp`)
and then interprets the AST using the same semantics the MLIR lowering in
`castLower.cpp` implements — one active state per cycle, registers that commit at
the cycle boundary, compile-time-unrolled `for` loops, per-element array storage,
and the "reads inside a loop see same-cycle writes" rule.

Use it to sanity-check array / for-loop behaviour and predict a program's `print`
output on a machine that does **not** have the LLVM + CIRCT + iverilog toolchain
installed. The real `castc` build remains the source of truth for synthesizable
Verilog; if the two ever disagree, that is a bug worth chasing.

It does not model interface channels, `instantiate` wiring, enums, `switch`, or
exceptions — programs using those must be checked with the real toolchain.

### Build

Only a C++20 compiler is needed (no LLVM/CIRCT):

```bash
c++ -std=c++20 -O2 -I src/Frontend \
    tools/cast_sim.cpp src/Frontend/CastParser.cpp -o tools/cast_sim
```

### Run

```bash
tools/cast_sim examples/array_sum.cast
tools/cast_sim examples/systolic_matmul.cast
tools/cast_sim --trace examples/array_sum.cast     # cycle-by-cycle state trace
tools/cast_sim --max-cycles=500 examples/foo.cast  # raise the cycle cap
```
