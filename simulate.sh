#!/usr/bin/env bash
# simulate.sh — compile a .cast file and simulate it with iverilog
# Usage:
#   ./simulate.sh <file.cast> [--duration=<ns>] [--vcd=<path>]
#
# A .cast file may declare the flags it needs with a comment like
#
#     // simulate: --duration=600000
#
# so that running it takes no special knowledge. Flags given on the command
# line override the ones in the file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASTC="$SCRIPT_DIR/build/castc"

if [[ $# -lt 1 || "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 <file.cast> [--duration=<ns>] [--vcd=<path>]"
    exit 0
fi

CAST_FILE="$1"; shift
USER_ARGS=("$@")

if [[ ! -f "$CAST_FILE" ]]; then
    echo "error: file not found: $CAST_FILE"
    exit 1
fi
if [[ ! -x "$CASTC" ]]; then
    echo "error: castc not found at $CASTC — run 'ninja -C build' first"
    exit 1
fi

BASE="$(basename "$CAST_FILE" .cast)"
SV_OUT="/tmp/${BASE}.sv"
SIM_OUT="/tmp/${BASE}_sim"
ERR_OUT="$(mktemp)"
trap 'rm -f "$ERR_OUT"' EXIT

# ── flags declared by the program ─────────────────────────────────────────────
# Anything after a `// simulate:` comment is treated as default flags. A flag
# the user passed explicitly wins, so --duration on the command line still
# overrides the file's own value.
FILE_ARGS=()
while read -r flag; do
    [[ -n "$flag" ]] && FILE_ARGS+=("$flag")
done < <(sed -n 's|^[[:space:]]*//[[:space:]]*simulate:[[:space:]]*||p' "$CAST_FILE" | tr ' ' '\n')

ARGS=()
for fa in ${FILE_ARGS[@]+"${FILE_ARGS[@]}"}; do
    overridden=0
    for ua in ${USER_ARGS[@]+"${USER_ARGS[@]}"}; do
        [[ "${ua%%=*}" == "${fa%%=*}" ]] && overridden=1
    done
    if (( overridden )); then
        echo "    ${fa%%=*} from $CAST_FILE overridden on the command line"
    else
        ARGS+=("$fa")
        echo "    using $fa (declared by $CAST_FILE)"
    fi
done
ARGS+=(${USER_ARGS[@]+"${USER_ARGS[@]}"})

# ── compile ───────────────────────────────────────────────────────────────────
echo "==> compiling $CAST_FILE"
if ! "$CASTC" --tb ${ARGS[@]+"${ARGS[@]}"} "$CAST_FILE" > "$SV_OUT" 2> "$ERR_OUT"; then
    cat "$ERR_OUT" >&2
    # The compiler shipped in the COSMOS node image predates arrays, loops and
    # machine wiring. It is the single most common cause of a failed compile,
    # and its error messages do not say so, so say it here.
    if grep -q "'array/slice indexing' is not yet supported" "$ERR_OUT"; then
        cat >&2 <<'EOF'

────────────────────────────────────────────────────────────────────────────
This is the OLD castc from the COSMOS node image, not a current build.
It has no support for arrays, working loops, or machine wiring, and it also
parses `c + a * b` as `(c + a) * b`.

Fix: install the statically linked binary from the repo's GitHub Actions
("build castc" -> Artifacts -> castc-linux-x64), then

    cp ~/cast/build/castc ~/cast/build/castc.old
    mv ~/castc ~/cast/build/castc
    chmod +x ~/cast/build/castc

Note that `omf load` wipes the node's disk, so this has to be redone after
every re-image. The examples in the node image are stale for the same reason
— copy the current ones over too. See GETTING_STARTED.md section 3.
────────────────────────────────────────────────────────────────────────────
EOF
    fi
    exit 1
fi
cat "$ERR_OUT" >&2
echo "    verilog written to $SV_OUT"

# ── elaborate ─────────────────────────────────────────────────────────────────
echo "==> elaborating"
iverilog -g2012 -gno-assertions -o "$SIM_OUT" "$SV_OUT"

# ── simulate ──────────────────────────────────────────────────────────────────
echo "==> simulating"
vvp "$SIM_OUT"
