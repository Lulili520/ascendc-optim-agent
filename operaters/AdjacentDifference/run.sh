#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="AdjacentDifference"

NUM_CASES=2
CASE_NAMES=("case1" "case2")
CASE_DIMS=(2097152 4194304)
CASE_LABELS=("[8, 1024, 256]" "[4, 2048, 512]")

CASE_IDX=${1:-}

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== Build ==="
rm -rf build
mkdir -p build
cd build
cmake .. || die "cmake failed"
make -j4 || die "make failed"
cd ..

run_case() {
    local i=$1
    local name="${CASE_NAMES[$i]}"
    local dims="${CASE_DIMS[$i]}"
    local label="${CASE_LABELS[$i]}"
    local case_num=$((i + 1))

    echo ""
    echo "=== Running case $i: $name shape=$label dims=$dims ==="
    cd build

    python3 ../scripts/gen_data.py "${case_num}" || die "gen_data $name failed"
    rm -f output/output.bin
    "./${OP_NAME}" "${dims}" || die "Kernel $name failed"
    [ -f output/output.bin ] || die "output.bin not generated ($name)"

    echo "=== Precision verification ($name $label) ==="
    python3 ../scripts/verify_result.py output/output.bin "output/golden_${name}.bin" \
        || die "Precision verification failed ($name)"
    echo "  [PASS] $name $label"

    cd ..
}

if [ -n "${CASE_IDX}" ]; then
    run_case "${CASE_IDX}"
else
    for i in $(seq 0 $((NUM_CASES - 1))); do
        run_case "$i"
    done
fi

echo ""
echo "=== Done ==="
exit 0
