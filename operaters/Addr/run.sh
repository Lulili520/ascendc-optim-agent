set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="Addr"

CASE_NAMES=("case1" "case2")
CASE_M=(512 1024)
CASE_N=(1024 2048)
CASE_LABELS=("[512, 1024]" "[1024, 2048]")

NUM_CASES=${#CASE_NAMES[@]}

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/3] Set CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"

echo "=== [2/3] Build ==="
rm -rf build
mkdir -p build
cd build
cmake .. || die "cmake failed"
make -j4  || die "make failed"
cd ..

run_case() {
    local i=$1
    local name="${CASE_NAMES[$i]}"
    local label="${CASE_LABELS[$i]}"
    local dim_m="${CASE_M[$i]}"
    local dim_n="${CASE_N[$i]}"

    echo ""
    echo "=== [3/3] Run Kernel (${name} ${label}) ==="
    cd build

    # Generate test data (generates all cases at once)
    if [ "$i" -eq 0 ] || [ ! -f "output/golden_${name}.bin" ]; then
        python3 ../scripts/gen_data.py || die "gen_data.py failed"
    fi

    ln -sf "input_x1_${name}.bin" input/input_x1.bin
    ln -sf "input_x2_${name}.bin" input/input_x2.bin
    ln -sf "input_x3_${name}.bin" input/input_x3.bin
    ln -sf "input_alpha_${name}.bin" input/input_alpha.bin
    ln -sf "input_beta_${name}.bin" input/input_beta.bin
    ln -sf "golden_${name}.bin" output/golden.bin
    rm -f output/output.bin
    "./${OP_NAME}" "${dim_m}" "${dim_n}" || die "Kernel run failed for ${name}"
    [ -f output/output.bin ] || die "output.bin not generated"

    echo "=== Precision verification (${name} ${label}) ==="
    python3 ../scripts/verify_result.py output/output.bin "output/golden_${name}.bin" \
        || { echo "  [FAIL] ${name} ${label}"; cd ..; return 1; }
    echo "  [PASS] ${name} ${label}"

    cd ..
}

echo "=== Running all ${NUM_CASES} cases ==="
PASS_COUNT=0
FAIL_COUNT=0
FAILED_CASES=()

for i in $(seq 0 $((NUM_CASES - 1))); do
    if run_case "$i"; then
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_CASES+=("${CASE_NAMES[$i]} ${CASE_LABELS[$i]}")
    fi
done

echo ""
echo "========================================="
echo "  Results: ${PASS_COUNT}/${NUM_CASES} passed"
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "  Failed cases:"
    for fc in "${FAILED_CASES[@]}"; do
        echo "    - ${fc}"
    done
    echo "========================================="
    exit 1
fi
echo "  All cases PASSED!"
echo "========================================="

echo ""
echo "=== Done ==="
exit 0
