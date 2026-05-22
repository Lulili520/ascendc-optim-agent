#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="Complex"

CASE_NAMES=("case0")
CASE_DIMS=(16777216)
CASE_LABELS=("[4,2048,2048]")

die() { echo "[ERROR] $*" >&2; exit 1; }

IDX=""
for arg in "$@"; do
    case "$arg" in
        0) IDX="$arg" ;;
    esac
done

echo "=== [1/5] Set CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"

echo "=== [2/5] Build ==="
rm -rf build
mkdir -p build
cd build
cmake .. || die "cmake failed"
make -j$(nproc) || die "make failed"
cd ..

run_test() {
    local idx=$1
    local total_length=${CASE_DIMS[$idx]}
    local label="${CASE_LABELS[$idx]}"

    echo ""
    echo "=== Testing Shape: ${label} (${total_length} elements) ==="
    cd build

    echo "  Generating test data..."
    python3 ../scripts/gen_data.py "${total_length}" || die "gen_data.py failed"

    echo "  Computing Golden..."
    python3 -c "
import numpy as np
import sys
sys.path.insert(0, '../scripts')
from golden import compute_golden
real = np.fromfile('input/input_real.bin', dtype=np.float16)
imag = np.fromfile('input/input_imag.bin', dtype=np.float16)
golden = compute_golden(real, imag)
golden.tofile('output/golden.bin')
print(f'Golden: {golden.shape}, {golden.dtype}')
" || die "golden failed"

    echo "  Running Kernel..."
    rm -f output/output.bin
    "./${OP_NAME}" "${total_length}" || die "Kernel run failed"
    [ -f output/output.bin ] || die "output.bin not found"

    echo "  Precision verification..."
    python3 ../scripts/verify_result.py output/output.bin output/golden.bin \
        || die "Precision verification failed: ${label}"
    echo "  [PASS] ${label}"

    cd ..
}

if [ -n "$IDX" ]; then
    run_test "$IDX"
else
    echo "=== [3/5] Testing Shape [4, 2048, 2048] ==="
    run_test 0
fi

echo ""
echo "=== [5/5] All tests passed ==="
exit 0
