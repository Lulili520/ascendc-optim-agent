set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="DiagPart"

CASE_NAMES=("case0" "case1")
CASE_DIMS=(128 512)
CASE_LABELS=("[128,128]" "[512,512]")

die() { echo "ERROR: $*" >&2; exit 1; }

IDX=""
for arg in "$@"; do
    case "$arg" in
        0|1) IDX="$arg" ;;
    esac
done

echo "=== [1/5] Set CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"

echo "=== [2/5] Build ==="
rm -rf build
mkdir -p build && cd build
cmake .. || die "cmake failed"
make -j$(nproc) || die "make failed"
cd ..

run_test() {
    local idx=$1
    local N=${CASE_DIMS[$idx]}
    local label="${CASE_LABELS[$idx]}"

    echo ""
    echo "=== Testing Shape: ${label} ==="
    cd build

    echo "  Generating data N=${N}..."
    python3 ../scripts/gen_data.py "${N}" || die "gen_data.py N=${N} failed"

    echo "  Running Kernel N=${N}..."
    rm -f output/output.bin
    "./${OP_NAME}" "${N}" || die "Kernel run failed N=${N}"
    [ -f output/output.bin ] || die "output.bin not found"

    echo "  Precision verification N=${N}..."
    python3 ../scripts/verify_result.py output/output.bin output/golden.bin \
        || die "Precision verification failed N=${N}"
    echo "  [PASS] ${label}"

    cd ..
}

if [ -n "$IDX" ]; then
    run_test "$IDX"
else
    echo "=== [3/5] Testing Shape [128, 128] ==="
    run_test 0

    echo ""
    echo "=== [4/5] Testing Shape [512, 512] ==="
    run_test 1
fi

echo ""
echo "=== [5/5] All tests passed ==="
exit 0
