#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] 设置 CANN 环境 ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH 未设置"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh 失败"

echo "=== [2/4] 编译 ==="
rm -rf build
mkdir -p build
cd build
cmake .. || die "cmake 失败"
make -j4  || die "make 失败"
cd ..

CASE_NAMES=("case0" "case1")
CASE_DIMS=($((8*1024*256)) $((4*2048*512)))
CASE_LABELS=("[8, 1024, 256]" "[4, 2048, 512]")
CASE_ARGS=("8 1024 256 1 0 0" "4 2048 512 1 0 0")

NUM_CASES=${#CASE_NAMES[@]}

# Support index execution: bash run.sh <index>
if [ $# -gt 0 ]; then
    START_IDX=$1
    END_IDX=$1
else
    START_IDX=0
    END_IDX=$((NUM_CASES - 1))
fi

for i in $(seq $START_IDX $END_IDX); do
    echo ""
    echo "=== 测试用例 $((i+1))/${NUM_CASES}: ${CASE_NAMES[$i]} ${CASE_LABELS[$i]} ==="
    cd build
    export TEST_CASE=$i
    python3 ../scripts/gen_data.py || die "gen_data.py 失败 (case $i)"

    rm -f output/output_y.bin
    ./Cumsum ${CASE_ARGS[$i]} || die "Kernel 运行失败 (case $i)"
    [ -f output/output_y.bin ] || die "output_y.bin 不存在"

    python3 ../scripts/verify_result.py \
        output/output_y.bin output/golden_y.bin \
        || die "精度验证失败 (case $i)"
    cd ..
done

echo ""
echo "=== 全部测试通过 ==="
