#!/usr/bin/env python3
"""NPU 算子精度验证脚本。

使用方法：
    python3 verify_result.py

配置方式：修改下方 OUTPUTS 列表，每个元素为四元组：
    (npu_path, golden_path, dtype, category)
    - npu_path:   NPU 输出文件路径
    - golden_path: CPU golden 文件路径
    - dtype:       numpy dtype（np.float16 / np.float32 / np.int32 / ...）
    - category:    "float" | "integer" | "bitwise"
"""
import numpy as np
import sys

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 配置区：按算子输出修改此列表
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUTS = [
    # (npu_path,                    golden_path,                   dtype,        category)
    # 示例（按实际算子修改）：
    # ("output/output_y.bin",       "output/golden_y.bin",        np.float16,   "float"),
    # ("output/output_idx.bin",     "output/golden_idx.bin",      np.int32,     "integer"),
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 验证逻辑（无需修改）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FLOAT_THRESHOLDS = {
    np.dtype(np.float16): 2 ** (-10),
    np.dtype(np.float32): 2 ** (-13),
    np.dtype(np.float64): 2 ** (-13),
}


def check_float(npu, golden, label):
    threshold = FLOAT_THRESHOLDS.get(npu.dtype, 2 ** (-10))
    mere = float(np.mean(np.abs(npu - golden) / (np.abs(golden) + 1e-7)))
    mare = float(np.max(np.abs(npu - golden) / (np.abs(golden) + 1e-7)))
    is_pass = (mere < threshold) and (mare < 10 * threshold)
    status = "PASS" if is_pass else "FAIL"
    print(f"  {status} {label} (float, {npu.dtype}): "
          f"MERE={mere:.6e}, MARE={mare:.6e}, "
          f"threshold={threshold:.6e}, mare_limit={10 * threshold:.6e}")
    if not is_pass:
        _show_mismatch_samples(npu, golden, label)
    return is_pass


def check_integer(npu, golden, label):
    is_pass = np.array_equal(npu, golden)
    status = "PASS" if is_pass else "FAIL"
    if not is_pass:
        mismatches = int(np.sum(npu != golden))
        print(f"  {status} {label} (integer, {npu.dtype}): "
              f"{mismatches}/{len(golden)} mismatches")
        _show_mismatch_samples(npu, golden, label)
    else:
        print(f"  {status} {label} (integer, {npu.dtype}): bitwise match")
    return is_pass


def check_bitwise(npu, golden, label):
    is_pass = np.array_equal(npu, golden)
    status = "PASS" if is_pass else "FAIL"
    if not is_pass:
        print(f"  {status} {label} (bitwise, {npu.dtype}): mismatch")
    else:
        print(f"  {status} {label} (bitwise, {npu.dtype}): match")
    return is_pass


def _show_mismatch_samples(npu, golden, label, max_samples=10):
    mask = npu != golden
    indices = np.where(mask)[0][:max_samples]
    for i in indices:
        print(f"    [{i}] npu={npu[i]} golden={golden[i]}")


CHECKERS = {
    "float": check_float,
    "integer": check_integer,
    "bitwise": check_bitwise,
}


def main():
    if not OUTPUTS:
        print("ERROR: OUTPUTS 列表为空，请在脚本中配置算子输出。")
        return 1

    all_pass = True
    for npu_path, golden_path, dtype, category in OUTPUTS:
        label = npu_path.split("/")[-1]
        try:
            npu = np.fromfile(npu_path, dtype=dtype)
        except Exception as e:
            print(f"  FAIL {label}: cannot read npu output: {e}")
            all_pass = False
            continue
        try:
            golden = np.fromfile(golden_path, dtype=dtype)
        except Exception as e:
            print(f"  FAIL {label}: cannot read golden: {e}")
            all_pass = False
            continue

        if npu.size != golden.size:
            print(f"  FAIL {label}: size mismatch npu={npu.size} golden={golden.size}")
            all_pass = False
            continue

        checker = CHECKERS.get(category)
        if checker is None:
            print(f"  FAIL {label}: unknown category '{category}'")
            all_pass = False
            continue

        if not checker(npu, golden, label):
            all_pass = False

    print()
    print("=" + (" PASS " if all_pass else " FAIL ") + "=")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
