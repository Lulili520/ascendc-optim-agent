#!/usr/bin/env python3
# [MODIFY] Verify Complex operator output against golden
import sys
import numpy as np


def verify_result(output_path, golden_path):
    dtype = np.float16
    rtol = 1e-3
    atol = 1e-3

    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"[FAIL] Shape mismatch: output={output.shape}, golden={golden.shape}")
        sys.exit(1)

    if np.allclose(output, golden, rtol=rtol, atol=atol):
        print(f"[PASS] Verified {output.size} elements, rtol={rtol}, atol={atol}")
        return True
    else:
        diff = np.abs(output.astype(np.float32) - golden.astype(np.float32))
        print(f"[FAIL] Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")
        mismatches = np.sum(diff > atol)
        print(f"  Mismatches: {mismatches}/{output.size}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <output.bin> <golden.bin>")
        sys.exit(1)
    verify_result(sys.argv[1], sys.argv[2])
