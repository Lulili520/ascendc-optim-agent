import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mare_mere_threshold import check_precision_threshold

dtype = np.float16


def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    result = check_precision_threshold(output, golden)

    print(f"Shape: {output.shape}")
    print(f"MERE: {result['mere']:.6f}, Threshold: {result['threshold']:.6f}, "
          f"Pass: {result['mere_pass']}")
    print(f"MARE: {result['mare']:.6f}, MareThreshold: {result['mare_threshold']:.6f}, "
          f"Pass: {result['mare_pass']}")
    print(f"Max diff: {np.max(np.abs(output - golden)):.6f}")

    rtol, atol = 1e-3, 1e-3
    allclose = np.allclose(output, golden, rtol=rtol, atol=atol)
    print(f"allclose(rtol={rtol}, atol={atol}): {allclose}")

    passed = result['is_pass'] or allclose
    if passed:
        print("Verification PASSED!")
    else:
        print("Verification FAILED!")
        if 'failure_reasons' in result:
            for reason in result['failure_reasons']:
                print(f"  Reason: {reason}")

    return passed


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
