import numpy as np
import sys

dtype = np.float16
rtol = 1e-3
atol = 1e-3


def verify_result(output_y_path, golden_y_path):
    output_y = np.fromfile(output_y_path, dtype=dtype)
    golden_y = np.fromfile(golden_y_path, dtype=dtype)

    if output_y.shape != golden_y.shape:
        print(f"y shape mismatch: output {output_y.shape} vs golden {golden_y.shape}")
        return False

    y_ok = np.allclose(output_y, golden_y, rtol=rtol, atol=atol)

    print(f"Values: {'PASS' if y_ok else 'FAIL'}")
    if not y_ok:
        diff = np.abs(output_y.astype(np.float32) - golden_y.astype(np.float32))
        print(f"  Max diff: {np.max(diff):.6f}, Mean diff: {np.mean(diff):.6f}")
        mismatches = np.where(diff > atol + rtol * np.abs(golden_y.astype(np.float32)))[0]
        print(f"  Mismatch count: {len(mismatches)} / {len(golden_y)}")
        if len(mismatches) > 0:
            for i in mismatches[:10]:
                print(f"  [{i}] output={output_y[i]:.6f} golden={golden_y[i]:.6f}")

    return y_ok


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output_y.bin> <golden_y.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
