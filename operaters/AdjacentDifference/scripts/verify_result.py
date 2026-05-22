import numpy as np
import sys

dtype = np.float16


def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    # Output is exactly 0 or 1, use exact comparison
    match = np.array_equal(output, golden)
    diff = np.abs(output.astype(np.float32) - golden.astype(np.float32))

    print(f"Shape: {output.shape}")
    print(f"Max diff: {np.max(diff):.6f}")
    print(f"Mismatch count: {np.sum(diff != 0)} / {len(golden)}")

    if match:
        print("Verification PASSED!")
    else:
        print("Verification FAILED!")
        mismatches = np.where(diff != 0)[0]
        if len(mismatches) > 0:
            print(f"First 10 mismatches:")
            for idx in mismatches[:10]:
                print(f"  [{idx}] output={output[idx]:.4f}, golden={golden[idx]:.4f}")

    return match


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
