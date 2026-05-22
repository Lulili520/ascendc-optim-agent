import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from golden import compute_golden


def main():
    if len(sys.argv) != 5:
        print(f"Usage: python3 gen_data.py <B> <T> <F> <dim>")
        print(f"  B:   batch size")
        print(f"  T:   sequence length (scan axis when dim=1)")
        print(f"  F:   feature dimension")
        print(f"  dim: scan axis (0=B, 1=T, 2=F)")
        sys.exit(1)

    B = int(sys.argv[1])
    T = int(sys.argv[2])
    F = int(sys.argv[3])
    dim = int(sys.argv[4])

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    dtype = np.float16
    x = np.random.randn(B, T, F).astype(dtype)
    x.tofile("input/input_x.bin")

    golden_y, golden_argmin = compute_golden(x, axis=dim)
    golden_y.tofile("output/golden_y.bin")
    golden_argmin.tofile("output/golden_argmin.bin")

    print(f"Shape: B={B}, T={T}, F={F}, dim={dim}, dtype={dtype}")
    print(f"  input/input_x.bin: {x.shape}")
    print(f"  output/golden_y.bin: {golden_y.shape}")
    print(f"  output/golden_argmin.bin: {golden_argmin.shape}")


if __name__ == "__main__":
    main()
