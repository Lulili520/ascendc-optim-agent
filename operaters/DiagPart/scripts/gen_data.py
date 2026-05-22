import numpy as np
import os
import sys

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

N = 128
if len(sys.argv) > 1:
    N = int(sys.argv[1])

dtype = np.float16

x = np.random.randn(N, N).astype(dtype)

x.tofile("input/input_x.bin")

golden = compute_golden(x)
golden.tofile("output/golden.bin")

print(f"Generated test data: [{N}, {N}] matrix, dtype={dtype}")
print(f"  input/input_x.bin: {x.shape}, {x.dtype}")
print(f"  output/golden.bin: {golden.shape}, {golden.dtype}")
