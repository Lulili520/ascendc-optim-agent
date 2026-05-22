#!/usr/bin/env python3
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

total_length = 8192
if len(sys.argv) > 1:
    total_length = int(sys.argv[1])

dtype = np.float16

real = np.random.randn(total_length).astype(dtype)
imag = np.random.randn(total_length).astype(dtype)

real.tofile("input/input_real.bin")
imag.tofile("input/input_imag.bin")

golden = compute_golden(real, imag)
golden.tofile("output/golden.bin")

print(f"Generated: {total_length} elements, dtype={dtype}")
print(f"  Input size: {real.nbytes} bytes each")
print(f"  Output size: {golden.nbytes} bytes")
