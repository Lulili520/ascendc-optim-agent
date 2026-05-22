import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from golden import compute_golden

TEST_CASES = [
    # (B, T, F, dim, exclusive, reverse)
    (8, 1024, 256, 1, 0, 0),
    (4, 2048, 512, 1, 0, 0),
]

case_idx = int(os.environ.get("TEST_CASE", "0"))
if case_idx >= len(TEST_CASES):
    print(f"Invalid test case index {case_idx}, max {len(TEST_CASES)-1}")
    sys.exit(1)

B, T, F, dim, exclusive, reverse = TEST_CASES[case_idx]

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

dtype = np.float16
x = np.random.randn(B, T, F).astype(dtype)
x.tofile("input/input_x.bin")

golden_y = compute_golden(x, axis=dim, exclusive=bool(exclusive), reverse=bool(reverse))
golden_y.tofile("output/golden_y.bin")

print(f"Case {case_idx}: shape=[{B},{T},{F}] dim={dim} exclusive={exclusive} reverse={reverse} dtype={dtype}")
print(f"  input/input_x.bin: {x.shape} {x.dtype}")
print(f"  output/golden_y.bin: {golden_y.shape} {golden_y.dtype}")
