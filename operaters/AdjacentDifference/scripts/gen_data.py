import numpy as np
import os
import sys

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

test_cases = {
    1: {"name": "case1", "shape": [8, 1024, 256]},
    2: {"name": "case2", "shape": [4, 2048, 512]},
}

case_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
if case_num not in test_cases:
    print(f"Invalid case number: {case_num}. Available: {list(test_cases.keys())}")
    sys.exit(1)

tc = test_cases[case_num]
shape = tc["shape"]
name = tc["name"]
total_length = int(np.prod(shape))

dtype = np.float16
np.random.seed(42 + case_num)

x_data = np.random.randn(*shape).astype(dtype)
x_data.ravel().tofile("input/input_x.bin")

golden = compute_golden(x_data)
golden.ravel().tofile(f"output/golden_{name}.bin")

print(f"Generated {name}: shape={shape}, {total_length} elements, dtype={dtype}")
print(f"  Unique output values: {np.unique(golden)}")
print(f"  Zero count: {np.sum(golden == 0)}, One count: {np.sum(golden == 1)}")
