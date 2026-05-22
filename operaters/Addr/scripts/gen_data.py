import numpy as np
import os

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

np.random.seed(42)

test_cases = [
    {"name": "case1", "M": 512, "N": 1024},
    {"name": "case2", "M": 1024, "N": 2048},
]

dtype = np.float16

for tc in test_cases:
    M = tc["M"]
    N = tc["N"]
    name = tc["name"]

    x1 = (np.random.randn(M, N).astype(np.float32) * 0.5).astype(dtype)
    x2 = (np.random.randn(M).astype(np.float32) * 0.5).astype(dtype)
    x3 = (np.random.randn(N).astype(np.float32) * 0.5).astype(dtype)
    alpha = float(np.random.rand() * 2 - 1)
    beta = float(np.random.rand() * 2 - 1)

    x1.ravel().tofile(f"input/input_x1_{name}.bin")
    x2.tofile(f"input/input_x2_{name}.bin")
    x3.tofile(f"input/input_x3_{name}.bin")
    np.array([alpha], dtype=np.float32).tofile(f"input/input_alpha_{name}.bin")
    np.array([beta], dtype=np.float32).tofile(f"input/input_beta_{name}.bin")

    golden = compute_golden(x1.ravel(), x2, x3, alpha, beta)
    golden.tofile(f"output/golden_{name}.bin")

    print(f"Generated case '{name}': M={M}, N={N}, alpha={alpha:.6f}, beta={beta:.6f}")

print("All test data generated.")
