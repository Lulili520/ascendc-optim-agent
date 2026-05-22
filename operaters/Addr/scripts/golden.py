import numpy as np


def compute_golden(x1, x2, x3, alpha, beta):
    """Compute golden reference for Addr.

    Formula: out = beta * self + alpha * outer(vec1, vec2)

    Args:
        x1: numpy array (FP16), self matrix [M, N] or flattened
        x2: numpy array (FP16), vec1 vector [M]
        x3: numpy array (FP16), vec2 vector [N]
        alpha: float scalar
        beta: float scalar

    Returns:
        Addr reference output (computed in FP32, returned as FP16, flattened)
    """
    M = len(x2)
    N = len(x3)
    self_fp32 = x1.astype(np.float32).reshape(M, N)
    vec1_fp32 = x2.astype(np.float32)
    vec2_fp32 = x3.astype(np.float32)

    # Outer product: vec1 ⊗ vec2 = vec1[:, None] * vec2[None, :]
    outer = np.outer(vec1_fp32, vec2_fp32)

    result = beta * self_fp32 + alpha * outer
    return result.astype(np.float16).ravel()
