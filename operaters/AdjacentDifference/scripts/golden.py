import numpy as np


def compute_golden(x):
    """AdjacentDifference golden computation.

    Formula:
        out[0] = (x[0] == 0) ? 0 : 1
        out[i] = (x[i] == x[i-1]) ? 0 : 1

    Args:
        x: numpy array (FP16), can be any shape (will be flattened)

    Returns:
        numpy array (FP16), same shape as input
    """
    x_flat = x.ravel().astype(np.float16)
    out = np.zeros_like(x_flat)

    # First element compared with 0
    out[0] = 0.0 if x_flat[0] == np.float16(0.0) else 1.0

    # Remaining elements compared with previous
    diff = x_flat[1:] != x_flat[:-1]
    out[1:] = diff.astype(np.float16)

    return out.reshape(x.shape)
