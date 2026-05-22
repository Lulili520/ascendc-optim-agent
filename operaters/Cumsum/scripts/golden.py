import numpy as np


def compute_golden(x, axis, exclusive=False, reverse=False):
    # Compute cumsum in FP32 for higher precision, matching kernel behavior
    x_fp32 = x.astype(np.float32)

    if reverse:
        x_fp32 = np.flip(x_fp32, axis=axis)

    golden_y = np.cumsum(x_fp32, axis=axis)

    if exclusive:
        golden_y = np.concatenate(
            [np.zeros_like(np.take(golden_y, [0], axis=axis)),
             np.take(golden_y, range(0, golden_y.shape[axis] - 1), axis=axis)],
            axis=axis
        )

    if reverse:
        golden_y = np.flip(golden_y, axis=axis)

    return golden_y.astype(x.dtype)
