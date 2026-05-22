import numpy as np


def compute_golden(x, axis):
    """计算 cummin 的参考输出 (values, indices)。

    Args:
        x: numpy array, shape [B, T, F]
        axis: int, 累积维度

    Returns:
        (values, indices): 累积最小值和对应索引
    """
    values = np.minimum.accumulate(x, axis=axis)

    indices = np.zeros(x.shape, dtype=np.int32)
    n = x.shape[axis]
    for i in range(n):
        slc = [slice(None)] * x.ndim
        slc[axis] = slice(0, i + 1)
        prefix = x[tuple(slc)]
        argmin_prefix = np.argmin(prefix, axis=axis)

        slc_out = [slice(None)] * x.ndim
        slc_out[axis] = i
        indices[tuple(slc_out)] = argmin_prefix

    return values, indices
