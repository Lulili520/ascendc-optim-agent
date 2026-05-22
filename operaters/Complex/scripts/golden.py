# [MODIFY] Golden computation for Complex operator
import numpy as np


def compute_golden(real, imag):
    """Block-interleaved complex construction (BLOCK=32).

    Output layout: [r0..r31, i0..i31, r32..r63, i32..i63, ...]
    """
    BLOCK = 32
    total = real.size
    flat_real = real.flatten()
    flat_imag = imag.flatten()
    output = np.empty(2 * total, dtype=real.dtype)

    for i in range(0, total, BLOCK):
        output[2 * i : 2 * i + BLOCK] = flat_real[i : i + BLOCK]
        output[2 * i + BLOCK : 2 * i + 2 * BLOCK] = flat_imag[i : i + BLOCK]

    # Handle tail (total not divisible by BLOCK)
    remainder = total % BLOCK
    if remainder != 0:
        base = (total // BLOCK) * BLOCK
        for j in range(remainder):
            output[2 * base + j] = flat_real[base + j]
            output[2 * base + remainder + j] = flat_imag[base + j]

    return output
