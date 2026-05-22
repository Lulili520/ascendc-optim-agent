import numpy as np


def calculate_mare(actual, golden):
    relative_errors = np.abs(actual - golden) / (np.abs(golden) + 1e-7)
    return np.max(relative_errors)


def calculate_mere(actual, golden):
    relative_errors = np.abs(actual - golden) / (np.abs(golden) + 1e-7)
    return np.mean(relative_errors)


def get_threshold_by_dtype(dtype):
    dtype_str = str(dtype).lower().replace(' ', '').replace('_', '')
    thresholds = {
        'float16': 2 ** (-10),
        'bfloat16': 2 ** (-7),
        'float32': 2 ** (-13),
        'float64': 2 ** (-13),
        'hifloat32': 2 ** (-11),
    }
    if 'float8e4m3' in dtype_str or 'float8e4m3fn' in dtype_str:
        return 2 ** (-3)
    elif 'float8e5m2' in dtype_str:
        return 2 ** (-2)
    return thresholds.get(dtype_str, 2 ** (-13))


def check_precision_threshold(npu_output, golden_output):
    mare = calculate_mare(npu_output, golden_output)
    mere = calculate_mere(npu_output, golden_output)

    threshold = get_threshold_by_dtype(npu_output.dtype)
    mare_threshold = 10 * threshold

    mere_pass = mere < threshold
    mare_pass = mare < mare_threshold

    is_pass = mere_pass and mare_pass

    result = {
        'is_pass': is_pass,
        'mare': mare,
        'mere': mere,
        'threshold': threshold,
        'mare_threshold': mare_threshold,
        'mere_pass': mere_pass,
        'mare_pass': mare_pass,
        'npu_dtype': str(npu_output.dtype),
        'golden_dtype': str(golden_output.dtype),
        'shape': npu_output.shape
    }

    if not is_pass:
        result['failure_reasons'] = []
        if not mere_pass:
            result['failure_reasons'].append(f'MERE {mere:.6f} >= threshold {threshold:.6f}')
        if not mare_pass:
            result['failure_reasons'].append(f'MARE {mare:.6f} >= mare_threshold {mare_threshold:.6f}')

    return result
