import numpy as np
from utils import verify_single_peaked

def generate_single_peaked_matrix(U, K, seed=None):
    """
    Generate single-peaked preference matrix with random peaks.
    Each row is generated from uniform distribution, then arranged to be unimodal.
    Values are randomly distributed between left and right sides of the peak.
    """
    if seed is not None:
        np.random.seed(seed)

    matrix = np.zeros((U, K))

    for u in range(U):
        # Generate random values from uniform distribution
        values = np.random.uniform(0.2, 0.9, K)  # Avoid extreme values

        # Pick random peak location
        peak_idx = np.random.randint(0, K)

        # Sort values and separate peak from the rest
        values_sorted = np.sort(values)
        peak_value = values_sorted[-1]
        remaining_values = values_sorted[:-1]  # All values except the largest

        # Randomly shuffle the remaining values
        np.random.shuffle(remaining_values)

        left_size = peak_idx
        right_size = K - peak_idx - 1

        # Assign largest value to peak
        matrix[u, peak_idx] = peak_value

        # Fill left side (increasing to peak)
        if left_size > 0:
            left_values = remaining_values[:left_size]
            left_values.sort()  # Sort ascending for increasing pattern
            matrix[u, :peak_idx] = left_values

        # Fill right side (decreasing from peak)
        if right_size > 0:
            right_values = remaining_values[left_size:left_size + right_size]
            right_values.sort()  # Sort ascending first
            matrix[u, peak_idx + 1:] = right_values[::-1]  # Then reverse for decreasing

    assert verify_single_peaked(matrix), "Generated matrix is not single-peaked"
    return matrix