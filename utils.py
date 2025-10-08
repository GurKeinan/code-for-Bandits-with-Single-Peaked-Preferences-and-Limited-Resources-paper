import numpy as np

def verify_single_peaked(P):
    """
    Verify that a matrix is single-peaked (assumes already in SP order).

    Returns:
        (is_sp, message): bool and descriptive message
    """
    U, K = P.shape

    for u in range(U):
        peak_idx = np.argmax(P[u])

        # Check increasing up to peak
        for k in range(peak_idx):
            if P[u, k] > P[u, k + 1] + 1e-9:  # Small tolerance for numerical issues
                return False

        # Check decreasing after peak
        for k in range(peak_idx, K - 1):
            if P[u, k] + 1e-9 < P[u, k + 1]:
                return False

    return True