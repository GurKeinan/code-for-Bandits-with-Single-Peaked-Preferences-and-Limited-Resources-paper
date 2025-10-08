import numpy as np
# from PADS_C1P import ConsecutiveOnesTest
# from tryalgo_C1P import consecutive_ones_test
# from tryalgo.PC_tree import PC_tree
# from c1p_via_lexbf import ConsecutiveOnesTest
from sagemath_c1p import consecutive_ones_test

def extract_ordering(P: np.ndarray, epsilon: float) -> list[int] | None:
    """
    Given a ``U × K`` preference matrix ``P`` (larger = preferred),
    return an ordering of the ``K`` items that makes it
    *approximately* single-peaked within margin ``epsilon``.
    """
    U, K = P.shape
    consecutive_sets: list[set[int]] = []

    for u in range(U):
        items = np.argsort(-P[u])  # descending
        for idx in range(K - 1):
            if P[u, items[idx]] - P[u, items[idx + 1]] > 2 * epsilon:
                consecutive_sets.append({int(x) for x in items[:idx + 1]})

    # construct the consecutive ones matrix
    consecutive_ones_matrix = np.zeros((len(consecutive_sets), K), dtype=int)
    for i, s in enumerate(consecutive_sets):
        for item in s:
            consecutive_ones_matrix[i, item] = 1

    # return consecutive_ones_test(consecutive_ones_matrix)
    # solver = ConsecutiveOnesTest()
    # has_property, ordering = solver.has_consecutive_ones(consecutive_ones_matrix)
    # if has_property:
    #     return ordering
    # else:
    #     return None
    has_property, ordering = consecutive_ones_test(consecutive_ones_matrix)
    if has_property:
        return ordering
    else:
        return None