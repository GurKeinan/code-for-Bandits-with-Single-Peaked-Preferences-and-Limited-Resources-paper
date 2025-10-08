import numpy as np
from itertools import product

def sp_matching(P, costs, budget):
    """
    Dynamic programming algorithm for single-peaked matching.

    Args:
        P: U x K preference matrix (must be single-peaked)
        costs: K-length cost vector (all ones for our case)
        budget: budget constraint

    Returns:
        matching: U-length array where matching[u] = assigned arm for user u
    """
    U, K = P.shape

    # Find peaks
    peaks = np.argmax(P, axis=1)

    # Add fictitious arm 0
    P_extended = np.zeros((U, K+1))
    P_extended[:, 1:] = P
    costs_extended = np.zeros(K+1)
    costs_extended[1:] = costs

    # Precompute G matrix
    G = {}
    for i in range(K+1):
        for j in range(i+1, K+1):
            G[i,j] = 0
            for u in range(U):
                if i < peaks[u] + 1 <= j:  # +1 because of 0-indexing
                    G[i,j] += max(P_extended[u,i], P_extended[u,j])

    # Dynamic programming
    F = {}
    for b in range(budget+1):
        F[0, b] = 0

    for k in range(1, K+1):
        for b in range(int(costs_extended[k]), budget+1):
            F[k, b] = -float('inf')
            for i in range(k):
                if b >= costs_extended[i] + costs_extended[k]:
                    val = F.get((i, b - int(costs_extended[k])), -float('inf'))
                    if val > -float('inf'):
                        F[k, b] = max(F[k, b], val + G.get((i,k), 0))

    # Find optimal last arm
    best_val = -float('inf')
    best_k = -1
    for k in range(1, K+1):
        val = F.get((k, budget), -float('inf'))
        if val > -float('inf'):
            # Add contribution from users with peaks > k
            extra = sum(P[u, k-1] for u in range(U) if peaks[u] + 1 > k)
            if val + extra > best_val:
                best_val = val + extra
                best_k = k

    # Backtrack to find selected arms
    selected_arms = []
    if best_k != -1:
        selected_arms = [best_k - 1]  # Convert back to 0-indexing
        k = best_k
        b = budget

        while k > 0:
            found = False
            for i in range(k):
                if b >= costs_extended[i] + costs_extended[k]:
                    val = F.get((i, b - int(costs_extended[k])), -float('inf'))
                    if val > -float('inf') and val + G.get((i,k), 0) == F[k, b]:
                        if i > 0:
                            selected_arms.append(i - 1)  # Convert to 0-indexing
                        k = i
                        b = b - int(costs_extended[k])
                        found = True
                        break
            if not found:
                break

    # Assign users to arms
    matching = np.zeros(U, dtype=int)
    selected_set = set(selected_arms)
    default_arm = selected_arms.pop()
    for u in range(U):
        best_arm = -1
        best_val = -float('inf')
        for arm in selected_set:
            if P[u, arm] > best_val:
                best_val = P[u, arm]
                best_arm = arm
        matching[u] = best_arm if best_arm != -1 else default_arm

    return matching

def brute_force_matching(P, costs, budget):
    """
    Brute force solution for the matching problem.
    WARNING: Exponential complexity - use only for small instances!
    """
    U, K = P.shape

    best_value = -float('inf')
    best_matching = None

    for matching_tuple in product(range(K), repeat=U):
        matching = np.array(matching_tuple)

        # Check budget constraint
        selected_arms = set(matching)
        total_cost = sum(costs[k] for k in selected_arms)

        if total_cost <= budget:
            # Compute value
            value = sum(P[u, matching[u]] for u in range(U))

            if value > best_value:
                best_value = value
                best_matching = matching.copy()

    return best_matching, best_value

def generate_single_peaked_matrix(U, K, seed=None):
    """
    Generate single-peaked preference matrix where each user has unimodal preferences.
    Each row represents a user's preferences that increase to a peak and then decrease.
    """
    if seed is not None:
        np.random.seed(seed)

    matrix = np.zeros((U, K))

    for u in range(U):
        # Generate random values and sort them
        values = np.random.uniform(0.2, 0.9, K)
        values_sorted = np.sort(values)

        # Choose random peak location
        peak_idx = np.random.randint(0, K)

        # Assign largest value to peak
        matrix[u, peak_idx] = values_sorted[-1]

        # Fill left side (increasing to peak)
        left_size = peak_idx
        for i in range(left_size):
            matrix[u, i] = values_sorted[i]

        # Fill right side (decreasing from peak)
        right_size = K - peak_idx - 1
        for i in range(right_size):
            matrix[u, peak_idx + 1 + i] = values_sorted[left_size + i]

    return matrix

def verify_single_peaked(matrix):
    """Verify that a matrix satisfies the single-peaked property."""
    U, K = matrix.shape
    for u in range(U):
        peak_idx = np.argmax(matrix[u])

        # Check increasing before peak
        for i in range(peak_idx):
            if matrix[u, i] > matrix[u, i+1]:
                return False

        # Check decreasing after peak
        for i in range(peak_idx, K-1):
            if matrix[u, i] < matrix[u, i+1]:
                return False

    return True

def compute_matching_value(P, matching):
    """Compute the total value of a matching."""
    return sum(P[u, matching[u]] for u in range(len(matching)))

def compare_algorithms(U, K, budget, seed=None):
    """Compare SP-Matching algorithm with brute force on a random instance."""

    # Generate single-peaked instance
    P = generate_single_peaked_matrix(U, K, seed)
    costs = np.ones(K)  # Unit costs

    print(f"Generated {U}x{K} single-peaked matrix (budget={budget})")
    print(f"Single-peaked verification: {verify_single_peaked(P)}")
    print(f"Preference matrix:\n{P.round(3)}")
    print(f"User peaks: {np.argmax(P, axis=1)}")
    print()

    # Run SP-Matching algorithm
    print("Running SP-Matching algorithm...")
    sp_matching_result = sp_matching(P, costs, budget)
    sp_value = compute_matching_value(P, sp_matching_result)
    print(f"SP-Matching result: {sp_matching_result}")
    print(f"SP-Matching value: {sp_value:.4f}")

    # Run brute force (only for small instances)
    if K**U <= 10000:  # Limit to avoid excessive computation
        print("\nRunning brute force algorithm...")
        bf_matching, bf_value = brute_force_matching(P, costs, budget)
        print(f"Brute force result: {bf_matching}")
        print(f"Brute force value: {bf_value:.4f}")

        # Compare results
        print(f"\nComparison:")
        print(f"Values match: {abs(sp_value - bf_value) < 1e-10}")
        print(f"Matchings identical: {np.array_equal(sp_matching_result, bf_matching)}")

        if abs(sp_value - bf_value) > 1e-10:
            print("WARNING: Values don't match! Check implementation.")
        else:
            print("✓ SP-Matching finds optimal solution")
    else:
        print(f"\nSkipping brute force (too large: {K}^{U} = {K**U} possibilities)")

    print("-" * 60)

def main():
    """Run several test cases to compare algorithms."""

    print("Testing SP-Matching Algorithm vs Brute Force")
    print("=" * 60)

    # Test cases: (U, K, budget, seed)
    test_cases = [
        (3, 3, 2, 42),    # Small case
        (3, 4, 3, 123),   # Medium case
        (4, 3, 2, 456),   # Different dimensions
        (2, 5, 3, 789),   # More arms than users
        (5, 3, 2, 999),   # More users than arms (brute force will be skipped)
    ]

    for i, (U, K, budget, seed) in enumerate(test_cases):
        print(f"Test Case {i+1}:")
        compare_algorithms(U, K, budget, seed)
        print()

if __name__ == "__main__":
    main()