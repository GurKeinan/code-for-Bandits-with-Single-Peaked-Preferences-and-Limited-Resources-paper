import numpy as np
from extract_order_c1p_wrapper import extract_ordering
from emc_algorithms import construct_sp_from_asp
from sp_matching import sp_matching


def compute_regret_from_matchings(env, all_matchings):
    """
    Compute cumulative regret from sequence of matchings using true reward matrix.
    """
    T = len(all_matchings)
    cumulative_regret = np.zeros(T)
    optimal_value = env.optimal_value
    running_regret = 0.0

    for t in range(T):
        matching_value = env.get_expected_reward(all_matchings[t])
        instant_regret = optimal_value - matching_value
        running_regret += instant_regret
        cumulative_regret[t] = running_regret

    return cumulative_regret


def binary_search_epsilon(UCB_matrix, T, max_iters):
    """
    Binary search to find the smallest epsilon such that extract_ordering succeeds.

    Args:
        UCB_matrix: U x K UCB matrix
        T: time horizon (for determining number of iterations)
        max_iters: maximum number of binary search iterations

    Returns:
        (epsilon, ordering): tuple of found epsilon and corresponding ordering
    """
    lo, hi = 0.0, 1.0
    best_eps = 1.0
    best_ordering = None

    # First check if epsilon=1 works (it should always work)
    ordering = extract_ordering(UCB_matrix, 1.0)
    if ordering is not None:
        best_eps = 1.0
        best_ordering = ordering
    else:
        # This shouldn't happen, but fallback to identity ordering
        best_ordering = list(range(UCB_matrix.shape[1]))

    for _ in range(max_iters):
        mid = (lo + hi) / 2
        ordering = extract_ordering(UCB_matrix, mid)

        if ordering is not None:
            # Success - try smaller epsilon
            best_eps = mid
            best_ordering = ordering
            hi = mid
        else:
            # Failure - need larger epsilon
            lo = mid

    return best_eps, best_ordering


def ucb_extract_order(env, T):
    """
    UCB-Extract-Order algorithm.

    In each round:
    1. Construct UCB matrix
    2. Binary search for smallest epsilon where Extract-Order succeeds
    3. Project UCB matrix to SP matrix using found order
    4. Run SP-Matching and play resulting matching
    5. Update statistics

    Args:
        env: BudgetedMatchingEnvironment instance
        T: Time horizon

    Returns:
        cumulative_regret: Array of cumulative regret over time
    """
    U, K = env.U, env.K
    max_iters = int(np.ceil(np.log(T)))  # log(T) iterations for binary search

    # Initialize statistics
    n_pulls = np.zeros((U, K))
    sum_rewards = np.zeros((U, K))

    # Store all matchings for final regret computation
    all_matchings = []

    # Phase 1: Pull each arm once (K rounds)
    for t in range(K):
        matching = np.full(U, t, dtype=int)  # All users to arm t
        all_matchings.append(matching.copy())
        rewards = env.sample_matching(matching)
        for u in range(U):
            n_pulls[u, matching[u]] += 1
            sum_rewards[u, matching[u]] += rewards[u]

    # Phase 2: Main loop
    for t in range(K, T):
        # Construct UCB matrix
        UCB = np.zeros((U, K))
        for u in range(U):
            for k in range(K):
                if n_pulls[u, k] > 0:
                    UCB[u, k] = sum_rewards[u, k] / n_pulls[u, k] + np.sqrt(2 * np.log(T) / n_pulls[u, k])
                else:
                    UCB[u, k] = 1.0  # Optimistic for unpulled arms

        # Binary search for smallest epsilon
        epsilon, ordering = binary_search_epsilon(UCB, T, max_iters)

        # Reorder UCB matrix according to found ordering: if no ordering was found, use identity
        if len(ordering) != K:
            ordering = list(range(K))
        UCB_reordered = UCB[:, ordering]

        # Project to SP matrix
        delta = 2 * K * epsilon
        P_sp = construct_sp_from_asp(UCB_reordered, delta)

        # Get reordered costs
        reordered_costs = env.costs[ordering]

        # Run SP-Matching on projected matrix
        matching_reordered = sp_matching(P_sp, reordered_costs, env.budget)

        # De-order the matching back to original indices
        matching = np.array([ordering[matching_reordered[u]] for u in range(U)])
        all_matchings.append(matching.copy())

        # Pull arms and observe rewards
        rewards = env.sample_matching(matching)

        # Update statistics
        for u in range(U):
            if matching[u] != -1:
                n_pulls[u, matching[u]] += 1
                sum_rewards[u, matching[u]] += rewards[u]

    # Compute final regret using stored matchings
    cumulative_regret = compute_regret_from_matchings(env, all_matchings)

    return cumulative_regret