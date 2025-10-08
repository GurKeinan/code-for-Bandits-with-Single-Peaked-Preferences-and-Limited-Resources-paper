import numpy as np
from sp_matching import sp_matching

def compute_regret_from_matchings(env, all_matchings):
    """
    Compute cumulative regret from sequence of matchings using true reward matrix.

    Args:
        env: BudgetedMatchingEnvironment instance
        all_matchings: List of matchings, one per time step

    Returns:
        cumulative_regret: Array of cumulative regret over time
    """
    T = len(all_matchings)
    cumulative_regret = np.zeros(T)

    # Get optimal value once
    optimal_value = env.optimal_value

    running_regret = 0.0
    for t in range(T):
        # Compute expected reward of chosen matching using true matrix
        matching_value = env.get_expected_reward(all_matchings[t])

        # Instantaneous regret
        instant_regret = optimal_value - matching_value
        running_regret += instant_regret

        cumulative_regret[t] = running_regret

    return cumulative_regret

def match_via_maximal(env, T):
    """
    Match-via-Maximal algorithm - store matchings for final regret computation.
    """
    U, K = env.U, env.K

    # Initialize statistics
    n_pulls = np.zeros((U, K))
    sum_rewards = np.zeros((U, K))

    # Store all matchings for final regret computation
    all_matchings = []

    # pull each arm once in the first round - assign all users to it.
    for t in range(K):
        matching = np.zeros(U, dtype=int)
        matching[:] = t % K # Round-robin assignment
        all_matchings.append(matching.copy())
        rewards = env.sample_matching(matching)
        for u in range(U):
            n_pulls[u, matching[u]] += 1
            sum_rewards[u, matching[u]] += rewards[u]

    for t in range(T - K):
        # Construct UCB matrix
        UCB = np.zeros((U, K))
        for u in range(U):
            for k in range(K):
                assert n_pulls[u, k] > 0
                UCB[u, k] = sum_rewards[u, k] / n_pulls[u, k] + np.sqrt(2 * np.log(T) / n_pulls[u, k])

        # Construct maximal matrix
        M = np.zeros((U, K))
        peaks = env.peaks

        for u in range(U):
            for k in range(K):
                if k <= peaks[u]:
                    M[u, k] = min(UCB[u, j] for j in range(k, peaks[u] + 1))
                else:
                    M[u, k] = min(UCB[u, j] for j in range(peaks[u], k+1))

        # Find optimal matching for maximal matrix
        matching = sp_matching(M, env.costs, env.budget)
        all_matchings.append(matching.copy())  # Store the matching

        # Pull arms and observe rewards (for learning, not regret)
        rewards = env.sample_matching(matching)

        # Update statistics
        for u in range(U):
            if matching[u] != -1:
                n_pulls[u, matching[u]] += 1
                sum_rewards[u, matching[u]] += rewards[u]

    # Compute final regret using stored matchings and true reward matrix
    cumulative_regret = compute_regret_from_matchings(env, all_matchings)

    return cumulative_regret