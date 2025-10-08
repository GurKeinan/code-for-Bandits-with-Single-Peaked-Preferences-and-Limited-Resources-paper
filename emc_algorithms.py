import numpy as np
from extract_order_c1p_wrapper import extract_ordering
from model import BudgetedMatchingEnvironment
from sp_matching import sp_matching
from utils import verify_single_peaked

def construct_sp_from_asp(P, delta):
    """
    Construct SP matrix from already-ordered matrix following Lemma 6.

    Args:
        P: U x K matrix already in SP ordering
        delta: Approximation parameter

    Returns:
        P_sp: U x K SP matrix (still in SP ordering)
    """
    U, K = P.shape
    P_sp = np.zeros((U, K))

    for u in range(U):
        peak_idx = np.argmax(P[u])

        # For positions up to peak: take max of all values up to k
        for k in range(peak_idx + 1):
            # P_sp[u, k] = max(P[u, i] for i in range(k + 1)) - delta
            P_sp[u, k] = max(P[u, i] for i in range(k + 1))

        # For positions after peak: take max of all values from k onwards
        for k in range(peak_idx + 1, K):
            # P_sp[u, k] = max(P[u, i] for i in range(k, K)) - delta
            P_sp[u, k] = max(P[u, i] for i in range(k, K))

    # Ensure non-negative values
    P_sp = np.maximum(P_sp, 0)

    return P_sp


def explore_then_commit(env, T):
    """
    Highly efficient Explore-then-Commit - returns only final regret.

    Args:
        env: BudgetedMatchingEnvironment instance
        T: Time horizon

    Returns:
        final_regret: Single float value of total regret after T rounds
    """

    U, K = env.U, env.K
    N = int(np.ceil(T**(2/3) * np.log(T)**(1/3)))

    # EXPLORATION PHASE - vectorized sampling
    P_hat = np.zeros((U, K))

    for k in range(K):
        # Sample all N rounds for arm k simultaneously
        bernoulli_probs = env.reward_matrix[:, k]
        samples = env.rng.binomial(1, bernoulli_probs[:, np.newaxis], size=(U, N))
        P_hat[:, k] = np.mean(samples, axis=1)

    # REGRET CALCULATION - no arrays, just final values
    optimal_value = env.optimal_value

    # Total exploration regret
    exploration_regret = 0.0
    rounds_used = 0

    for k in range(K):
        if rounds_used >= T:
            break
        arm_expected_reward = np.sum(env.reward_matrix[:, k])
        arm_regret_per_round = optimal_value - arm_expected_reward
        rounds_for_this_arm = min(N, T - rounds_used)
        exploration_regret += rounds_for_this_arm * arm_regret_per_round
        rounds_used += rounds_for_this_arm

    # If we've used all T rounds in exploration, return
    if rounds_used >= T:
        return exploration_regret

    # COMMITMENT PHASE
    commitment_matching = _get_commitment_matching(P_hat, T, N, K, env)
    commitment_value = env.get_expected_reward(commitment_matching)
    commitment_regret_per_round = optimal_value - commitment_value
    assert commitment_regret_per_round >= 0
    maximal_possible_regret = 2 * U * (2 * K + 1) * np.sqrt(2 * np.log(T) / N)
    assert commitment_regret_per_round <= maximal_possible_regret
    commitment_rounds = T - rounds_used
    commitment_regret = commitment_rounds * commitment_regret_per_round

    return exploration_regret + commitment_regret

def _get_commitment_matching(P_hat, T, N, K, env):
    """Extract ordering and compute commitment matching."""
    epsilon = np.sqrt(2 * np.log(T) / N)
    ordering = extract_ordering(P_hat, epsilon)

    if ordering is None:
        return np.zeros(env.U, dtype=int)

    # Reorder P_hat according to the extracted ordering
    P_hat = P_hat[:, ordering]

    # Construct SP matrix from the ordered P_hat
    delta = 2 * K * epsilon
    P_sp = construct_sp_from_asp(P_hat, delta)

    # check if P_sp is single-peaked
    assert verify_single_peaked(P_sp), "Constructed SP matrix is not single-peaked"

    # check if P_sp is (2K + 1)epsilon close to the real reward matrix
    reordered_reward_matrix = env.reward_matrix[:, ordering]
    assert np.all(np.abs(P_sp - reordered_reward_matrix) <= (2 * K + 1) * epsilon), \
        "Constructed SP matrix is not close enough to the original reward matrix."

    reordered_costs = env.costs[ordering]
    commitment_matching_before_deordering = sp_matching(P_sp, reordered_costs, env.budget)

    # Deorder the matching to match the original column order
    commitment_matching = np.zeros(env.U, dtype=int)
    for u in range(env.U):
        commitment_matching[u] = ordering[commitment_matching_before_deordering[u]]

    return commitment_matching