# model.py
import numpy as np
from extract_order_c1p_wrapper import extract_ordering
from utils import verify_single_peaked

class BudgetedMatchingEnvironment:
    """Simple environment for budgeted contextual bandit matching with single-peaked preferences."""

    def __init__(self, U, K, reward_matrix, is_ordered, budget=None, costs=None, seed=None):
        """
        Args:
            U: number of users
            K: number of arms
            reward_matrix: U x K matrix of Bernoulli parameters (must be single-peaked)
            budget: budget constraint (defaults to K if None)
            costs: cost vector (default=s to all ones if None)
            seed: random seed
        """
        self.U = U
        self.K = K
        self.reward_matrix = reward_matrix.copy()
        self.is_ordered = is_ordered
        self.budget = budget if budget is not None else K // 2  # Default budget is half the number of arms
        self.costs = costs if costs is not None else np.ones(K)

        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        # Find peaks and verify single-peaked structure
        self.peaks = np.argmax(self.reward_matrix, axis=1)

        # Compute optimal matching value for regret calculation
        self.optimal_value = self.get_optimal_value()

    def sample(self, user, arm):
        """Sample reward for user-arm pair."""
        p = self.reward_matrix[user, arm]
        return self.rng.binomial(1, p)

    def sample_matching(self, matching):
        """Sample rewards for all users given a matching."""
        rewards = np.zeros(self.U)
        for u in range(self.U):
            if matching[u] != -1:  # -1 indicates no assignment
                rewards[u] = self.sample(u, matching[u])
        return rewards

    def get_expected_reward(self, matching):
        """Get expected reward of a matching."""
        total = 0
        for u in range(self.U):
            if matching[u] != -1:
                total += self.reward_matrix[u, matching[u]]
        return total

    def get_optimal_value(self):
        """Get optimal matching value (computed only once)."""

        # reorder the matrix if needed
        if not self.is_ordered:
            order = extract_ordering(self.reward_matrix, 0)
            if order is None:
                raise ValueError("Could not extract a valid ordering from the reward matrix.")
        else:
            order = np.arange(self.K)

        reordered_reward_matrix = self.reward_matrix[:, order]
        reordered_costs = self.costs[order]
        assert verify_single_peaked(reordered_reward_matrix)


        from sp_matching import sp_matching
        optimal_matching_before_deordering = sp_matching(reordered_reward_matrix, reordered_costs, self.budget)
        # Deorder the matching to match the original column order
        optimal_matching = np.zeros(self.U, dtype=int)
        for u in range(self.U):
            optimal_matching[u] = order[optimal_matching_before_deordering[u]] if order is not None else optimal_matching_before_deordering[u]

        return self.get_expected_reward(optimal_matching)

    def reset_random_state(self, seed):
        """Reset random state for reproducibility."""
        self.rng = np.random.RandomState(seed)