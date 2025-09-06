import math
from collections import defaultdict

class BanditStats:
    def __init__(self, bandit_lambda: float = 1.0, beta: float = 0.5):
        self.bandit_lambda = bandit_lambda
        self.beta = beta
        self.counts = defaultdict(int)
        self.rewards = defaultdict(float)
        self.total_steps = 0

    def update(self, expert_id: str, accuracy: float, cost: float):
        reward = accuracy - self.bandit_lambda * cost
        self.counts[expert_id] += 1
        self.rewards[expert_id] += reward
        self.total_steps += 1

    def mean_reward(self, expert_id: str) -> float:
        n = self.counts[expert_id]
        return self.rewards[expert_id] / n if n else 0.0

    def get_bonus(self, expert_id: str) -> float:
        """
        Pure exploration bonus  β * sqrt(log(t) / n).
        """
        n = self.counts[expert_id]
        if n == 0:
            return float("inf")
        return self.beta * math.sqrt(math.log(1 + self.total_steps) / n)