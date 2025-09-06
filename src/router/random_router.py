import math
from .cost_models import compute_cost
import json
import random

from typing import List, Optional


class RandomRouter:

    def __init__(self,
                 labels_json: str,
                 cost_type: str = "n_params"):
        with open(labels_json, "r") as f:
            self.labels: List[str] = json.load(f)
        self.cost_type = cost_type

    def route(self,
              prompt: str | dict | None = None,      
              candidates: Optional[List[str]] = None,
              lambda_coeff: float = 0.0,
              n_tokens: int | None = None,
              **_kwargs) -> str:
        pool = candidates if candidates else self.labels
        n_tok = n_tokens or 0

        if lambda_coeff is None or lambda_coeff <= 0:
            return random.choice(pool)

        weights = []
        for m in pool:
            cost = compute_cost(m, n_tok, cost_type=self.cost_type)
            weights.append(math.exp(-lambda_coeff * cost))

        Z = sum(weights)
        probs = [w / Z for w in weights]
        return random.choices(pool, weights=probs, k=1)[0]