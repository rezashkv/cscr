from typing import Dict, List, Optional, Union
from .cost_models import compute_cost

class OracleRouter:
    def __init__(self,
                 cost_type: str = "n_params",
                 fallback_pool: Optional[List[str]] = None):
        self.cost_type = cost_type
        self.fallback_pool = fallback_pool or []


    def route(self,
              prompt: str | dict | None = None,
              candidates: Optional[List[str]] = None,
              label_map: Dict[str, int] | None = None,
              n_tokens: int | None = None,
              lambda_coeff: float = 0.0
              ) -> str:
        assert label_map is not None, "OracleRouter requires correctness labels"
        pool = candidates if candidates else list(label_map.keys())
        n_tok = n_tokens or 0

        best_id = None
        best_score = float("inf")

        for m in pool:
            err = 1 - label_map.get(m, 0)       
            cost = compute_cost(m, n_tok, cost_type=self.cost_type)
            score = err + lambda_coeff * cost   
            if score < best_score:
                best_score, best_id = score, m
                
        return best_id