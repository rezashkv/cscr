import faiss
import numpy as np
import json
from typing import List, Optional, Union
from .cost_models import compute_cost
from .bandit import BanditStats
from .query_encoder import QueryEncoder

class KNNRouter:
    def __init__(
        self,
        index_path: str,
        labels_path: str,
        k: int = 1,
        bandit_beta: float = 0.0,
        bandit_lambda: float = 0.1,
        encoder_ckpt: Optional[str] = None,
        device: str = "cuda",
        cost_type: str = "n_params",
        multiplier = 1
    ):
        multiplier = int(multiplier)
        self.index = faiss.read_index(index_path)
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        self.k = k

        self.bandit_enabled = bandit_beta > 0.0
        if self.bandit_enabled:
            self.bandit_stats = BanditStats(bandit_lambda=bandit_lambda, beta=bandit_beta)
        
        if encoder_ckpt:
            self.encoder = QueryEncoder.load(encoder_ckpt, multiplier=multiplier)
        else:
            self.encoder = QueryEncoder.load("sentence-transformers/all-MiniLM-L6-v2")
        
        self.encoder.to(device)
        self.encoder.eval()

        self.cost_type = cost_type

    def route(self, prompt: Union[str, List[str]], candidates: List[str] | None = None,
               n_tokens: Optional[int] = None, **kwargs) -> List[str]:
        """
        Route a prompt embedding to expert(s).
        If bandit_enabled, returns a single expert with UCB exploration.
        Does NOT update bandit stats; update must be done explicitly via register_feedback.
        """
        if kwargs.get("lambda_coeff", None):
            self.bandit_stats.bandit_lambda = kwargs["lambda_coeff"]
        if kwargs.get("bandit_beta", None):
            self.bandit_stats.beta = kwargs["bandit_beta"]
            
        query_emb = self.encoder.encode(prompt)
        emb = query_emb.astype(np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        sims, indices = self.index.search(emb, self.k)
        
        sims = sims[0]
        idxs = indices[0]

        candidates = [self.labels[i] for i in idxs]

        if not self.bandit_enabled:
            if isinstance(prompt, str):
                return candidates[0]
            return candidates

        best_score = -np.inf
        best_label = None
        for sim, label in zip(sims, candidates):
            cost = compute_cost(label, n_tokens or 0, cost_type=self.cost_type)            
            score = self.bandit_stats.get_bonus(label) if self.bandit_enabled else sim
            score += sim  
            score -= self.bandit_stats.bandit_lambda * cost
            if score > best_score:
                best_score = score
                best_label = label

        return best_label
        
    def register_feedback(self, expert_id: str, accuracy: float, cost: float):
        """
        Call *after* you know whether the chosen expert was correct.
        """
        if self.bandit_enabled:
            self.bandit_stats.update(expert_id, accuracy=accuracy, cost=cost)
