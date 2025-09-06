

import math
import json
import random
import functools
from typing import List, Optional, Dict, Callable
import numpy as np

from .umr import UMRRouter
from .knn_router import KNNRouter
from .oracle_router import OracleRouter
from .random_router import RandomRouter as _RandomRouter
from .query_encoder import QueryEncoder
from .cost_models import compute_cost

class ParametricScorerRouter:
    def __init__(self, W: np.ndarray, b: float = 0.0,
                  encoder_ckpt: Optional[str] = None,
                    cost_type: str = "n_params",
                    expert_embeds: Optional[Dict[str, np.ndarray]] = None):
        self.W = W.reshape(-1)
        self.b = b
        self.cost_type = cost_type
        self.encoder = QueryEncoder.load(encoder_ckpt or "sentence-transformers/all-MiniLM-L6-v2")
        self.encoder.eval()
        self.expert_embeds = expert_embeds

    def route(self, prompt, candidates: Optional[List[str]] = None, n_tokens: Optional[int] = None,
              lambda_coeff: float = 0.0, **kwargs) -> str:
        if candidates is None or self.expert_embeds is None:
            raise ValueError("ParametricScorerRouter requires candidates and expert_embeds dict")
        n_tok = n_tokens or 0
        q = self.encoder.encode(prompt)
        # For each candidate, compute s = W·d_expert + b - λ·cost
        best, best_score = None, -np.inf
        for m in candidates:
            d = self.expert_embeds[m].reshape(-1)
            s = float(np.dot(self.W, d)) + self.b
            cost = compute_cost(m, n_tok, cost_type=self.cost_type)
            s -= lambda_coeff * cost
            if s > best_score:
                best, best_score = m, s
        return best


class SoftGatedMoERouter:
    def __init__(self, encoder_ckpt: Optional[str] = None, tau: float = 1.0,
                  cost_type: str = "n_params", top_k: int = 1,
                  expert_embeds: Optional[Dict[str, np.ndarray]] = None, proj_dim: int = 256):
        self.encoder = QueryEncoder.load(encoder_ckpt or "sentence-transformers/all-MiniLM-L6-v2", 
                                          proj_dim=proj_dim)
        self.encoder.eval()
        self.tau = tau
        self.cost_type = cost_type
        self.top_k = top_k
        self.expert_embeds = expert_embeds

    def route(self, prompt, candidates: Optional[List[str]] = None, n_tokens: Optional[int] = None,
              lambda_coeff: float = 0.0, **kwargs):
        if candidates is None or self.expert_embeds is None:
            raise ValueError("SoftGatedMoERouter requires candidates and expert_embeds dict")
        n_tok = n_tokens or 0
        q = self.encoder.encode(prompt)
        # Compute similarity for each candidate
        sims = []
        costs = []
        for m in candidates:
            d = self.expert_embeds[m].reshape(-1)
            sim = float(np.dot(q, d)) / (np.linalg.norm(q) * np.linalg.norm(d) + 1e-8)
            sims.append(sim)
            costs.append(compute_cost(m, n_tok, cost_type=self.cost_type))
        sims = np.array(sims)
        costs = np.array(costs)
        logits = (sims - lambda_coeff * costs) / self.tau
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        idxs = np.argsort(-probs)
        if self.top_k == 1:
            return candidates[idxs[0]]
        else:
            return [candidates[i] for i in idxs[:self.top_k]]


class StaticThompsonRouter:

    def __init__(self, candidates: List[str], cost_type: str = "n_params", alpha: float = 1.0, beta: float = 1.0):
        self.candidates = list(candidates)
        self.cost_type = cost_type
        self.alpha = {m: float(alpha) for m in self.candidates}
        self.beta = {m: float(beta) for m in self.candidates}

    def route(self, prompt=None, candidates: Optional[List[str]] = None, n_tokens: Optional[int] = None,
              lambda_coeff: float = 0.0, **kwargs) -> str:
        pool = candidates if candidates else self.candidates
        n_tok = n_tokens or 0
        best, best_score = None, -np.inf
        for m in pool:
            theta = np.random.beta(self.alpha[m], self.beta[m])
            cost = compute_cost(m, n_tok, cost_type=self.cost_type)
            score = theta - lambda_coeff * cost
            if score > best_score:
                best, best_score = m, score
        return best

    def register_feedback(self, expert_id: str, accuracy: float, cost: float):
        if expert_id not in self.alpha:
            self.alpha[expert_id] = 1.0
            self.beta[expert_id] = 1.0
        self.alpha[expert_id] += accuracy
        self.beta[expert_id] += 1.0 - accuracy


class RandomRouter(_RandomRouter):
    def route(self, prompt=None, candidates: Optional[List[str]] = None,
              lambda_coeff: float = 0.0, n_tokens: Optional[int] = None,
              stats: Optional[Dict[str, Dict[str, float]]] = None, **_kwargs) -> str:
        pool = candidates if candidates else self.labels
        n_tok = n_tokens or 0

        if lambda_coeff == "pareto" and stats is not None:
            items = [(m, stats[m]["cost"], stats[m]["error"]) for m in pool if m in stats]
            if not items:
                return super().route(prompt, pool, 0.0, n_tok)
            pareto = []
            for i, (m, c, e) in enumerate(items):
                dominated = False
                for j, (m2, c2, e2) in enumerate(items):
                    if j == i:
                        continue
                    if (c2 <= c and e2 <= e) and (c2 < c or e2 < e):
                        dominated = True
                        break
                if not dominated:
                    pareto.append(m)
            if not pareto:
                pareto = [m for m, _, _ in items]
            return random.choice(pareto)
        else:
            return super().route(prompt, pool, lambda_coeff, n_tok)


def get_baseline(name: str, **kwargs) -> Callable[..., object]:
    name = name.lower()
    if name == "umr":
        return functools.partial(UMRRouter, **kwargs)
    elif name == "knn":
        return functools.partial(KNNRouter, **kwargs)
    elif name == "oracle":
        return functools.partial(OracleRouter, **kwargs)
    elif name == "random":
        return functools.partial(RandomRouter, **kwargs)
    elif name == "pareto_random":
        return functools.partial(RandomRouter, **kwargs)
    elif name == "parametric":
        return functools.partial(ParametricScorerRouter, **kwargs)
    elif name == "softmoe":
        return functools.partial(SoftGatedMoERouter, **kwargs)
    elif name == "thompson":
        return functools.partial(StaticThompsonRouter, **kwargs)
    else:
        raise ValueError(f"Unknown baseline router: {name}")