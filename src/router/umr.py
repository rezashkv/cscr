import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import torch
import torch.nn.functional as F

from .query_encoder import QueryEncoder
from .cost_models import compute_cost


class UMRBuilder:
    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        self.encoder = QueryEncoder(embed_model, device=device)
        self.device = device
        self.embed_model = embed_model

    def _embed_batch(self, texts: Sequence[str], bsz: int = 32) -> np.ndarray:
        reps = []
        for i in range(0, len(texts), bsz):
            reps.append(self.encoder.encode(texts[i:i + bsz], project=False))
        return np.vstack(reps)  # (N,D)

    def build(self,
              train_prompts: List[str],
              val_prompts: List[str],
              val_labels: Dict[str, List[int]],
              out_dir: str,
              k: int = 20,
              cost_lambda: float = 0.0,
              cost_type: str = "n_params"):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. cluster on train prompts
        train_emb = self._embed_batch(train_prompts)
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(train_emb)
        centroids = km.cluster_centers_  # (K,D)
        json.dump({"embed_model": self.embed_model,
                   "centroids": centroids.tolist()},
                  open(out / "clusters.json", "w"))

        # 2. assign val prompts to clusters
        val_emb = self._embed_batch(val_prompts)
        assign = pairwise_distances_argmin(val_emb, centroids, metric="euclidean")  

        # 3. per‑model error vec
        errors: Dict[str, List[float]] = {}
        counts = np.bincount(assign, minlength=k)
        for model, lbls in val_labels.items():
            lbls = np.array(lbls)
            err = []
            for c in range(k):
                mask = assign == c
                if mask.any():
                    err.append(1.0 - lbls[mask].mean())
                else:
                    err.append(0.5)  
            errors[model] = err
        json.dump(errors, open(out / "errors.json", "w"))

        # 4. meta (costs)
        meta = {"lambda": cost_lambda, "cost_type": cost_type}
        json.dump(meta, open(out / "meta.json", "w"))
        print(f"✅  Saved UMR artifacts to {out}")


class UMRRouter:
    def __init__(self, work_dir: str,
                 device: str = "cpu",
                 override_lambda: float | None = None):
        self.work = Path(work_dir)
        assert (self.work / "clusters.json").exists(), "run UMRBuilder first"
        self._load(device, override_lambda)

    def _load(self, device: str, override_lambda: float | None):
        clusters = json.load(open(self.work / "clusters.json"))
        self.centroids = torch.tensor(clusters["centroids"], dtype=torch.float, device=device)
        self.k, self.dim = self.centroids.shape
        embed_model = clusters["embed_model"]
        self.encoder = QueryEncoder(embed_model, device=device)
        self.centroids = F.normalize(self.centroids, dim=-1)

        self.errors = json.load(open(self.work / "errors.json"))
        meta = json.load(open(self.work / "meta.json"))
        self.lmbda = override_lambda if override_lambda is not None else meta["lambda"]
        self.cost_type = meta.get("cost_type", "n_params")
        self.device = device

    @torch.no_grad()
    def _cluster_idx(self, prompt: str) -> int:
        emb = self.encoder.encode(prompt, project=False)  # (D,)
        if isinstance(emb, np.ndarray):
            emb = torch.tensor(emb, dtype=torch.float, device=self.device)
        emb = F.normalize(emb, dim=-1)
        sims = emb @ self.centroids.T  # (K,)
        return int(torch.argmax(sims).item())

    def route(self, prompt: str, candidates: List[str] | None = None,
              n_tokens: int | None = None, **kwargs) -> str:
        """Return best expert id among *candidates* (default: all known)."""
        if kwargs.get("lambda_coeff", None):
            self.lmbda = kwargs["lambda_coeff"]
            
        idx = self._cluster_idx(prompt)
        cand = candidates or list(self.errors.keys())

        best, best_score = None, 1e9
        for m in cand:
            if m not in self.errors:
                continue
            err = self.errors[m][idx]
            score = err + self.lmbda * compute_cost(m, n_tokens or 0, self.cost_type)
            if score < best_score:
                best_score = score
                best = m
        return best