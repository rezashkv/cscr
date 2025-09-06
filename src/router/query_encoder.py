from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class QueryEncoder(nn.Module):
    def __init__(self, model_name: str,
                device: str | None = None,
                proj_dim: int = 256,
                proj_multiplier: int = 1):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
        self.hidden_size = self.model.config.hidden_size
        self.proj_dim = proj_dim 
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, proj_multiplier * self.hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(proj_multiplier * self.hidden_size, proj_dim, bias=False)
        ).to(self.device)

        self.model.config.proj_dim = proj_dim

    @torch.no_grad()
    def encode(self, texts: List[str] | str, project: bool = True) -> np.ndarray:
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        ).to(self.device)

        outputs = self.model(**batch)
        
        last_hidden = outputs.last_hidden_state       # (B, L, H)
        cls_vec = last_hidden[:, 0]                   # (B, H)
        if project:
            z = self._project(cls_vec)                    # (B, D)
            vec = z.cpu().numpy().astype(np.float32)
        else:
            vec = cls_vec.cpu().numpy().astype(np.float32)
        
        return vec[0] if single_input else vec
    
    def _project(self, cls_vec: torch.Tensor) -> torch.Tensor:
        z = self.proj(cls_vec)            # (B, proj_dim)
        z = torch.nn.functional.normalize(z, dim=-1)
        return z
    
    def contrastive_loss(self, q: torch.Tensor, p: torch.Tensor, temperature: float = 0.07, mode: str = "sim-cse") -> torch.Tensor:
        if mode == "sim-cse":
            sim = q @ p.T / temperature          # (B, B)
            labels = torch.arange(q.size(0), device=q.device)
            return nn.CrossEntropyLoss()(sim, labels)
        
        elif mode == "clip":
            def cross_entropy(preds, targets, reduction='none'):
                log_softmax = nn.LogSoftmax(dim=-1)
                loss = (-targets * log_softmax(preds)).sum(1)
                if reduction == "none":
                    return loss
                elif reduction == "mean":
                    return loss.mean()

            
            logits = q @ p.T / temperature
            q_sim = q @ q.T
            p_sim = p @ p.T               
            targets = F.softmax((q_sim + p_sim) / 2 * temperature, dim=-1)

            q_loss = cross_entropy(logits, targets, reduction="none")
            p_loss = cross_entropy(logits.T, targets.T, reduction="none")
            loss = (q_loss + p_loss) / 2
            return loss.mean()
        else:
            raise ValueError(f"We currenlty only support 'sim-cse' and 'clip' modes, not {mode}.")

    def save(self, out_dir: str | Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        torch.save(self.proj.state_dict(), out_dir / "proj.pt")

    @classmethod
    def load(cls, ckpt_path: str | Path, proj_dim=256, multiplier=1) -> "QueryEncoder":
        import json
        ckpt_path = Path(ckpt_path)
        if ckpt_path.exists():
            config_path = ckpt_path / "config.json"
            if not config_path.exists():
                raise ValueError(f"Checkpoint {ckpt_path} does not contain a config.json file.")
            config = json.load(config_path.open())
            proj_dim = config.get("proj_dim", 256)
        enc = cls(str(ckpt_path), proj_dim=proj_dim, proj_multiplier=multiplier)
        proj_path = Path(ckpt_path) / "proj.pt"
        if proj_path.exists():
            enc.proj.load_state_dict(torch.load(proj_path, map_location=enc.device))
        return enc