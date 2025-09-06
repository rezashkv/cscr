import json

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from router.utils import load_probes

def _get_embedding(model, input_ids: torch.Tensor) -> torch.Tensor:
    return model.get_input_embeddings()(input_ids)

def compute_gradient_descriptor(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    probes: List[str],
    proj_dim: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 4,
) -> np.ndarray:
    """
    Gradient fingerprint (Fisher style):
    • For each prompt, compute the grad of log‑prob of the next token w.r.t input embeddings.
    • Average squared‑grad across tokens, then PCA‑project (random Gaussian proj) to proj_dim dimensions.
    """
    tokenizer.pad_token = tokenizer.eos_token
    hid_size = model.config.hidden_size
    rand_proj = torch.randn(hid_size, proj_dim, device=device)
    grad_accum = torch.zeros(proj_dim, device=device)

    for start in tqdm(range(0, len(probes), batch_size), desc="grad‑fp"):
        batch_prompts = probes[start: start+batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        input_ids = enc["input_ids"]
        input_emb = _get_embedding(model, input_ids).detach().clone().requires_grad_(True)

        out = model(inputs_embeds=input_emb, attention_mask=enc["attention_mask"])
        # use last token logits
        logits = out.logits[:, -1]  # (B, vocab)
        target = logits.argmax(dim=-1)
        logprob = F.log_softmax(logits, dim=-1)
        loss = -logprob[torch.arange(logprob.size(0), device=device), target].mean()
        loss.backward()

        g = input_emb.grad     # (B, L, H)
        g = (g ** 2).mean(dim=1) @ rand_proj   # (B, proj_dim)
        grad_accum += g.mean(dim=0)

    descriptor = grad_accum.cpu().numpy()
    descriptor /= np.linalg.norm(descriptor) + 1e-12
    return descriptor.astype(np.float32)

def _get_shared_vocab_topk(tokenizer, probes: List[str], k: int = 256) -> List[int]:
    freq = {}
    for prompt in probes:
        ids = tokenizer(prompt).input_ids
        for tid in ids:
            freq[tid] = freq.get(tid, 0) + 1
    topk = sorted(freq, key=freq.get, reverse=True)[:k]
    return topk

def compute_logit_descriptor(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    probes: List[str],
    top_token_ids: List[int] | None = None,
    top_k: int = 256,
    n_tokens: int = 10,
    batch_size: int = 8,
) -> Tuple[np.ndarray, List[int]]:
    """Return (descriptor, token_ids_used).

    *descriptor* ∈ ℝ^{top_k} is the mean softmax prob for each selected token.
    If *top_token_ids* is not supplied, we derive it from probe token
    frequencies on the fly and return it for reuse by other experts.

    multi-token version: average softmax probs over the first *n_tokens* generated
    tokens (greedy sampling).  This is a bit slower but captures more
    information about the expert's behaviour.
    """
    tokenizer.pad_token = tokenizer.eos_token 
    if top_token_ids is None:
        top_token_ids = _get_shared_vocab_topk(tokenizer, probes, k=top_k)

    token_to_idx = {tid: i for i, tid in enumerate(top_token_ids)}
    probs_accum = np.zeros(len(top_token_ids), dtype=np.float32)
    n_total = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for start in tqdm(range(0, len(probes), batch_size), desc="descriptors"):
            batch_prompts = probes[start : start + batch_size]
            enc = tokenizer(batch_prompts, return_tensors="pt", padding=True if batch_size > 1 else False).to(device)
            gen = model.generate(**enc, max_new_tokens=n_tokens, do_sample=False, output_scores=True,
                                 return_dict_in_generate=True)
            logits = torch.stack(gen.scores, dim=1)  # (B, T, vocab) 
            probs = torch.softmax(logits, dim=-1)

            for b in range(probs.size(0)):
                for tid, idx in token_to_idx.items():
                    probs_accum[idx] += probs[b, :, tid].mean().item()
                n_total += 1

    descriptor = probs_accum / n_total
    # L2 normalise for cosine similarity use‑case
    norm = np.linalg.norm(descriptor) + 1e-12
    descriptor /= norm
    return descriptor.astype(np.float32), top_token_ids

def save_descriptors(model, tokenizer, probes, out, topk=256, n_tokens=10, batch_size=8):

    probes = load_probes(probes)
    descriptor, token_ids = compute_logit_descriptor(model, tokenizer, probes, top_k=topk, n_tokens=n_tokens,
                                                      batch_size=batch_size)

    out_path = Path(f"{out}.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, descriptor)

    # Also save token ids for consistency across experts
    with open(out_path.with_suffix(".tokens.json"), "w") as f:
        json.dump(token_ids, f)

    print(f"saved descriptor → {out_path} (dim={descriptor.size})")


__all__ = [
    "save_descriptors",
]