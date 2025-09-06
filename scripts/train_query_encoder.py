import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from router.query_encoder import QueryEncoder
from router.embedllm import EmbedLLMOracle
from router.mix_instruct import MixInstructOracle
from router.routerbench import RouterBenchOracle
from router.utils import load_descriptors, load_cost_dict

import logging
logging.basicConfig(level=logging.INFO)


def collate(batch, tokenizer, device):
    texts, idxs, costs = zip(*batch)
    toks = tokenizer(
        list(texts), padding=True, truncation=True, return_tensors="pt", max_length=256
    )
    return toks, torch.tensor(idxs, dtype=torch.long), torch.stack(costs) if costs[0] is not None else None


def get_best_expert(label: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    B, N = label.shape

    cost_matrix = cost.unsqueeze(0).expand(B, N)
    masked_cost = torch.where(label == 1, cost_matrix, float('inf'))
    
    best = torch.argmin(masked_cost, dim=1)

    no_valid = (label.sum(dim=1) == 0)
    best[no_valid] = -1

    return best


def vanilla_contrastive_loss(q, E, tgt, tau):
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    logits = (q @ E.T) / tau
    loss = loss_fn(logits, tgt)
    return loss


def pam_info_nce(z_q, E, label, cost_norm,
                 tau, lam=0.5, mu=0.5, beta=0.5):

    sim = z_q @ E.T
    u   = label - lam * cost_norm           # (B,M)
    pos_mask = u > 0                        # (B,M)

    # keep only rows with ≥1 positive
    keep = pos_mask.any(dim=1)
    if keep.sum() == 0:
        return torch.tensor(0., device=z_q.device, requires_grad=True)

    sim, u, label = sim[keep], u[keep], label[keep]
    pos_mask = pos_mask[keep]

    tau_pos = beta * (u[pos_mask].std() + 1e-4)
    w = torch.exp((u / tau_pos).masked_fill(~pos_mask, -1e9))
    w = w / (w.sum(dim=1, keepdim=True) + 1e-9)      # ε avoids 0-div

    neg_pen = -mu * (1 - label) * (1 + cost_norm)
    logits  = (sim + neg_pen) / tau

    numer = (w * torch.exp(sim / tau)).sum(dim=1)
    denom = torch.exp(logits).sum(dim=1)

    return -(numer / denom).log().mean()


def cost_spectrum_info_nce(
        z_q: torch.Tensor,          # (B,d) query reps
        E: torch.Tensor,            # (M,d) expert reps
        label: torch.Tensor,        # (B,M) 0/1 correctness
        cost_norm: torch.Tensor,    # (M,) 
        tau: float = 0.07,
        n_bands: int = 5,
        alpha: float = 0.25,
        tau_min: float = 0.05,
        gamma: float = 0.2
    ) -> torch.Tensor:
    """
    Implements the Cost‑Spectrum InfoNCE described in CS‑InfoNCE:
      • Partition experts into `n_bands` percentile bands of cost_norm.
      • Treat the *cheapest correct* expert in each band as a positive.
      • Band‑specific temperature   τ_b = τ_min + alpha * cost_centroid.
      • Negatives are all other experts; the denominator is cost‑penalized by γ.
    """
    device = z_q.device
    B, M = label.shape


    percentiles = torch.linspace(0, 1, n_bands + 1, device=device)
    cost_bins   = torch.quantile(cost_norm.view(-1), percentiles)
    band_idx    = torch.bucketize(cost_norm, cost_bins[1:-1])   # (M,)

    sim = (z_q @ E.T)                                  # (B,M)

    loss_accum, band_cnt = 0.0, 0
    for k in range(n_bands):
        b_mask = (band_idx == k)                                # (M,)
        if b_mask.sum() == 0:
            continue

        pos_mask = label.clone()                                 # (B,M)
        pos_mask[:, ~b_mask] = 0
        
        any_pos = pos_mask.any(1)
        if any_pos.sum() == 0:
            continue
        sim_k   = sim[any_pos]                                  # (b,M)
        pos_k   = pos_mask[any_pos]
        neg_k   = (~pos_k)

        # band‑specific temperature
        tau_b   = tau_min + alpha * cost_norm[b_mask].mean()

        # Numerator: standard band-softmax over positives (no cost penalty)
        exp_pos = torch.exp(sim_k / tau_b)                      # (b,M)
        numer = (exp_pos * pos_k).sum(1)

        # Denominator: all experts, cost-penalized by gamma
        cost_pen = gamma * cost_norm.unsqueeze(0)               # (1,M)
        logits_k = (sim_k - cost_pen) / tau_b                   # (b,M)
        denom = torch.exp(logits_k).sum(1)

        loss_accum += -(numer / (denom + 1e-9)).log().mean()
        band_cnt += 1

    if band_cnt == 0:
        return torch.tensor(0., device=device, requires_grad=True)

    return loss_accum / band_cnt

# Cost‑aware InfoNCE – positives down‑weighted by cost, denominator cost‑penalized
def cost_info_nce(z_q, E, label, cost_norm, tau=0.07, lam=0.5):
    """
    z_q       : (B, d)   query embeddings
    E         : (M, d)   expert descriptors (fixed)
    label     : (B, M)   binary oracle correctness
    cost_norm : (M)     normalized cost in [0,1]
    tau       : temperature
    lam       : cost penalty strength in the denominator
    """
    sim = (z_q @ E.T) / tau                          # (B,M)
    pos_mask = label.bool()                          # (B,M)

    # Skip batches with no positive experts
    keep = pos_mask.any(dim=1)
    if keep.sum() == 0:
        return torch.tensor(0., device=z_q.device, requires_grad=True)

    sim, pos_mask = sim[keep], pos_mask[keep]

    w_pos = (1.0 - cost_norm).unsqueeze(0)           # (1,M)
    w_pos = (w_pos * pos_mask).masked_fill(~pos_mask, 0.)
    # avoid divide‑by‑0
    w_pos = w_pos / (w_pos.sum(dim=1, keepdim=True) + 1e-9)

    numer = (torch.exp(sim) * w_pos).sum(dim=1)      # (B,)

    denom = torch.exp(sim - lam * cost_norm).sum(dim=1)  # (B,)

    loss = -(numer / denom).clamp(min=1e-9).log().mean()
    return loss

def per_prompt_cost_info_nce(q, E, label, cost_norm, tau=0.07, lam=0.5):
    """
    q           (B,d)
    E           (M,d)
    label       (B,M) 0/1
    cost_norm   (B,M) normalized [0,1] **per prompt**
    """
    sim = (q @ E.T) / tau                         # (B,M)
    pos_mask = label.bool()                       # (B,M)

    masked_cost = torch.where(pos_mask, cost_norm, 2.0)  # >1 means “ignore”
    pos_idx = masked_cost.argmin(1)                       # (B,)
    valid   = pos_mask.gather(1, pos_idx[:,None]).squeeze(1)
    q, sim, pos_idx, cost_norm = q[valid], sim[valid], pos_idx[valid], cost_norm[valid]
    if q.size(0) == 0:
        return torch.tensor(0., device=q.device, requires_grad=True)

    pos_logits = sim.gather(1, pos_idx[:,None]).squeeze(1)      # (b,)
    pos_weight = 1. - cost_norm.gather(1, pos_idx[:,None]).squeeze(1)

    numer = torch.exp(pos_logits) * pos_weight                  # (b,)
    denom = torch.exp(sim - lam * cost_norm).sum(1)             # (b,)

    return -(numer / (denom + 1e-9)).log().mean()

def capability_spectrum_info_nce(
        z_q: torch.Tensor,       # (B, d)
        E: torch.Tensor,         # (M, d)
        label: torch.Tensor,     # (B, M) 0/1 correctness
        cost_norm: torch.Tensor, # (M,)  [0,1]
        tau: float = 0.07,
        synth_frac: float = 0.25,
        noise_std: float = 0.05
    ) -> torch.Tensor:
    """
    Capability‑spectrum InfoNCE.

    • Positive = cheapest correct expert  h⁺  for each query.
    • Negatives = experts that are strictly more expensive than h⁺.
    • To teach robustness to unseen descriptors, with probability `synth_frac`
      we append one synthetic negative per query obtained by adding
      N(0, noise_std²) to a randomly chosen expert descriptor.
    """
    device = z_q.device
    B, M = label.shape

    
    cost_mat = cost_norm.unsqueeze(0).expand(B, M)          # (B,M)
    masked_cost = torch.where(label.bool(), cost_mat, 1e9)  # inf for wrong
    pos_idx = masked_cost.argmin(dim=1)                     # (B,)
    valid   = label.gather(1, pos_idx.unsqueeze(1)).squeeze(1).bool()
    z_q, pos_idx = z_q[valid], pos_idx[valid]
    if z_q.size(0) == 0:
        return torch.tensor(0., device=device, requires_grad=True)
    
    sim = (z_q @ E.T) / tau                                 # (b,M)
 
    pos_cost = cost_norm[pos_idx]                           # (b,)
    neg_mask = cost_norm.unsqueeze(0) > pos_cost.unsqueeze(1)  # (b,M)

    if synth_frac > 0:
        b = z_q.size(0)
        synth_mask = torch.rand(b, device=device) < synth_frac
        if synth_mask.any():
            rand_j  = torch.randint(0, M, (synth_mask.sum(),), device=device)
            synth_E = E[rand_j] + noise_std * torch.randn_like(E[rand_j])
            
            E_ext   = torch.cat([E, synth_E], dim=0)             # (M+S,d)
            new_sim = (z_q[synth_mask] @ synth_E.T).diag() / tau # (S,)
            pad = torch.zeros(b, synth_mask.sum(), device=device) - 1e9
            sim = torch.cat([sim, pad], dim=1)
            sim[synth_mask, M + torch.arange(synth_mask.sum(), device=device)] = new_sim
            neg_mask = torch.cat([neg_mask, torch.zeros_like(pad, dtype=torch.bool)], dim=1)
            M = E_ext.size(0)
            E = E_ext

    pos_logits = sim.gather(1, pos_idx.unsqueeze(1)).squeeze(1)   # (b,)
    numer = torch.exp(pos_logits) 
    denom = torch.exp(sim.masked_fill(~neg_mask, -1e9)).sum(dim=1) + numer

    loss = -(numer / denom).clamp(min=1e-9).log().mean()
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dataset", choices=["embedllm", "mix-instruct", "routerbench"], default="embedllm",
                        help="Which supervision dataset to use")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--unfreeze_lm", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cost_type", choices=["latency", "n_params", "usd"], default="n_params",
                    help="Cost type to compute")
    parser.add_argument("--pool", type=str, default=None)
    parser.add_argument(
        "--contrastive_loss_type",
        choices=["best_expert", "pam_info_nce", "cost_info_nce", "cost_spectrum_info_nce", "capability_spectrum_info_nce"],
        default="cost_info_nce",
        help="Which loss to use for training the query encoder"
    )
    parser.add_argument(
        "--descriptor_aug",
        choices=["none", "dropout", "gaussian", "mixup"],
        default="none",
        help="Descriptor augmentation mode for robustness to unseen experts"
    )
    parser.add_argument("--dropout_p", type=float, default=0.15,
        help="Drop‑out probability for descriptor dropout augmentation")
    parser.add_argument("--noise_std", type=float, default=0.05,
        help="Std of Gaussian noise for descriptor gaussian augmentation")
    parser.add_argument("--filter_prompt_cat", default=None, type=str,
        help="path to the json file to filter out prompts with certain categories in the dataset.")
    
    parser.add_argument("--n_bands", type=int, default=None,
        help="Number of cost bands for cost spectrum loss")
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="Negative cost penalty (γ) for cost_spectrum_info_nce denominator")
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="Band slope α in τ_k = τ_min + α * cost_centroid for cost_spectrum_info_nce")
    parser.add_argument("--tau_min", type=float, default=0.05,
                        help="Base temperature τ_min (cheapest band) for cost_spectrum_info_nce")
    
    args = parser.parse_args()
    if args.pool is not None:
        with open(args.pool) as f:
            pool = json.load(f)
    else:
        pool = None
    
    E, desc_names = load_descriptors(args.desc_dir, pool=pool)

   
    E = torch.from_numpy(np.stack(E))
    E = E.to(args.device)
    E = E / (E.norm(dim=1, keepdim=True) + 1e-9)
    logging.info(E.norm())

    def augment_descriptors(E: torch.Tensor,
                            mode: str = "none",
                            p: float = 0.15,
                            sigma: float = 0.05) -> torch.Tensor:
        if mode == "none":
            return E
        if mode == "dropout":
            mask = (torch.rand_like(E) > p).float()
            return E * mask
        if mode == "gaussian":
            noise = torch.randn_like(E) * sigma
            return E + noise
        if mode == "mixup":
            idx = torch.randperm(E.size(0), device=E.device)
            lam = torch.rand(E.size(0), 1, device=E.device)
            return lam * E + (1 - lam) * E[idx]
        raise ValueError(f"Unknown aug mode {mode}")

   
    encoder = QueryEncoder(args.model_name, device=args.device, proj_dim=E.size(1))

    if args.dataset == "embedllm":
        ds = EmbedLLMOracle(desc_names, split="train", filter_prompt_cat=args.filter_prompt_cat)
        logging.info(f"Loaded {len(ds)} prompts from {args.filter_prompt_cat}")
    elif args.dataset == "mix-instruct":
        ds = MixInstructOracle(desc_names, split="train")
    else:
        ds = RouterBenchOracle(desc_names, split="train")
    
    dummy_tokenizer = encoder.tokenizer
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, dummy_tokenizer, args.device),
    )

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.lr)
    tau = args.temperature
    encoder.model.train()

    if args.unfreeze_lm:
        logging.info("Unfreezing LM parameters")
        for p in encoder.model.parameters():
            p.requires_grad = True
    else:
        logging.info("Freezing LM parameters")
        for p in encoder.model.parameters():
            p.requires_grad = False
        for p in encoder.proj.parameters():
            p.requires_grad = True
    
    nuum_total_params = sum(p.numel() for p in encoder.parameters())
    nuum_trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logging.info(f"{nuum_trainable_params} / {nuum_total_params} trainable parameters")
    
    cost = load_cost_dict(desc_names, cost_type=args.cost_type)
    cost_tensor = torch.tensor([cost[name] for name in desc_names])
    cost_tensor = (cost_tensor - cost_tensor.min()) / \
        (cost_tensor.max() - cost_tensor.min() + 1e-9)

    per_prompt_cost = False
    for ep in range(args.epochs):
        for step, (tok, label, cost) in enumerate(loader):
    
            tok = tok.to(args.device)
            if args.unfreeze_lm:
                cls = encoder.model(**tok).last_hidden_state[:, 0]
            else:   
                with torch.no_grad():
                    cls = encoder.model(**tok).last_hidden_state[:, 0]

            q = encoder._project(cls)
            E_aug = augment_descriptors(E, args.descriptor_aug,
                                        p=args.dropout_p,
                                        sigma=args.noise_std)

            if args.contrastive_loss_type == "best_expert":
                tgt = get_best_expert(label, cost_tensor).to(args.device)
                loss = vanilla_contrastive_loss(q, E_aug, tgt, tau)

            elif args.contrastive_loss_type == "pam_info_nce":
                label = label.to(args.device)
                c_norm = cost_tensor.to(args.device)
                loss = pam_info_nce(q, E_aug, label, c_norm,
                                    tau=tau, lam=0.5, mu=0.5, beta=0.5)

            elif args.contrastive_loss_type == "cost_info_nce":
                label = label.to(args.device)
                c_norm = cost_tensor.to(args.device)
                if per_prompt_cost:
                    loss  = per_prompt_cost_info_nce(q, E_aug, label, c_norm,
                                                     tau=tau, lam=0.5)
                else:
                    loss  = cost_info_nce(q, E_aug, label, c_norm,
                                        tau=tau, lam=0.5)

            elif args.contrastive_loss_type == "cost_spectrum_info_nce":
                label = label.to(args.device)
                c_norm = cost_tensor.to(args.device)
                if len(c_norm.shape) == 2:
                    c_norm = c_norm.mean(dim=0)

                loss  = cost_spectrum_info_nce(
                            q, E_aug, label, c_norm,
                            tau=tau,
                            alpha=args.alpha,
                            tau_min=args.tau_min,
                            gamma=args.gamma,
                            n_bands=int(E.size(0) ** 0.5) if args.n_bands is None else args.n_bands
                        )

            elif args.contrastive_loss_type == "capability_spectrum_info_nce":
                label = label.to(args.device)
                c_norm = cost_tensor.to(args.device)
                loss = capability_spectrum_info_nce(q, E_aug, label, c_norm,
                                                    tau=tau, synth_frac=0.25, noise_std=0.05)
            else:
                raise ValueError(f"Unknown loss {args.contrastive_loss_type}")
   
            loss.backward()
            opt.step()
            opt.zero_grad()

            if step % 100 == 0:
                logging.info(f"epoch {ep} step {step} loss {loss.item():.4f}")
        if ep % 1 == 0:
            encoder.save(f"{args.out_dir}/checkpoint-{ep}/")
            logging.info(f"Saved checkpoint to {args.out_dir}/checkpoint-{ep}")
    encoder.save(f"{args.out_dir}/final/")
    logging.info(f"Saved oracle‑trained encoder ({args.dataset}) to {args.out_dir}")


if __name__ == "__main__":
    main()
