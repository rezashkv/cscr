from __future__ import annotations
import argparse, json, math, pickle
from pathlib import Path
from typing import Callable, List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

from router.utils import eval_router, load_descriptors
from router.embedllm import load_embedllm
import os
import time

from router.mix_instruct import load_mixinstruct
from router.routerbench import load_routerbench

from router.baselines import get_baseline

from scipy.interpolate import interp1d
from scipy.stats import binomtest


def build_cost_grid(all_costs, N_grid=50):
    min_c, max_c = all_costs.min(), all_costs.max()
    return np.linspace(min_c, max_c, N_grid)

def interp_to_grid(costs, accs, grid):
    """Piecewise‑linear interpolation; extrapolate w/ edge values."""
    f = interp1d(costs, accs,
                 kind="linear",
                 bounds_error=False,
                 fill_value=(accs[0], accs[-1]))
    return f(grid)

def compute_per_prompt_matrices(factory: Callable[[float], object],
                                lambdas: List[float],
                                dataset: List[Dict],
                                cost_type: str,
                                candidates: List[str] = None):
    """
    Precompute (per-λ) average cost and per-prompt correctness vectors so that
    bootstrap resampling can be done without re-evaluating the router.
    Returns (costs: [T], Y: [T, N]) where Y[t] is the per-prompt correctness at λ_t,
    or (None, None) if eval_router does not support return_per_prompt=True.
    """
    costs = []
    rows = []
    for lam in lambdas:
        router = factory(lam)
        try:
            metrics = eval_router(router, dataset,
                                  name=f"{lam:.3f}",
                                  cost_type=cost_type,
                                  candidates=candidates,
                                  quiet=True,
                                  lambda_coeff=lam,
                                  return_per_prompt=True)
        except TypeError:
            print("[FAST-BOOTSTRAP] eval_router(...) lacks return_per_prompt=True support; falling back to slow path.")
            return None, None
        y = extract_per_prompt_correct(metrics)  # [N]
        c = metrics["avg_cost"]
        if cost_type == "n_params":
            c = c * 100 / 3
        costs.append(c)
        rows.append(y.astype(int))
    costs = np.asarray(costs)
    order = np.argsort(costs)
    costs = costs[order]
    Y = np.stack(rows, axis=0)[order]  # [T, N]
    return costs, Y

def get_router_factory(router_name: str, args) -> Callable[[float], object]:
    """
    Generic factory builder that leverages `router.baselines.get_baseline`.
    For each λ in the sweep we instantiate a *fresh* router to avoid
    state‑sharing across λ values.
    """
    name = router_name.lower()

    common = dict(cost_type=args.cost_type)

    if name == "umr":
        kwargs = dict(work_dir=Path(args.umr_work_dir), device="cuda")
        def factory(lam: float):
            return get_baseline("umr", **kwargs)(override_lambda=lam)
    elif name == "knn":
        kwargs = dict(index_path=args.knn_index,
                      labels_path=args.knn_labels,
                      k=args.k,
                      bandit_beta=args.knn_bandit_beta,
                      encoder_ckpt=args.knn_encoder_ckpt,
                      **common)
        def factory(lam: float):
            return get_baseline("knn", **kwargs)(bandit_lambda=lam)
    elif name in {"random", "pareto_random"}:
        kwargs = dict(labels_json=args.knn_labels, **common)
        def factory(lam: float):
            return get_baseline(name, **kwargs)()
    elif name == "oracle":
        kwargs = dict(fallback_pool=json.load(open(args.knn_labels)), **common)
        def factory(lam: float):
            return get_baseline("oracle", **kwargs)()
    elif name == "parametric":
        W = np.load(args.parametric_npz)["W"]
        b = np.load(args.parametric_npz)["b"]
        kwargs = dict(W=W,
                      b=b,
                      encoder_ckpt=args.parametric_encoder_ckpt,
                     expert_embeds=args.parametric_embedding_dict,
                      **common)
        def factory(lam: float):
            return get_baseline("parametric", **kwargs)()
    elif name == "softmoe":
        kwargs = dict(encoder_ckpt=args.softmoe_encoder_ckpt,
                      tau=args.softmoe_tau,
                      top_k=args.softmoe_top_k,
                      expert_embeds=args.parametric_embedding_dict,
                      proj_dim=256 if args.dataset != "routerbench" else 192,
                      **common)
        def factory(lam: float):
            return get_baseline("softmoe", **kwargs)()
    elif name == "thompson":
        cand = json.load(open(args.knn_labels))
        kwargs = dict(candidates=cand,
                      alpha=args.thompson_alpha,
                      beta=args.thompson_beta,
                      **common)
        def factory(lam: float):
            return get_baseline("thompson", **kwargs)()
    else:
        raise ValueError(f"Unsupported router: {router_name}")

    return factory

def sweep_lambdas(factory: Callable[[float], object],
                  lambdas: List[float],
                  dataset: List[Dict],
                  cost_type: str,
                  candidates: List[str] = None) -> List[Tuple[float, float]]:

    costs, accs = [], []
    for lam in lambdas:
        router = factory(lam)
        metrics = eval_router(router, dataset,
                              name=f"{lam:.3f}",
                              cost_type=cost_type,
                              candidates=candidates,
                              quiet=True,
                              lambda_coeff=lam)  
        costs.append(metrics["avg_cost"] if cost_type=="n_params" else metrics["avg_cost"])
        accs.append(metrics["avg_acc"])

    costs, accs = np.asarray(costs), np.asarray(accs)
    order = np.argsort(costs)
    return costs[order], accs[order]

def area_under_curve(curve: List[Tuple[float, float]], global_range: tuple[float, float] | None = None) -> float:
    if len(curve) < 2:
        return 0.0

    if global_range is None:
        
        area = 0.0
        for (c0, a0), (c1, a1) in zip(curve[:-1], curve[1:]):
            area += 0.5 * (a0 + a1) * (c1 - c0)
        return area

    min_c, max_c = global_range
    if max_c <= min_c:
        return 0.0

    filtered = [(c,a) for c,a in curve if min_c <= c <= max_c]

    if not filtered:
        return 0.0

    if filtered[0][0] > min_c:
        for i in range(1, len(curve)):
            c0, a0 = curve[i-1]
            c1, a1 = curve[i]
            if c0 <= min_c <= c1:
                t = (min_c - c0) / (c1 - c0)
                a_interp = a0 + t * (a1 - a0)
                filtered.insert(0, (min_c, a_interp))
                break
        else:
            filtered.insert(0, (min_c, filtered[0][1]))

    if filtered[-1][0] < max_c:
        for i in range(1, len(curve)):
            c0, a0 = curve[i-1]
            c1, a1 = curve[i]
            if c0 <= max_c <= c1:
                t = (max_c - c0) / (c1 - c0)
                a_interp = a0 + t * (a1 - a0)
                filtered.append((max_c, a_interp))
                break
        else:
            filtered.append((max_c, filtered[-1][1]))
    area = 0.0
    for i in range(len(filtered) - 1):
        c0, a0 = filtered[i]
        c1, a1 = filtered[i+1]
        c0_norm = (c0 - min_c) / (max_c - min_c)
        c1_norm = (c1 - min_c) / (max_c - min_c)
        area += 0.5 * (a0 + a1) * (c1_norm - c0_norm)

    return area

def extract_per_prompt_correct(metrics: Dict) -> np.ndarray:
    """
    Try to extract a per-prompt correctness array from eval_router metrics.
    Accepts several common keys and binarizes with threshold 0.5.
    """
    for key in ["per_prompt_correct", "per_prompt_acc", "per_prompt_correctness"]:
        if key in metrics:
            arr = np.asarray(metrics[key])
            try:
                return (arr.astype(float) >= 0.5).astype(int)
            except Exception:
                return arr.astype(int)
    raise KeyError("eval_router did not return a per-prompt correctness array")

def paired_bootstrap_audc(factory_a,
                          factory_b,
                          lam_list: np.ndarray,
                          dataset: List[Dict],
                          cost_type: str,
                          B: int = 2000,
                          seed: int = 0) -> tuple[float, tuple[float, float], float]:
    """
    Resample prompts with replacement; recompute both methods' deferral curves
    on the resample; compute ΔAUDC = AUDC_A - AUDC_B under a common cost range.
    Returns (mean Δ, 95% CI, one-sided p-value for Δ <= 0).

    Logs progress approximately every 5% of iterations.
    """
    rng = np.random.default_rng(seed)
    N = len(dataset)
    deltas = np.empty(B, dtype=float)

    start = time.time()
    log_every = max(1, B // 20) 

    for b in range(B):
        idx = rng.integers(0, N, size=N)
        subset = [dataset[i] for i in idx]
        c_a, a_a = sweep_lambdas(factory_a, lam_list, subset, cost_type)
        c_b, a_b = sweep_lambdas(factory_b, lam_list, subset, cost_type)

        if len(c_a) < 2 or len(c_b) < 2:
            deltas[b] = 0.0
        else:
            min_c = float(min(c_a.min(), c_b.min()))
            max_c = float(max(c_a.max(), c_b.max()))
            curve_a = list(zip(c_a, a_a))
            curve_b = list(zip(c_b, a_b))
            audc_a = area_under_curve(curve_a, (min_c, max_c))
            audc_b = area_under_curve(curve_b, (min_c, max_c))
            deltas[b] = audc_a - audc_b

        if (b + 1) % log_every == 0 or b == 0 or (b + 1) == B:
            done = b + 1
            elapsed = time.time() - start
            rate = done / max(elapsed, 1e-9)
            remaining = B - done
            eta = remaining / max(rate, 1e-9)
            print(f"[Bootstrap ΔAUDC] {done}/{B} ({done/B:0.1%}) | elapsed {elapsed:.1f}s | ETA {eta:.1f}s", flush=True)

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_one_sided = (1 + (deltas <= 0).sum()) / (B + 1)
    return float(deltas.mean()), (float(lo), float(hi)), float(p_one_sided)

def select_lambda_for_cost(costs: np.ndarray, lam_list: np.ndarray, target_cost: float) -> float:
    """Pick the λ whose average cost is closest to the target_cost."""
    idx = int(np.argmin(np.abs(costs - target_cost)))
    return float(lam_list[idx])

def mcnemar_at_matched_cost(factory_a,
                            factory_b,
                            lam_list: np.ndarray,
                            dataset: List[Dict],
                            cost_type: str,
                            candidates: list[str],
                            target_mode: str = "median") -> dict[str, float] | None:
    """
    Choose a target cost (median of combined cost grids by default), match each router
    to the nearest average-cost operating point, then run McNemar's test on per-prompt
    correctness at those two operating points.
    Returns a dict with n10, n01, p, win_rate, target_cost, lam_a, lam_b.
    """
    
    c_a, a_a = sweep_lambdas(factory_a, lam_list, dataset, cost_type, candidates=candidates)
    c_b, a_b = sweep_lambdas(factory_b, lam_list, dataset, cost_type, candidates=candidates)

    combined = np.concatenate([c_a, c_b])
    if target_mode == "median":
        target_cost = float(np.median(combined))
    elif target_mode == "mean":
        target_cost = float(np.mean(combined))
    else:
        try:
            target_cost = float(target_mode)
        except Exception:
            target_cost = float(np.median(combined))

    lam_a = select_lambda_for_cost(c_a, lam_list, target_cost)
    lam_b = select_lambda_for_cost(c_b, lam_list, target_cost)

    
    router_a = factory_a(lam_a)
    router_b = factory_b(lam_b)
    try:
        met_a = eval_router(router_a, dataset,
                            name=f"McNemar_A_{lam_a:.3f}",
                            cost_type=cost_type, candidates=candidates,
                            quiet=True, lambda_coeff=lam_a, return_per_prompt=True)
        met_b = eval_router(router_b, dataset,
                            name=f"McNemar_B_{lam_b:.3f}",
                            cost_type=cost_type, candidates=candidates,
                            quiet=True, lambda_coeff=lam_b, return_per_prompt=True)
        y_a = extract_per_prompt_correct(met_a)
        y_b = extract_per_prompt_correct(met_b)
    except TypeError:
        print("[McNemar] eval_router does not support return_per_prompt=True; skipping McNemar.")
        return None
    except KeyError as e:
        print(f"[McNemar] {e}; skipping McNemar.")
        return None

    y_a = y_a.astype(int)
    y_b = y_b.astype(int)
    n10 = int(((y_a == 1) & (y_b == 0)).sum())
    n01 = int(((y_a == 0) & (y_b == 1)).sum())
    n = n10 + n01
    if n == 0:
        p = 1.0
        win_rate = 0.5
    else:
        p = float(binomtest(k=n10, n=n, p=0.5, alternative="greater").pvalue)
        win_rate = n10 / n if n > 0 else 0.5

    return {
        "target_cost": float(target_cost),
        "lam_a": lam_a,
        "lam_b": lam_b,
        "n10": n10,
        "n01": n01,
        "p": p,
        "win_rate": float(win_rate),
    }

# Fast significance utilities using cached per-prompt matrices
def paired_bootstrap_audc_cached(costs_a: np.ndarray,
                                 Y_a: np.ndarray,
                                 costs_b: np.ndarray,
                                 Y_b: np.ndarray,
                                 B: int = 2000,
                                 seed: int = 0) -> tuple[float, tuple[float, float], float]:
    """
    Fast paired bootstrap using cached per-prompt correctness matrices.
    costs_*: [T], Y_*: [T, N] where Y_[t] is per-prompt correctness at λ_t.

    Logs progress approximately every 5% of iterations.
    """
    rng = np.random.default_rng(seed)
    if costs_a is None or costs_b is None or Y_a is None or Y_b is None:
        raise ValueError("paired_bootstrap_audc_cached requires cached costs and Y matrices")

    min_c = float(min(costs_a.min(), costs_b.min()))
    max_c = float(max(costs_a.max(), costs_b.max()))
    global_range = (min_c, max_c)

    T_a, N = Y_a.shape
    T_b, N_b = Y_b.shape
    assert N == N_b, "Cached matrices must have the same number of prompts (columns)"

    deltas = np.empty(B, dtype=float)

    start = time.time()
    log_every = max(1, B // 20) 

    for b in range(B):
        idx = rng.integers(0, N, size=N)
        acc_a = Y_a[:, idx].mean(axis=1)  # [T_a]
        acc_b = Y_b[:, idx].mean(axis=1)  # [T_b]
        curve_a = list(zip(costs_a, acc_a))
        curve_b = list(zip(costs_b, acc_b))
        audc_a = area_under_curve(curve_a, global_range)
        audc_b = area_under_curve(curve_b, global_range)
        deltas[b] = audc_a - audc_b

        # progress logging
        if (b + 1) % log_every == 0 or b == 0 or (b + 1) == B:
            done = b + 1
            elapsed = time.time() - start
            rate = done / max(elapsed, 1e-9)
            remaining = B - done
            eta = remaining / max(rate, 1e-9)
            print(f"[Bootstrap ΔAUDC — fast] {done}/{B} ({done/B:0.1%}) | elapsed {elapsed:.1f}s | ETA {eta:.1f}s", flush=True)

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_one_sided = (1 + (deltas <= 0).sum()) / (B + 1)
    return float(deltas.mean()), (float(lo), float(hi)), float(p_one_sided)


def mcnemar_at_matched_cost_cached(costs_a: np.ndarray,
                                   Y_a: np.ndarray,
                                   costs_b: np.ndarray,
                                   Y_b: np.ndarray,
                                   target_mode: str = "median") -> dict[str, float]:
    """
    McNemar using cached per-prompt matrices. Picks a matched cost (median/mean or numeric),
    chooses the nearest λ row for each router, and runs exact McNemar on those two rows.
    """
    if costs_a is None or costs_b is None or Y_a is None or Y_b is None:
        raise ValueError("mcnemar_at_matched_cost_cached requires cached costs and Y matrices")

    combined = np.concatenate([costs_a, costs_b])
    if target_mode == "median":
        target_cost = float(np.median(combined))
    elif target_mode == "mean":
        target_cost = float(np.mean(combined))
    else:
        try:
            target_cost = float(target_mode)
        except Exception:
            target_cost = float(np.median(combined))

    i_a = int(np.argmin(np.abs(costs_a - target_cost)))
    i_b = int(np.argmin(np.abs(costs_b - target_cost)))
    y_a = Y_a[i_a].astype(int)
    y_b = Y_b[i_b].astype(int)

    n10 = int(((y_a == 1) & (y_b == 0)).sum())
    n01 = int(((y_a == 0) & (y_b == 1)).sum())
    n = n10 + n01
    if n == 0:
        p = 1.0
        win_rate = 0.5
    else:
        p = float(binomtest(k=n10, n=n, p=0.5, alternative="greater").pvalue)
        win_rate = n10 / n

    return {
        "target_cost": float(target_cost),
        "lam_idx_a": i_a,
        "lam_idx_b": i_b,
        "n10": n10,
        "n01": n01,
        "p": p,
        "win_rate": float(win_rate),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routers", default=["umr", "knn", "random", "oracle"],
                    nargs="+", help="Router types to evaluate")
    ap.add_argument("--umr_work_dir", default="baselines/umr/umr_artifacts_embedllm")
    ap.add_argument("--knn_index")
    ap.add_argument("--knn_labels")
    ap.add_argument("--knn_encoder_ckpt")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--knn_bandit_beta", type=float, default=0.00001)
    ap.add_argument("--cost_type", choices=["latency", "n_params", "usd"],
                    default="n_params")
    ap.add_argument("--lambda_min", type=float, default=-4)
    ap.add_argument("--lambda_max", type=float, default=2)
    ap.add_argument("--n_points", type=int, default=20,
                    help="#points in λ sweep")
    ap.add_argument("--save_curve", default=None,
                    help="Optional pickle file to dump the curve")
    ap.add_argument("--save_plot", default=None,
                    help="Path (.png/.pdf) to save the deferral‑curve figure")
    ap.add_argument("--no_show", action="store_true",
                    help="Do not display the plot (useful in headless runs)")
    ap.add_argument("--cost_grid_points", type=int, default=25,
                    help="Number of points in the cost grid for interpolation")
    ap.add_argument("--dataset", choices=["embedllm", "mix-instruct", "routerbench"], default="embedllm", help="Dataset to evaluate deferral curves on")

    ap.add_argument("--sig_pair", type=str, default=None,
                    help="Comma-separated pair of routers to compare for significance (e.g., 'knn,umr'). If omitted and >=2 routers provided, uses the first two.")
    ap.add_argument("--bootstrap_B", type=int, default=2000,
                    help="Number of bootstrap resamples for paired ΔAUDC.")
    ap.add_argument("--sig_target", type=str, default="median",
                    help="Target cost for McNemar matching: 'median', 'mean', or a numeric cost value.")

    ap.add_argument("--parametric_npz", type=str, help="Path to .npz weight vector for ParametricScorerRouter")
    ap.add_argument("--parametric_encoder_ckpt", type=str)
    ap.add_argument("--parametric_embedding_dir", type=str, default=None)

    ap.add_argument("--softmoe_encoder_ckpt", type=str, default=None)
    ap.add_argument("--softmoe_tau", type=float, default=1.0)
    ap.add_argument("--softmoe_top_k", type=int, default=1)

    ap.add_argument("--thompson_alpha", type=float, default=1.0)
    ap.add_argument("--thompson_beta", type=float, default=1.0)

    ap.add_argument("--filter_prompt_cat", default=None, type=str,
    help="path to the json file to filter out prompts with certain categories in the dataset.")

    args = ap.parse_args()

    
    POOL = json.load(open(args.knn_labels)) if args.knn_labels else None
    E, labels = load_descriptors(args.parametric_embedding_dir, pool=POOL)
    
    args.parametric_embedding_dict = {k: v for k, v in zip(labels, E)}

    if args.dataset == "embedllm":
        dataset = load_embedllm("test", candidates=POOL, filter_prompt_cat=args.filter_prompt_cat)
    elif args.dataset == "mix-instruct":
        dataset = load_mixinstruct(split="test", candidates=POOL, metric_key="bartscore")
    else:
       dataset = load_routerbench("test", candidates=POOL)

    lam_list = np.logspace(args.lambda_min, args.lambda_max, args.n_points)
    

    curves = {}
    peak_acc_dict = {}
    qnc_dict = {}

    for router_name in args.routers:
        factory = get_router_factory(router_name, args)
        c, a   = sweep_lambdas(factory, lam_list, dataset, args.cost_type, candidates=POOL)
        curves[router_name] = (c, a)

        peak_idx = np.argmax(a)
        peak_acc_dict[router_name] = float(a[peak_idx])
        qnc_dict[router_name] = float(c[peak_idx])   # cost required to hit peak acc

        
        print(f"Router: {router_name}")
        for i in range(len(c)):
            print(f"  {c[i]:.4f} {a[i]:.4f}")
        print()        


    all_costs_concat = np.concatenate([v[0] for v in curves.values()])
    grid = build_cost_grid(all_costs_concat, N_grid=args.cost_grid_points)

    audc = {}
    interp_curves = {}
    for r, (c, a) in curves.items():
        a_grid = interp_to_grid(c, a, grid)
        interp_curves[r] = (grid, a_grid)
        audc[r] = np.trapz(a_grid, grid) / (grid[-1] - grid[0])

    print("=== Normalized Deferral Curves ===")
    header = f"{'Router':>10s} | {'AUDC':>7s} | {'QNC':>8s} | {'PeakAcc':>8s}"
    print(header)
    print('-' * len(header))
    for r in args.routers:
        print(f"{r:>10s} | {audc[r]:7.4f} | {qnc_dict[r]:8.3f} | {peak_acc_dict[r]:8.3f}")

    if args.sig_pair:
        try:
            rA, rB = [s.strip() for s in args.sig_pair.split(",")]
        except Exception:
            rA, rB = None, None
    else:
        rA, rB = (args.routers[0], args.routers[1]) if len(args.routers) >= 2 else (None, None)

    if rA and rB:
        print(f"\n=== Significance between {rA} (A) and {rB} (B) ===")
        factory_a = get_router_factory(rA, args)
        factory_b = get_router_factory(rB, args)

        
        costs_a, Y_a = compute_per_prompt_matrices(factory_a, lam_list, dataset, args.cost_type, candidates=POOL)
        costs_b, Y_b = compute_per_prompt_matrices(factory_b, lam_list, dataset, args.cost_type, candidates=POOL)

        if costs_a is not None and costs_b is not None:
            print("[FAST-BOOTSTRAP].")
            
            mean_delta, (ci_lo, ci_hi), p_boot = paired_bootstrap_audc_cached(
                costs_a, Y_a, costs_b, Y_b, B=args.bootstrap_B, seed=0
            )
            print(f"[Bootstrap ΔAUDC — fast] mean Δ = {mean_delta:.6f}, 95% CI = [{ci_lo:.6f}, {ci_hi:.6f}], one-sided p = {p_boot:.6g}")

            
            mc = mcnemar_at_matched_cost_cached(costs_a, Y_a, costs_b, Y_b, target_mode=args.sig_target)
            print(f"[McNemar @ matched cost={mc['target_cost']:.3f} — fast] "
                  f"n10={mc['n10']}, n01={mc['n01']}, win_rate={mc['win_rate']:.3f}, p={mc['p']:.3g}")
        else:
            
            print("[FAST-BOOTSTRAP] Falling back to slow significance path.")
            mean_delta, (ci_lo, ci_hi), p_boot = paired_bootstrap_audc(
                factory_a, factory_b, lam_list, dataset, args.cost_type, B=args.bootstrap_B, seed=0
            )
            print(f"[Bootstrap ΔAUDC] mean Δ = {mean_delta:.6f}, 95% CI = [{ci_lo:.6f}, {ci_hi:.6f}], one-sided p = {p_boot:.6g}")

            mc = mcnemar_at_matched_cost(factory_a, factory_b, lam_list, dataset, args.cost_type,
                                         candidates=POOL, target_mode=args.sig_target)
            if mc is not None:
                print(f"[McNemar @ matched cost={mc['target_cost']:.3f}] "
                      f"n10={mc['n10']}, n01={mc['n01']}, win_rate={mc['win_rate']:.3f}, p={mc['p']:.3g}")
    else:
        print("\n(Significance skipped: provide --sig_pair or at least two routers.)")

    
    plt.figure()
    for r, (c, a) in interp_curves.items():
        plt.plot(c, a, marker="o", label=r.upper())
    plt.xlabel("Average cost"); plt.ylabel("Average accuracy")
    plt.title("Deferral curves (common cost axis)")
    plt.legend()
    plt.grid(True, which="minor", linestyle="--")
    if args.save_plot:
        plt.savefig(args.save_plot, dpi=150)
    if not args.no_show:
        plt.show()
    if args.save_curve:
        if not os.path.exists(os.path.dirname(args.save_curve)):
            os.makedirs(os.path.dirname(args.save_curve))
        with open(args.save_curve, "wb") as f:
            pickle.dump(curves, f)
        print(f"Curve saved to {args.save_curve}")

if __name__ == "__main__": 
    main()
