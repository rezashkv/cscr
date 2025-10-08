import os
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from .cost_models import compute_cost
from .registry import REGISTRY
import logging

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_descriptors(desc_dir: str | Path, verbose: bool = False,
                      pool: list[str] = None) -> tuple[list[np.ndarray], list[str]]:
    X = []
    labels = []
    for root, _, files in os.walk(desc_dir):
        for file in files:
            if file.endswith('.npy'):
                if pool is not None and os.path.basename(file)[:-4] not in pool:
                    if verbose:
                        print(f"skipping {file} as it's not in {pool}")
                    continue
                filepath = os.path.join(root, file)
                if verbose:
                    print(f"Loading descriptor from {filepath}")
                try:
                    desc = np.load(filepath)
                    logging.info(f"Loaded descriptor from {filepath}, norm: {np.linalg.norm(desc)}")
                    X.append(desc)
                    label = os.path.splitext(file)[0]
                    labels.append(label)
                except Exception as e:
                    if verbose:
                        print(f"Failed to load {filepath}: {e}")
    
    return X, labels

def load_probes(paths: str | Path | List[str] | List[Path], load_ids=True) -> List[str]:
    if not isinstance(paths, list):
        if isinstance(paths, str):
            paths = [paths]
    all_probes = []
    for path in paths:
        path = Path(path)
        with path.open() as f:
            probes = json.load(f)
            if not load_ids:
                probes = [p["prompt"] for p in probes]
                
        all_probes.extend(probes)
    
    return all_probes

def collect_counts(router, encoder, dataset, name, mode, dataset_name=None, cost_type="latency"):
    counts = defaultdict(lambda: defaultdict(int))
    if mode == "full":
        from router.cost_models import compute_cost
        import time
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import re

        model_cache = {}
        tokenizer_cache = {}

        def get_model(name):
            if name not in model_cache:
                model_cache[name] = AutoModelForCausalLM.from_pretrained(_REGISTRY[name]['hf_id']).to('cuda').eval()
                tokenizer_cache[name] = AutoTokenizer.from_pretrained(_REGISTRY[name]['hf_id'])
            return model_cache[name], tokenizer_cache[name]

        def aime_exact_match(pred, gold):
            pattern = r"\\boxed\{(.+?)\}"
            matches = re.findall(pattern, pred)
            pred_val = int(matches[-1].strip()) if matches else None
            return pred_val == gold.strip()
        stats = defaultdict(lambda: {'n': 0, 'acc': 0, 'lat': 0.0, 'cost': 0.0})

    for ex in dataset:
        if mode == "full":
            if dataset_name == "embedllm":
                prompt = ex["prompt"]
                difficulty = ex["category"]
            elif dataset_name == "aime":
                prompt = ex.get("Problem") 
                difficulty = ex.get("level", "unknown")
            else:
                prompt = ex.get("Problem") 
                difficulty = ex.get("level", "unknown")
        else:
            if dataset_name == "embedllm":
                prompt = ex["prompt"]
                difficulty = ex["category"]
            else:
                prompt = ex.get("question_content") or ex.get("question")
                difficulty = ex.get("difficulty", "unknown")
        tokens = len(prompt.split())  # proxy

        emb = encoder.encode(prompt) if encoder else np.random.randn(256).astype(
            np.float32
        )
        label = router.route(emb, n_tokens=tokens, cost_type=cost_type)[0]
        counts[difficulty][label] += 1

        if mode == "full":
            if dataset_name == "embedllm":
                acc = 1 if ex["label_map"].get(label,0)==1 else 0
                stats[difficulty]['n'] += 1
                stats[difficulty]['acc'] += acc
                # cost
                stats[difficulty]['cost'] += compute_cost(label, 0, cost_type=cost_type)
                # Send real reward to bandit
                if hasattr(router, "register_feedback"):
                    router.register_feedback(label, accuracy=float(acc), cost=compute_cost(label, 0, cost_type=cost_type))
            else:
                gold = ex["Answer"]
                m, t = get_model(label)
                t0 = time.time()
                out = m.generate(**t(prompt, return_tensors='pt').to('cuda'),
                                 max_new_tokens=128000)
                latency = time.time() - t0
                text = t.decode(out[0], skip_special_tokens=True)
                ok = aime_exact_match(text, gold)
                stats[difficulty]['n'] += 1
                stats[difficulty]['acc'] += ok
                stats[difficulty]['lat'] += latency
                stats[difficulty]['cost'] += compute_cost(label, out.shape[-1], cost_type=cost_type)
                if hasattr(router, "register_feedback"):
                    router.register_feedback(label, accuracy=float(ok), cost=compute_cost(label, out.shape[-1], cost_type=cost_type))
                m.to("cpu")

    if mode == "full":
        return counts, stats
    else:
        return counts

def tabulate(name, counts):
    print(f"\n=== {name} ===")
    print("Difficulty\tExpert\tCount\t%")
    for diff, sub in sorted(counts.items()):
        total = sum(sub.values())
        for label, cnt in sorted(sub.items(), key=lambda x: -x[1]):
            pct = 100 * cnt / total
            print(f"{diff}\t{label}\t{cnt}\t{pct:.1f}%")

def tabulate_full(stats):
    print("\n=== Full Evaluation Metrics ===")
    print("Difficulty\tCount\tAccuracy\tLatency(s)\tCost")
    for diff, vals in sorted(stats.items()):
        n = vals['n']
        acc = vals['acc'] / n if n > 0 else 0.0
        lat = vals['lat'] / n if n > 0 else 0.0
        cost = vals['cost'] / n if n > 0 else 0.0
        print(f"{diff}\t{n}\t{acc:.3f}\t{lat:.3f}\t{cost:.3f}")

def avg_acc_vs_avg_cost(stats):
    print("\n=== Average Accuracy vs Average Cost ===")
    avg_cost = 0.0
    avg_acc = 0.0
    total = 0
    for diff, vals in stats.items():
        n = vals['n']
        acc = vals['acc'] / n if n > 0 else 0.0
        cost = vals['cost'] / n if n > 0 else 0.0
        avg_cost += cost * n
        avg_acc += acc * n
        total += n
    avg_cost /= total
    avg_acc /= total
    print(f"\nAverage Cost: {avg_cost:.3f}")
    print(f"Average Accuracy: {avg_acc:.3f}")

def eval_router(router, data, name, candidates=None, cost_type="usd", quiet=False,
                lambda_coeff=None, return_per_prompt: bool = False, **kwargs):
    counts = defaultdict(int)
    stats = {"n": 0, "acc": 0.0, "cost": 0.0}
    per_prompt_correct = []  # NEW
    per_prompt_pred = []     # optional, can be handy

    for ex in data:
        prompt = ex["prompt"]
        tokens = len(prompt.split())
        label = router.route(
            prompt, candidates=candidates, n_tokens=tokens,
            label_map=ex["label_map"], lambda_coeff=lambda_coeff, **kwargs
        )
        counts[label] += 1
        acc = ex["label_map"].get(label, 0)
        stats["acc"] += acc
        stats["cost"] += compute_cost(label, 0, cost_type=cost_type)
        stats["n"] += 1

        # NEW: collect per-prompt outputs
        if return_per_prompt:
            try:
                per_prompt_correct.append(1 if float(acc) >= 0.5 else 0)
            except Exception:
                per_prompt_correct.append(int(bool(acc)))
            per_prompt_pred.append(label)

        if hasattr(router, "register_feedback"):
            router.register_feedback(
                label,
                accuracy=float(acc),
                cost=compute_cost(label, 0, cost_type=cost_type),
            )

    mean_acc = stats["acc"] / stats["n"]
    mean_cost = stats["cost"] / stats["n"]

    if not quiet:
        print(f"\n=== {name} ===")
        print("Selections:")
        for m, c in counts.items():
            print(f"  {m:50s}  {c}")
        print(f"Average-accuracy: {mean_acc:.3f} | Average-cost(units): {mean_cost:.3f}")

    out = {"avg_acc": mean_acc, "avg_cost": mean_cost}
    if return_per_prompt:
        out["per_prompt_correct"] = per_prompt_correct
        out["per_prompt_pred"] = per_prompt_pred
    return out

def load_cost_dict(candidates, cost_type="n_params"):
    cost = {}
    for candidate in candidates:
        if candidate not in REGISTRY:
            raise ValueError(f"Candidate {candidate} not found in registry.")
        
        cost[candidate] = compute_cost(
            candidate,
            0,
            cost_type=cost_type,
        )
    return cost

def plot_deferral_curves(
    curves: Dict[str, Iterable[Tuple[float, float]]],
    global_min: float,
    global_max: float,
    save_path: str | None = None,
    *,
    show: bool = True,
) -> None:
    """Render a minimalistic and aesthetically pleasing deferral‐curve plot."""

    # --- Styling -----------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")  # clean white background w/ subtle grid

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    # --- Plot each router's curve -----------------------------------------
    for router_name, curve in curves.items():
        costs, accs = zip(*curve)
        ax.plot(
            costs,
            accs,
            marker="o",
            markersize=4,
            linewidth=1.3,
            label=router_name.upper(),
        )

    # --- Axis labels & limits ---------------------------------------------
    ax.set_xlabel("Average cost", labelpad=6)
    ax.set_ylabel("Average accuracy", labelpad=6)
    ax.set_xlim(global_min, global_max)

    # --- Title & legend ----------------------------------------------------
    ax.set_title("Deferral curves", pad=10, fontsize=12, weight="bold")
    ax.legend(frameon=False, fontsize=8)

    # --- Cosmetic tweaks ---------------------------------------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    fig.tight_layout()

    # --- Save / show -------------------------------------------------------
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved ➜ {save_path}")

    if show:
        plt.show()

def load_model_and_tokenizer(model_name):
    if "flan-t5" in model_name:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",   
        )
    elif model_name == "TheBloke/WizardLM-13B-V1.2-GGUF":
        from transformers import AutoModel
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            model_file="wizardlm-13b-v1.2.q4_K_M.gguf",
            model_type="llama",
        )
    elif model_name == "SUSTech/SUS-Chat-72B":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto",
        )
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        except AttributeError:
            if "moss-moon" in model_name:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    revision="refs/pr/6", 
                )
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

        if "chatglm" in model_name:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            
    return model, tokenizer