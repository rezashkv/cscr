import argparse
import json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_CE_MODEL_NAME = "gpt2"
try:
    _CE_TOKENIZER = AutoTokenizer.from_pretrained(_CE_MODEL_NAME)
    _CE_MODEL     = AutoModelForCausalLM.from_pretrained(_CE_MODEL_NAME)
    _CE_MODEL.eval()
    _CE_MODEL.to("cuda" if torch.cuda.is_available() else "cpu")
except Exception as ex:
    print("⚠️  Could not load cross‑entropy fingerprint model:", ex)
    _CE_TOKENIZER = None
    _CE_MODEL     = None

def cross_entropy_fingerprint(text: str, max_len: int = 1024) -> float:
    """
    Return the average per‑token cross‑entropy (in nats) of `text`
    under the language model `_CE_MODEL`. A lower value indicates
    the text is *easier* for the LM; we typically take this scalar
    as one dimension in a descriptor vector (e.g. per‑probe CE).
    """
    if _CE_MODEL is None or _CE_TOKENIZER is None:
        return float("nan")

    with torch.no_grad():
        enc = _CE_TOKENIZER(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )
        input_ids = enc["input_ids"].to(_CE_MODEL.device)
        logits = _CE_MODEL(input_ids).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return token_loss.mean().item()

def perplexity_fingerprint(text: str, max_len: int = 1024) -> float:
    """
    Convenience wrapper that returns the perplexity of `text`
    (exp of mean cross‑entropy) under `_CE_MODEL`.
    """
    ce = cross_entropy_fingerprint(text, max_len=max_len)
    return float("inf") if np.isnan(ce) else float(np.exp(ce))


def main():
    parser = argparse.ArgumentParser(description="Compute perplexity descriptors.")
    parser.add_argument("--probe_ids", required=True,
                        help="JSON file containing list of prompt_id probes")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="Dataset split to load from hub")
    parser.add_argument("--seed", type=int, default=47, help="Random seed for reproducibility.")
    parser.add_argument("--out", required=True, help="Output dir  path for descriptors.")
    parser.add_argument("--dataset", default="routerbench", choices=["mix-instruct", "embedllm"],
                        help="Dataset to use for descriptor generation.")
    parser.add_argument("--plot", action="store_true", help="Plot cosine similarity heatmap.")
    args = parser.parse_args()

    probes = json.load(open(args.probe_ids))

    probe_ids = set([p["prompt_id"] for p in probes])
    print(f"Loaded {len(probe_ids)} probe ids")

    if args.dataset == "mix-instruct":
        ds = load_dataset("llm-blender/mix-instruct", split="validation")
        ds = ds.shuffle(seed=args.seed)
        
    elif args.dataset == "routerbench":
        ds_file = hf_hub_download(repo_id="withmartian/routerbench",
                                repo_type="dataset",
                                filename="routerbench_0shot.pkl")
        ds = load_dataset("pandas", data_files={"data": ds_file}, split="data")

        ds = ds.shuffle(seed=args.seed)

        n_total = len(ds)
        n_train = int(0.9 * n_total)
        if args.split == "train":
            ds = ds.select(range(n_train))
        else:
            ds = ds.select(range(n_train, n_total))
        print(f"Dataset {args.dataset}: {args.split} subset → {len(ds)} rows")


    if args.dataset == "routerbench":
        from router.routerbench import RouterBenchOracle
        MODEL_LIST = RouterBenchOracle.MODELS
        token_ids  = {m.replace('/','__'): [] for m in MODEL_LIST}
    else:
        from router.mix_instruct import MixInstructOracle
        MODEL_LIST = MixInstructOracle.NAME_TO_HF.keys()
        token_ids  = {MixInstructOracle.NAME_TO_HF[m]: [] for m in MODEL_LIST}

    n_models = len(MODEL_LIST)
    
    descriptor = np.zeros((n_models, len(probe_ids), 1), dtype=np.float32)

    
    probe_to_idx = {pid: idx for idx, pid in enumerate(sorted(probe_ids))}
    skipped = 0
    for row in tqdm(ds, desc="embedding responses"):
        if args.dataset == "routerbench":
            pid = row["sample_id"]
        else:
            pid = row["id"]
        if pid not in probe_to_idx:
            continue
        col_idx = probe_to_idx[pid]
        for m_idx, m_name in enumerate(MODEL_LIST):
            if args.dataset == "routerbench":
                col_key = f"{m_name}|model_response"
                if col_key not in row or row[col_key] is None:
                    skipped += 1
                    continue
                text = row[col_key]
            else:
                text = [x["text"] for x in row["candidates"] if x["model"] == m_name][0]
            vec = perplexity_fingerprint(text)
            descriptor[m_idx, col_idx] = vec
            if args.dataset == "routerbench":
                token_ids[m_name.replace('/','__')].append(row["model_response_tokens"])
            
    print(f"Finished. Missing {skipped} <pid, model> pairs.")

    descriptor = np.mean(descriptor, axis=2)
    descriptor = descriptor / np.linalg.norm(descriptor, axis=1, keepdims=True)

    if args.plot:
        from sklearn.metrics.pairwise import cosine_similarity
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd


        sim = cosine_similarity(descriptor)
        plt.figure(figsize=(6, 5))
        sns.heatmap(sim, xticklabels=MODEL_LIST, yticklabels=MODEL_LIST, annot=True, cmap="viridis")
        plt.title("Cosine Similarity Between Descriptors")
        plt.tight_layout()
        plt.savefig("cosine_similarity.png")
        plt.show()



    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for m_idx, m_name in enumerate(MODEL_LIST):
        if args.dataset == "routerbench":
            m_name = m_name.replace('/','__')
        else:
            m_name = MixInstructOracle.NAME_TO_HF[m_name]
        desc = descriptor[m_idx]
        out_name = out / f"{m_name}.npy"    

        if args.dataset == "routerbench":
            tok_ids = token_ids[m_name]
            json.dump(tok_ids, open(out_name.with_suffix(".tokens.json"), "w"))
        np.save(out_name, desc)
        
    print(f"saved descriptors → {out} (dim={descriptor.shape})")

if __name__ == "__main__":
    main()
