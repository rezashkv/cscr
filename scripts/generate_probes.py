import argparse
import json
from pathlib import Path
from typing import List

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import numpy as np


def _sample_prompts(dataset, fields: List[str], n: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n, replace=False)
    if isinstance(fields, str):
        fields = [fields]
    return [{field: dataset[int(i)][field] for field in fields} for i in indices]

def build_embedllm_probes(n: int, seed: int) -> List[str]:
    downloaded_file = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="val.csv")
    ds = load_dataset("csv", data_files=downloaded_file, split="train")
    ds = ds.filter(lambda x: x["model_id"] == 0)
    if n == -1:
        n = 2 * len(ds) // 3
    return _sample_prompts(ds, ["prompt", "prompt_id"], n, seed)

def build_mix_instruct_probes(n: int, seed: int) -> List[str]:
    ds = load_dataset("llm-blender/mix-instruct", split="validation")
    ds = ds.map(lambda x: {"prompt": f"{x['instruction']} {x['input']}"})
    return _sample_prompts(ds, ["prompt", "id"], n, seed)


def build_routerbench_probes(n: int, seed: int) -> List[str]:
    downloaded_file = hf_hub_download(repo_id="withmartian/routerbench", repo_type="dataset", filename="routerbench_0shot.pkl")
    ds = load_dataset("pandas", data_files={"data": downloaded_file}, split="data")
    ds = ds.shuffle(seed=seed)
    n_total = len(ds)
    n_train = int(0.9 * n_total)
    ds = ds.select(range(n_train))
    return _sample_prompts(ds, ["prompt", "sample_id"], n, seed)

def write_json(lst: List[str], path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(lst, f, indent=2)

def main():  
    parser = argparse.ArgumentParser(description="Generate probe sets for descriptors")
    parser.add_argument("--n_embedllm", type=int, default=0)
    parser.add_argument("--n_mix-instruct", type=int, default=0)
    parser.add_argument("--n_routerbench", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data", help="Directory to save probe JSON files")
    args = parser.parse_args()

    if args.n_embedllm != 0:
        embedllm_prompts = build_embedllm_probes(args.n_embedllm, args.seed + 3)
        write_json(embedllm_prompts, Path(args.out_dir) / f"probes_embedllm-{len(embedllm_prompts)}.json")
        print(f"Saved {len(embedllm_prompts)} embedllm prompts to {args.out_dir}/probes_embedllm-{len(embedllm_prompts)}.json")

    if args.n_mix_instruct != 0:
        mix_instruct_prompts = build_mix_instruct_probes(args.n_mix_instruct, args.seed + 4)
        write_json(mix_instruct_prompts, Path(args.out_dir) / f"probes_mix-instruct-{len(mix_instruct_prompts)}.json")
        print(f"Saved {len(mix_instruct_prompts)} mix_instruct prompts to {args.out_dir}/probes_mix_instruct-{len(mix_instruct_prompts)}.json")

    if args.n_routerbench != 0:
        routerbench_prompts = build_routerbench_probes(args.n_routerbench, args.seed + 5)
        write_json(routerbench_prompts, Path(args.out_dir) / f"probes_routerbench-{len(routerbench_prompts)}.json")
        print(f"Saved {len(routerbench_prompts)} routerbench prompts to {args.out_dir}/probes_routerbench-{len(routerbench_prompts)}.json")


if __name__ == "__main__":
    main()
