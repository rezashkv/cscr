from typing import List, Dict, Any
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download


class RouterBenchOracle(torch.utils.data.Dataset):
    MODELS = [
        'claude-instant-v1', 
        'claude-v1',
        'claude-v2',
        'gpt-3.5-turbo-1106',
        'gpt-4-1106-preview',
        'meta/code-llama-instruct-34b-chat',
        'meta/llama-2-70b-chat',
        'mistralai/mistral-7b-chat', 
        'mistralai/mixtral-8x7b-chat',
        'zero-one-ai/Yi-34B-Chat',
        'WizardLM/WizardLM-13B-V1.2', 
    ]

    def __init__(
        self,
        desc_names: List[str],
        *,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 47,
    ):
        super().__init__()
        downloaded_file = hf_hub_download(repo_id="withmartian/routerbench",
                                           repo_type="dataset",
                                             filename="routerbench_0shot.pkl")
        
        ds = load_dataset("pandas", data_files={"data": downloaded_file}, split="data")       
        ds = ds.shuffle(seed=seed)
        train_size = int(len(ds) * train_ratio)
        train_ds = ds.select(range(train_size))
        test_ds = ds.select(range(train_size, len(ds)))

        if split == "train":
            raw = train_ds
        elif split == "test":
            raw = test_ds
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train' or 'test'.")
    
        pool = set(desc_names)
        self.items = []
        for item in raw:
            prompt = item["prompt"]
            scores, costs = [], []
            for m in self.MODELS:
                if m.replace("/", "__") in pool:
                    scores.append(item[m])
                    costs.append(item[f"{m}|total_cost"])

            self.items.append((prompt, scores, torch.tensor(costs, dtype=torch.float32)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def load_routerbench(
    split,
    candidates: List[str] | None = None,
    train_ratio: float = 0.9,
    seed: int = 47,
) -> List[Dict[str, Any]]:
    
    downloaded_file = hf_hub_download(repo_id="withmartian/routerbench",
                                           repo_type="dataset",
                                             filename="routerbench_0shot.pkl")
    ds = load_dataset("pandas", data_files={"data": downloaded_file}, split="data")
    ds = ds.shuffle(seed=seed)
    train_size = int(len(ds) * train_ratio)
    train_ds = ds.select(range(train_size))
    test_ds = ds.select(range(train_size, len(ds)))
    if split == "train":
        raw = train_ds
    elif split == "test":
        raw = test_ds
    else:
        raise ValueError(f"Invalid split: {split}. Use 'train' or 'test'.")

    pool = set(candidates) if candidates is not None else None

    samples = []
    for item in raw:
        prompt = item["prompt"]
        prompt_id = item["sample_id"]
        lbl_map = {m.replace("/", "__"): item[m] for m in RouterBenchOracle.MODELS if m.replace("/", "__") in pool}
        cost_map = {m.replace("/", "__"): item[f"{m}|total_cost"] for m in RouterBenchOracle.MODELS if m.replace("/", "__") in pool}
        samples.append(
            dict(
                prompt=prompt,
                label_map=lbl_map,
                prompt_id=prompt_id,
                cost_map=cost_map,
            )
        )
    return samples