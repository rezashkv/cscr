import torch
from huggingface_hub import hf_hub_download
from datasets import load_dataset

class EmbedLLMOracle(torch.utils.data.Dataset):
    """
    Yields (prompt, oracle_idx) where oracle_idx indexes the descriptor matrix.
    """
    def __init__(self, desc_names, split="train", filter_prompt_cat=None):
        downloaded_file = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset",
                                           filename=f"{split}.csv")
        data = load_dataset("csv", data_files=downloaded_file, split=split)

        if filter_prompt_cat is not None:
            import json
            with open(filter_prompt_cat, "r") as f:
                filter_prompt_cat = json.load(f)

            data = data.filter(lambda x: x["category"] in filter_prompt_cat[split])
        
        by_pid = {}
        for row in data:
            if row["model_name"] not in desc_names:
                # skip rows for models we don't have descriptors for
                continue
            by_pid.setdefault(int(row["prompt_id"]), {})[row["model_name"]] = row
    
        self.items = []
        skipped = 0
        for rows in by_pid.values():
            self.items.append(
                (rows[desc_names[0]]["prompt"], [rows[desc]["label"] if desc in rows else 0 for desc in desc_names], None)
            )

        print(f"EmbedLLM: {len(self.items)} samples, skipped {skipped}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def load_embedllm(split: str = "test", candidates: list = None, filter_prompt_cat: str = None):
    csv_path = hf_hub_download(
        repo_id="RZ412/EmbedLLM", repo_type="dataset", filename=f"{split}.csv"
    )
    raw = load_dataset("csv", data_files=csv_path, split="train")

    if filter_prompt_cat is not None:
        import json
        with open(filter_prompt_cat, "r") as f:
            filter_prompt_cat = json.load(f)

        raw = raw.filter(lambda x: x["category"] in filter_prompt_cat[split if split in ["train", "test"] else "train"])

    # collapse rows with same prompt into one record containing label-map
    by_pid = {}
    for r in raw:
        if candidates is not None and r["model_name"] not in candidates:
            continue
        by_pid.setdefault(r["prompt_id"], []).append(r)

    samples = []
    for rows in by_pid.values():
        prompt = rows[0]["prompt"]
        prompt_id = rows[0]["prompt_id"]
        cat = rows[0].get("category", "embed")
        lbl_map = {r["model_name"]: r["label"] for r in rows}
        samples.append({"prompt": prompt, "category": cat, "label_map": lbl_map, "prompt_id": prompt_id})
    return samples

