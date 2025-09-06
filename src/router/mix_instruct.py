from typing import List, Dict
import math
from datasets import load_dataset
from torch.utils.data import Dataset


class MixInstructOracle(Dataset):

    METRIC_KEY = "bartscore"       
    MARGIN     = 1e-1           
    NAME_TO_HF = {
        "vicuna-13b-1.1": "eachadea__vicuna-13b-1.1",
        "alpaca-native": "chavinlo__alpaca-native",
        "dolly-v2-12b": "databricks__dolly-v2-12b",
        "stablelm-tuned-alpha-7b": "stabilityai__stablelm-tuned-alpha-7b",
        "oasst-sft-4-pythia-12b-epoch-3.5": "OpenAssistant__oasst-sft-4-pythia-12b-epoch-3.5",
        "koala-7B-HF": "TheBloke__koala-7B-HF",
        "llama-7b-hf-baize-lora-bf16": "mosesjun0h__llama-7b-hf-baize-lora-bf16",
        "flan-t5-xxl": "google__flan-t5-xxl",
        "chatglm-6b": "THUDM__chatglm-6b",
        "moss-moon-003-sft": "fnlp__moss-moon-003-sft",
        "mpt-7b-instruct": "mosaicml__mpt-7b-instruct",
        "mpt-7b": "mosaicml__mpt-7b-instruct"
    }

    def __init__(
        self,
        expert_names: List[str],
        split: str = "train",
        score_key: str | None = None,
        max_len: int = 2048,
    ) -> None:
        super().__init__()

        if score_key is None:
            score_key = self.METRIC_KEY
        self.score_key = score_key
        self.max_len   = max_len

        self.name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(expert_names)}
        self.num_experts = len(expert_names)

        raw = load_dataset("llm-blender/mix-instruct", split=split)
        self.items = []
        skipped = 0
        for rec in raw:
            prompt = self._build_prompt(rec["instruction"], rec["input"])

            # gather candidate scores keyed by canonical model name
            scores: Dict[str, float] = {}
            for cand in rec["candidates"]:

                model_name = self._canon_name(cand["model"])
                if model_name not in self.name_to_idx:
                    continue
                metric_score = cand["scores"].get(self.score_key)

                if self.score_key == "bartscore":
                    metric_score = math.exp(metric_score)

                if metric_score is None:
                    continue
                if (model_name not in scores) or (metric_score > scores[model_name]):
                    scores[model_name] = metric_score

            if not scores:
                skipped += 1
                continue  

            best_val = max(scores.values())
            label = [0 for _ in range(self.num_experts)]
            for m, sc in scores.items():
                if best_val - sc <= self.MARGIN:
                    label[self.name_to_idx[m]] = 1.0

            self.items.append((prompt, label, None))

        if skipped:
            print(f"[MixInstructOracle] skipped {skipped} examples without matching experts.")

    
    @staticmethod
    def _build_prompt(instr: str, inp: str) -> str:
        instr = (instr or "").strip()
        inp   = (inp   or "").strip()
        if inp:
            return f"{instr}\n\n{inp}"
        return instr

    @staticmethod
    def _canon_name(name: str) -> str:
        return MixInstructOracle.NAME_TO_HF[name]

    
    def __len__(self) -> int:  
        return len(self.items)

    def __getitem__(self, idx): 
        return self.items[idx]

def load_mixinstruct(
    split: str = "test",
    candidates: list[str] | None = None,
    metric_key: str = "bartscore",
):
    from datasets import load_dataset

    raw = load_dataset("llm-blender/mix-instruct", split=split)

    samples: list[dict] = []
    for rec in raw:
        prompt_txt = MixInstructOracle._build_prompt(rec["instruction"], rec["input"])
        prompt_id  = rec["id"]

        
        model_to_score: dict[str, float] = {}
        for cand in rec["candidates"]:
            mdl = MixInstructOracle._canon_name(cand["model"])
            if candidates is not None and mdl not in candidates:
                continue
            sc  = cand["scores"].get(metric_key)
            if metric_key == "bartscore":
                sc = math.exp(sc)
            if sc is None:
                continue
            if (mdl not in model_to_score) or (sc > model_to_score[mdl]):
                model_to_score[mdl] = sc

        if not model_to_score:
            continue

        samples.append(
            {
                "prompt": prompt_txt,
                "category": "mixinstruct",
                "label_map": model_to_score,
                "prompt_id": prompt_id,
            }
        )

    return samples
