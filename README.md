# [NeurIPS 2025 spotlight] Cost Aware Contrastive Router for LLMs

This repository is the implementation of the CSCR paper:

- arXiv:2508.12491 — https://arxiv.org/pdf/2508.12491
 
##  Quickstart

```bash
# 1) Install
pip install -e .

# 2) Generate probes
python scripts/generate_probes.py \
  --n_embedllm 192 \
  --n_mix-instruct 192 \
  --n_routerbench 192 \
  --seed 42

# 3) Compute a logit‑based descriptor for one expert model (EmbedLLM + MixInstruct probes)
python scripts/compute_descriptors.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --probes_files data/probes_embedllm-192.json data/probes_mix_instruct-192.json \
  --out experts/llama2-7b \
  --topk 256 --n_tokens 10

# 4) Build a FAISS index over all logit‑based expert descriptors
python scripts/build_faiss.py \
  --desc_dir experts \
  --index_out artifacts/experts.index \
  --index_type flat_ip --verbose

# 5) Run a quick routing evaluation
python scripts/run_router_eval.py \
  --index_path artifacts/experts.index \
  --labels_path artifacts/experts.index.labels.json \
  --dataset livecodebench --mode quick --k 1
```

For RouterBench, descriptors are perplexity‑based. See [the script.](scripts/compute_descriptors_perplexity.py)
 
---
 
## 📁 Directory Structure

```
├── README.md
├── LICENSE
├── pyproject.toml
├── data/                   ← probe sets (JSON) saved by scripts
├── plots/                  ← generated figures
├── scripts/                ← CLI entrypoints (see Quickstart)
└── src/
    └── router/            ← core routing modules
        ├── descriptors.py
        ├── query_encoder.py
        ├── knn_router.py
        ├── bandit.py
        ├── umr.py
        ├── random_router.py
        ├── cost_models.py
        ├── registry.py
        └── utils.py
```
## 📚 Citation

If you use this repository, please cite the paper:

```
@misc{shirkavand2025cscr,
      title={Cost-Aware Contrastive Routing for LLMs}, 
      author={Reza Shirkavand and Shangqian Gao and Peiran Yu and Heng Huang},
      year={2025},
      eprint={2508.12491},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.12491}, 
}
```
