#!/usr/bin/env bash
set -euo pipefail


N_PROBES=192
PROBE_SEED=42
PROBE_DIR=data
DESC_DIR=experts/descriptors
INDEX_OUT=${DESC_DIR}/faiss_index.index
ENCODER_OUT=checkpoints/contrastive_encoder

# ------------------ generate probes ------------------ #
python scripts/generate_probes.py \
  --n_embedllm ${N_PROBES} --n_mix-instruct ${N_PROBES} --n_routerbench ${N_PROBES} \
  --seed ${PROBE_SEED} --out_dir ${PROBE_DIR}

echo "✅ probe sets ready"

# ---------------- compute descriptors ---------------- #
PROBES_EMBEDLLM=${PROBE_DIR}/probes_embedllm-${N_PROBES}.json
PROBES_MIXINSTRUCT=${PROBE_DIR}/probes_mix-instruct-${N_PROBES}.json
PROBES_ROUTERBENCH=${PROBE_DIR}/probes_routerbench-${N_PROBES}.json

# logit based descriptors for embdellm
# for MODEL_ID in $(jq -r 'keys[]' experts/registry-embedllm.json); do
#   HF_ID=$(jq -r --arg m "$MODEL_ID" '.[$m].hf_id' experts/registry-embedllm.json)
#   OUT_FILE=${DESC_DIR}/embedllm/${MODEL_ID}_desc.npy
#   if [[ ! -f ${OUT_FILE} ]]; then
#     echo "Computing descriptor for $MODEL_ID ($HF_ID)"
#     python scripts/compute_descriptors.py \
#       --model ${HF_ID} \
#       --probes_files ${PROBES_EMBEDLLM} \
#       --out ${OUT_FILE} \
#       --topk 256 --n_tokens 10
#   fi
# done

# logit based descriptors for mix-instruct
# for MODEL_ID in $(jq -r 'keys[]' experts/registry-mix-instruct.json); do
#   HF_ID=$(jq -r --arg m "$MODEL_ID" '.[$m].hf_id' experts/registry-mix-instruct.json)
#   OUT_FILE=${DESC_DIR}/mix-instruct//${MODEL_ID}_desc.npy
#   if [[ ! -f ${OUT_FILE} ]]; then
#     echo "Computing descriptor for $MODEL_ID ($HF_ID)"
#     python scripts/compute_descriptors.py \
#       --model ${HF_ID} \
#       --probes_files ${PROBES_MIXINSTRUCT} \
#       --out ${OUT_FILE} \
#       --topk 256 --n_tokens 10
#   fi
# done

# perplexity based descriptors
python scripts/compute_descriptors_perplexity.py \
  --probe_ids ${PROBES_ROUTERBENCH} \
  --dataset routerbench \
  --split train \
  --out ${DESC_DIR}/routerbench \
  --plot

echo "✅ descriptors computed"

# ------------------- build FAISS --------------------- #
python scripts/build_faiss.py --desc_dir experts/descriptors/embedllm/ --index_out experts/indices/embedllm/faiss_index.ivf --index_type flat_ip --verbose
python scripts/build_faiss.py --desc_dir experts/descriptors/mix-instruct/ --index_out experts/indices/mix-instruct/faiss_index.ivf --index_type flat_ip --verbose
python scripts/build_faiss.py --desc_dir experts/descriptors/routerbench/ --index_out experts/indices/routerbench/faiss_index.ivf --index_type flat_ip --verbose

# ----------------- train encoder --------------------- #
python scripts/train_query_encoder.py --epochs 10 --batch_size 512 --lr 5e-4 --out_dir checkpoints/embedllm/contrastive_encoder_expert_aware_cost-spectrum-info-nce --desc_dir experts/descriptors/embedllm/ --contrastive_loss_type cost_spectrum_info_nce --pool experts/indices/embedllm/faiss_index.ivf.labels.json --dataset embedllm

# ------------------ build umr ----------------------- #
python scripts/build_umr.py --work_dir baselines/umr/umr_artifacts_embedllm --knn_index experts/indices/embedllm/faiss_index.ivf --knn_labels experts/indices/embedllm/faiss_index.ivf.labels.json --val_prompts data/probes_embedllm-192-id.json --encoder_ckpt checkpoints/embedllm/contrastive_encoder_expert_aware_cost-spectrum-info-nce/checkpoint-6/ --lambda_ 0.1 --bandit_beta 0.000001 --k 20 --dataset embedllm --cost_type "n_params"

# ------------------ eval ----------------------- #
python scripts/run_audc_eval.py --umr_work_dir baselines/umr/umr_artifacts_embedllm/ --knn_index experts/indices/embedllm/faiss_index.ivf --knn_labels experts/indices/embedllm/faiss_index.ivf.labels.json --knn_encoder_ckpt checkpoints/embedllm/contrastive_encoder_expert_aware_cost-spectrum-info-nce/checkpoint-10/ --routers umr knn --save_plot "plots/deferral_embedllm.png" --no_show  --knn_bandit_beta 0.000001 --n_points 50 --cost_grid_points 20  --parametric_npz baselines/parametric/embedllm.npz --parametric_embedding_dir experts/descriptors/embedllm/ --dataset embedllm --cost_type "n_params" --save_curve curves/embedllm-512.pkl

