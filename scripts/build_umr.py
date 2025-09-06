import argparse, json, random
from pathlib import Path

from router.umr import UMRBuilder, UMRRouter
from router.knn_router import KNNRouter   
from router.random_router import RandomRouter 
from router.oracle_router import OracleRouter     
from router.utils import eval_router
from router.embedllm import load_embedllm
from router.mix_instruct import load_mixinstruct
from router.routerbench import load_routerbench

import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
        

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="umr_artifacts", help="Where to save centroids")
    ap.add_argument("--k", type=int, default=20, help="#clusters for UMR")
    ap.add_argument("--lambda_", type=float, default=0.1, help="Cost weight λ")
    ap.add_argument("--encoder_ckpt", default=None, help="Optional QueryEncoder checkpoint")
    ap.add_argument("--bandit_beta", type=float, default=0.00001,
                    help="Bandit beta for KNN router")
    
    ap.add_argument("--quick_build", action="store_true",
                    help="Skip builder if artifacts already exist")
    ap.add_argument("--knn_index", required=True, help="Faiss index for contrastive router")
    ap.add_argument("--knn_labels", required=True, help="labels.json for contrastive router")
    ap.add_argument("--knn_labels_train", required=False, default=None, help="labels.json for train LLMs")
    ap.add_argument("--val_prompts", default=None, help="Val prompts for UMR - Probes for KNN") 
    ap.add_argument("--cost_type", choices=["latency", "n_params", "usd"], default="n_params",
                    help="Cost type to compute")
    ap.add_argument("--dataset", choices=["embedllm","mix-instruct", "routerbench"],
                     default="embedllm", help="Dataset to evaluate deferral curves on")
    ap.add_argument("--filter_prompt_cat", default=None, type=str,
        help="path to the json file to filter out prompts with certain categories in the dataset.")

    args = ap.parse_args()

    with open(args.knn_labels, 'r') as f:
        TEST_POOL = json.load(f)
        POOL = TEST_POOL.copy()
    
    if args.knn_labels_train is not None:
        with open(args.knn_labels_train, 'r') as f:
            POOL_TRAIN = json.load(f)  
        POOL = POOL_TRAIN + TEST_POOL

    work = Path(args.work_dir)
    if not work.exists() or not args.quick_build:
        if args.dataset == "embedllm":
            train_set = load_embedllm("train", candidates=POOL, filter_prompt_cat=args.filter_prompt_cat)
            val_set = load_embedllm("val", candidates=POOL, filter_prompt_cat=args.filter_prompt_cat)
        elif args.dataset == "mix-instruct":
            train_set = load_mixinstruct("train", candidates=POOL, metric_key="bartscore")
            val_set = load_mixinstruct("validation", candidates=POOL, metric_key="bartscore")
        elif args.dataset == "routerbench":
            train_set = load_routerbench("train", candidates=POOL)
            val_set = train_set
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        random.seed(0)
        random.shuffle(train_set)
        n = len(train_set)
        val_subset = None
        if args.val_prompts is not None:
            with open(args.val_prompts, "r") as f:
                val_subset = json.load(f)
                val_subset_ids = [s["prompt_id"] for s in val_subset]

            train_prompts = [s["prompt"] for s in train_set if s["prompt_id"] not in val_subset_ids]
            val_subset = [s for s in val_set if s["prompt_id"] in val_subset_ids]
        
        if val_subset is None:
            train_prompts = [s["prompt"] for s in train_set[: int(0.9 * n)]]
            val_subset = train_set[int(0.9 * n):]

        
        val_labels = {m: [] for m in POOL}
        val_prompts = []
        for s in val_subset:
            val_prompts.append(s["prompt"])
            for m in POOL:
                val_labels[m].append(s["label_map"].get(m, 0))

        print("▶ Building UMR artefacts ...")
        builder = UMRBuilder(device="cpu")

        builder.build(
            train_prompts=train_prompts,
            val_prompts=val_prompts,
            val_labels=val_labels,
            out_dir=work,
            k=args.k,
            cost_lambda=args.lambda_,
        )
    else:
        print("▶ Skipping build (artifacts found).")

    
    umr = UMRRouter(work_dir=work, override_lambda=args.lambda_, device="cpu")

    knn = KNNRouter(
        index_path=args.knn_index,
        labels_path=args.knn_labels,
        k=args.k,
        bandit_beta=args.bandit_beta,
        bandit_lambda=args.lambda_,
        encoder_ckpt=args.encoder_ckpt,
    )

    random_router = RandomRouter(
        labels_json=args.knn_labels,
    )
    
    oracle_router = OracleRouter(
        cost_type=args.cost_type,
        fallback_pool=POOL,
    )

    
    if args.dataset == "embedllm":
        test_set = load_embedllm("test", candidates=POOL, filter_prompt_cat=args.filter_prompt_cat)
        logging.info(f"Test set size: {len(test_set)}")
    elif args.dataset == "mix-instruct":
        test_set = load_mixinstruct("test", candidates=POOL, metric_key="bartscore")
    else:
        test_set = load_routerbench("test", candidates=POOL)        

    
    eval_router(umr, test_set, "UMR", cost_type=args.cost_type, lambda_coeff=args.lambda_)
    eval_router(knn, test_set, "Contrastive-KNN", cost_type=args.cost_type, lambda_coeff=args.lambda_)
    eval_router(random_router, test_set, "Random", cost_type=args.cost_type, lambda_coeff=args.lambda_)
    eval_router(oracle_router, test_set, "Clairvoyant Upper-Bound", cost_type=args.cost_type, lambda_coeff=args.lambda_)


if __name__ == "__main__":
    main()