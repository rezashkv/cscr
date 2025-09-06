import argparse, json, numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from router.embedllm import load_embedllm
from router.mix_instruct import load_mixinstruct
from router.routerbench import load_routerbench
from router.utils import load_descriptors
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn_labels", required=True,
                    help="labels.json with {idx: expert_id}")
    ap.add_argument("--desc_dir", required=True,
                    help="dir with descriptor files")
    ap.add_argument("--alpha", type=float, default=1e-3,
                    help="ridge λ")
    ap.add_argument("--save_ckpt", required=True,
                    help="Path to .npz where W and b are stored")
    ap.add_argument("--cost_type", default="n_params")
    ap.add_argument("--dataset", default="embedllm", type=str,
                    choices=["embedllm", "mix-instruct", "routerbench"],)
    ap.add_argument("--filter_prompt_cat", default=None, type=str,
        help="path to the json file to filter out prompts with certain categories in the dataset.")
    args = ap.parse_args()

    with open(args.knn_labels) as fh:
        pool = json.load(fh)       
    
    X, labels = load_descriptors(args.desc_dir, pool=pool, verbose=False)
    X = np.stack(X) 
    
    name2desc = {labels[i]: X[i] for i in range(X.shape[0])}
    X, y = [], []

    if args.dataset == "embedllm":     
        train_set = load_embedllm(split="train", candidates=pool, filter_prompt_cat=args.filter_prompt_cat)
    elif args.dataset == "mix-instruct":
        train_set = load_mixinstruct(split="train", candidates=pool, metric_key="bartscore")
    elif args.dataset == "routerbench":
        train_set = load_routerbench(split="train", candidates=pool)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    for rec in train_set:            
        lbl_map = rec["label_map"] 
        for m, correct in lbl_map.items():
            X.append(name2desc[m])
            y.append(float(correct))
    X, y = np.vstack(X), np.asarray(y, dtype=np.float32)

    reg = Ridge(alpha=args.alpha, fit_intercept=True)
    reg.fit(X, y)
    W  = reg.coef_.astype(np.float32)
    b  = float(reg.intercept_)

    Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.save_ckpt, W=W, b=b)
    print(f"✓ saved -> {args.save_ckpt}  (||W||={np.linalg.norm(W):.3f})")

if __name__ == "__main__":
    main()