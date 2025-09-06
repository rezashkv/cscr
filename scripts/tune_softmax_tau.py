
import argparse, json
from pathlib import Path
from router.baselines import get_baseline
from router.embedllm import load_embedllm
from router.utils    import eval_router

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau_grid", default="0.1,0.2,0.5,1,2,5", type=str)
    ap.add_argument("--knn_labels", required=True)
    ap.add_argument("--save_json", required=True)
    ap.add_argument("--cost_type", default="n_params")
    args = ap.parse_args()

    pool = json.load(open(args.knn_labels))
    val   = load_embedllm("val", candidates=pool)

    best, best_auc = None, -1
    taus = [float(t) for t in args.tau_grid.split(",")]
    for tau in taus:
        router = get_baseline("softmoe",
                              encoder_ckpt=None,
                              tau=tau,
                              cost_type=args.cost_type)()
        metrics = eval_router(router, val, quiet=True)
        if metrics["audc_norm"] > best_auc:
            best, best_auc = tau, metrics["audc_norm"]
        print(f"τ={tau:<4}  AUDC={metrics['audc_norm']:.4f}")

    Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tau": best}, open(args.save_json, "w"))
    print(f"✓ best τ={best}  saved -> {args.save_json}")

if __name__ == "__main__":
    main()