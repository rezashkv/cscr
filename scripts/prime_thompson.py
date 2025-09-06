import argparse, json
from pathlib import Path
from router.embedllm import load_embedllm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn_labels", required=True)
    ap.add_argument("--save_json", required=True)
    args = ap.parse_args()

    pool = json.load(open(args.knn_labels))
    train = load_embedllm("train", candidates=pool)

    alpha, beta = {m:1.0 for m in pool}, {m:1.0 for m in pool}
    for ex in train:
        gt = ex["answer"]
        for m in ex["candidates"]:
            correct = float(gt == ex["responses"][m])
            alpha[m] += correct
            beta[m]  += 1.0 - correct

    Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"alpha": alpha, "beta": beta}, open(args.save_json,"w"))
    print(f"✓ primed priors saved -> {args.save_json}")

if __name__ == "__main__":
    main()