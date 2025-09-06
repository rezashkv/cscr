import os
import argparse
import json
import numpy as np
import faiss
from router.utils import load_descriptors


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from descriptor files")
    parser.add_argument("--desc_dir", required=True, help="Directory containing .npy descriptor files")
    parser.add_argument("--index_out", required=True, help="Output file for FAISS index")
    parser.add_argument("--index_type", choices=["flat_l2", "flat_ip", "ivf_hnsw"], default="flat_ip", help="Type of index to build")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--test_ratio", type=float, default=1.0,
                         help="Ratio of descriptors to use for making the index")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling descriptors")
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    if args.verbose:
        print("Starting FAISS index build...")
        print(f"Descriptor directory: {args.desc_dir}")
        print(f"Index output: {args.index_out}")
        print(f"Index type: {args.index_type}")

   
    out_dir = os.path.dirname(args.index_out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        if args.verbose:
            print(f"Created output directory: {out_dir}")

    X, labels = load_descriptors(args.desc_dir, args.verbose)
    if len(X) == 0:
        print("No descriptors found. Exiting.")
        return

    X = np.stack(X)
    if args.verbose:
        print(f"Loaded {X.shape[0]} descriptors with dimension {X.shape[1]}")

    if args.test_ratio < 1.0:
        perm = np.random.permutation(len(X))
        X = X[perm]
        labels = [labels[i] for i in perm]
        num_test = int(args.test_ratio * len(X))

        if args.verbose:
            print(f"Using {num_test} descriptors for training")
        
        train_labels = labels[num_test:]
        X = X[:num_test]
        labels = labels[:num_test]
    
    d = X.shape[1] 

    if args.index_type == "flat_ip":
        # L2 normalize for cosine similarity
        if args.verbose:
            print("Normalizing descriptors for cosine similarity (inner product index)")
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-10)
        index = faiss.IndexFlatIP(d)

    elif args.index_type == "flat_l2":
        if args.verbose:
            print("Building L2 index")
        index = faiss.IndexFlatL2(d)
        
    else:
        if args.verbose:
            print("Building IVF HNSW index")
        nlist = 100
        m = 16
        index = faiss.IndexIVFFlat(faiss.IndexHNSWFlat(d, m), d, nlist, faiss.METRIC_L2)
        index.hnsw.efConstruction = 40
        index.nprobe = 10

    if args.verbose:
        print("Adding descriptors to index")
    index.add(X)

    if args.index_type == "ivf_hnsw":
        if args.verbose:
            print("Training IVF HNSW index")
        index.train(X)
        if args.verbose:
            print("Training complete")

    if args.verbose:
        print("Writing index to disk")
    faiss.write_index(index, args.index_out)

    labels_out = args.index_out + ".labels.json"
    if args.verbose:
        print(f"Writing labels to {labels_out}")
    with open(labels_out, "w") as f:
        json.dump(labels, f)
        
    if args.test_ratio < 1.0:
        train_labels_out = args.index_out + ".train_labels.json"
        if args.verbose:
            print(f"Writing training labels to {train_labels_out}")
        with open(train_labels_out, "w") as f:
            json.dump(train_labels, f)
    

    if args.verbose:
        print("FAISS index build complete.")

if __name__ == "__main__":
    main()