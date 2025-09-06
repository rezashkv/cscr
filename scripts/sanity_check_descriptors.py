import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


def load_descriptors(path: str | Path, pattern: str = "*.npy"):
    path = Path(path)
    vecs = []
    labels = []
    for file in sorted(path.glob(pattern)):
        vecs.append(np.load(file))
        labels.append(file.stem)
    X = np.stack(vecs)
    return X, labels


def plot_tsne(X: np.ndarray, labels: list[str]):
    tsne = TSNE(n_components=2, perplexity=min(len(X) - 1, 5), metric="cosine", random_state=42)
    X_2d = tsne.fit_transform(X)
    plt.figure(figsize=(6, 5))
    for i, label in enumerate(labels):
        plt.scatter(X_2d[i, 0], X_2d[i, 1], label=label)
        plt.text(X_2d[i, 0], X_2d[i, 1], label, fontsize=9)
    plt.title("t-SNE of Expert Descriptors")
    plt.tight_layout()
    plt.savefig("tsne_descriptors.png")
    plt.show()


def plot_similarity(X: np.ndarray, labels: list[str]):
    names = [label.split("_")[-1] for label in labels]
    sim = cosine_similarity(X)
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim, xticklabels=names, yticklabels=names, annot=True, cmap="viridis")
    plt.title("Cosine Similarity Between LLM Descriptors", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig("cosine_similarity.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory with *.npy descriptors")
    parser.add_argument("--pattern", type=str, default="*.npy")
    args = parser.parse_args()

    X, labels = load_descriptors(args.dir, args.pattern)
    print(f"Loaded {len(labels)} descriptors of dim {X.shape[1]}")
    plot_tsne(X, labels)
    plot_similarity(X, labels)


if __name__ == "__main__":
    main()
