import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

FEATURES = ["R", "G", "B", "gray", "grad_mag", "local_std", "laplacian"]

df = pd.read_csv("C:/Users/BIT/OneDrive/Desktop/ied1002022/sml2/ml_assignment/pixel_dataset.csv")
print(f"Loaded {len(df)} pixel samples ({df.shape[1]-3} features) from YOUR BSDS500 images.")

# t-SNE is O(n^2)-ish, so subsample for speed while keeping it representative
sample_df = df.sample(n=min(3000, len(df)), random_state=42)
X = sample_df[FEATURES].values
y = sample_df["boundary"].values

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA explained variance ratio (PC1, PC2): {pca.explained_variance_ratio_}")
print(f"Total variance captured by 2 components: {pca.explained_variance_ratio_.sum():.2%}")

tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
X_tsne = tsne.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
labels_map = {0: "non-boundary", 1: "boundary"}
colors = {0: "tab:blue", 1: "tab:red"}

for cls in [0, 1]:
    mask = y == cls
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], s=8, alpha=0.5,
                     c=colors[cls], label=labels_map[cls])
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=8, alpha=0.5,
                     c=colors[cls], label=labels_map[cls])

axes[0].set_title("PCA projection (2D) — preserves global variance")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].legend()
axes[1].set_title("t-SNE projection (2D) — preserves local neighborhoods")
axes[1].set_xlabel("t-SNE dim 1"); axes[1].set_ylabel("t-SNE dim 2"); axes[1].legend()

plt.suptitle("Dimensionality Reduction on YOUR BSDS500 pixel features\n(boundary vs non-boundary pixels)")
plt.tight_layout()
plt.savefig("outputs/q5_pca_vs_tsne.png", dpi=150)
print("\nSaved comparison plot to outputs/q5_pca_vs_tsne.png")

sil_pca = silhouette_score(X_pca, y)
sil_tsne = silhouette_score(X_tsne, y)
print(f"\nSilhouette score (higher = better-separated clusters):")
print(f"  PCA  2D : {sil_pca:.4f}")
print(f"  t-SNE 2D: {sil_tsne:.4f}")
print("\nNote: unlike MNIST digits (very distinct classes), boundary vs")
print("non-boundary pixels overlap heavily in feature space -- edges exist")
print("on a continuum of gradient strength -- so expect much lower silhouette")
print("scores here than with clean, well-separated classes like MNIST digits.")
