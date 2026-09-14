"""
5. Apply PCA and t-SNE to reduce image data to 2D and visualize clusters.

Adapted for the uploaded image dataset (BSDS500-style: natural photos in
images/train, images/test, images/val -- no digit/class labels like MNIST).

Since there are no ground-truth class labels for these photos, each image
is converted into a feature vector (like flattening an MNIST digit into a
pixel vector), reduced to 2D with PCA and t-SNE, and then colored by
K-Means cluster so we can see whether visually similar images group
together in the 2D projection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = r"C:\Users\BIT\Desktop\ied1001723\sem 7\SML - II lab\archive (1)"   # <-- change this to your extracted folder path
SPLIT = "train"                 # "train", "test", or "val"
IMG_SIZE = (32, 32)             # resize target (like flattening MNIST to a vector)

image_dir = os.path.join(BASE_DIR, "images", SPLIT)
image_files = sorted(
    f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

# --- Load each image, resize, convert to grayscale, and flatten to a vector ---
features = []
for fname in image_files:
    img = Image.open(os.path.join(image_dir, fname)).convert("L")  # grayscale
    img = img.resize(IMG_SIZE)
    features.append(np.array(img).flatten())

X = np.array(features, dtype=float)
print(f"Loaded {X.shape[0]} images, each flattened to a {X.shape[1]}-dim vector")

# --- Scale features ---
X_scaled = StandardScaler().fit_transform(X)

# --- K-Means clustering (used only to color the 2D plots, since there are
#     no ground-truth class labels for these photos) ---
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# --- PCA to 2D ---
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# --- t-SNE to 2D ---
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# --- Plot both side by side ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="tab10", s=25)
axes[0].set_title("PCA (2D)")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap="tab10", s=25)
axes[1].set_title("t-SNE (2D)")
axes[1].set_xlabel("Dim 1")
axes[1].set_ylabel("Dim 2")

fig.colorbar(scatter2, ax=axes, label="K-Means cluster", ticks=range(k))
plt.suptitle(f"Dimensionality reduction on {SPLIT} images (colored by K-Means cluster)")
plt.show()

print(f"PCA explained variance ratio (2 components): {pca.explained_variance_ratio_.sum():.4f}")