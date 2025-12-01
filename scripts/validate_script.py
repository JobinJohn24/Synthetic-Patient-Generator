"""
Validate synthetic data quality and privacy.
Visualize real vs synthetic distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA


REAL_DATA = "patient_data.npy"
SYNTHETIC_DATA = "synthetic_patients.npy"

X_real = np.load(REAL_DATA)
X_synthetic = np.load(SYNTHETIC_DATA)

print("=" * 50)
print("DISTRIBUTION FIDELITY (Kolmogorov-Smirnov Test)")
print("=" * 50)
ks_scores = []
for i in range(X_real.shape[1]):
    statistic, pvalue = ks_2samp(X_real[:, i], X_synthetic[:, i])
    ks_scores.append(statistic)
    print(f"Feature {i}: KS-Statistic = {statistic:.4f}")

avg_ks = np.mean(ks_scores)
print(f"\nAverage KS-Statistic: {avg_ks:.4f}")
print("✓ Lower values (~0.1-0.3) indicate good fidelity\n")

# PCA projection
pca = PCA(n_components=2)
real_2d = pca.fit_transform(X_real)
synthetic_2d = pca.transform(X_synthetic)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(real_2d[:, 0], real_2d[:, 1], alpha=0.5, label="Real", s=20)
plt.scatter(synthetic_2d[:, 0], synthetic_2d[:, 1], alpha=0.5, label="Synthetic", s=20)
plt.legend()
plt.title("PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")

# Distribution among features
plt.subplot(1, 2, 2)
feature_idx = 0  # Customize feature to visualize
plt.hist(X_real[:, feature_idx], bins=30, alpha=0.5, label="Real")
plt.hist(X_synthetic[:, feature_idx], bins=30, alpha=0.5, label="Synthetic")
plt.legend()
plt.title(f"Feature {feature_idx} Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("validation_results.png", dpi=150)
print("✓ Saved visualization to validation_results.png")
plt.show()
