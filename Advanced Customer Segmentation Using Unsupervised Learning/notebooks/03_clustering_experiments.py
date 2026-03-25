"""
# Clustering Experiments
## Step 3: Implementing and Comparing Clustering Algorithms

This notebook covers:
- K-Means Clustering with optimal K selection
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Model (GMM)
- Model comparison and evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Load processed data
print("Loading processed data...")
df = pd.read_csv('../data/processed/feature_engineered_data.csv')

print("\n" + "="*60)
print("CLUSTERING EXPERIMENTS")
print("="*60)

X = df.values

# ============== K-MEANS CLUSTERING ==============
print("\n--- K-Means Clustering ---")

# Find optimal K using Elbow Method and Silhouette Score
k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(sil_score)
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.4f}")

# Plot Elbow Method
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, inertias, 'bo-', linewidth=2)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')
axes[1].grid(True, alpha=0.3)

# Mark optimal K
optimal_k = list(k_range)[np.argmax(silhouette_scores)]
axes[1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
axes[1].legend()

plt.tight_layout()
plt.savefig('../results/kmeans_optimization.png', dpi=150)
plt.show()

print(f"\nOptimal K (based on Silhouette): {optimal_k}")

# Final K-Means with optimal K
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X)

print(f"\nK-Means Results (K={optimal_k}):")
print(f"Silhouette Score: {silhouette_score(X, kmeans_labels):.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X, kmeans_labels):.4f}")

# ============== HIERARCHICAL CLUSTERING ==============
print("\n--- Hierarchical Clustering ---")

# Try different linkage methods
linkage_methods = ['ward', 'complete', 'average']
hier_results = {}

for method in linkage_methods:
    hc = AgglomerativeClustering(n_clusters=optimal_k, linkage=method)
    labels = hc.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    hier_results[method] = {
        'labels': labels,
        'silhouette': sil_score
    }
    print(f"Linkage={method}: Silhouette={sil_score:.4f}")

# Use best linkage
best_linkage = max(hier_results.keys(), key=lambda x: hier_results[x]['silhouette'])
hier_labels = hier_results[best_linkage]['labels']
print(f"\nBest Linkage: {best_linkage}")

# ============== DBSCAN ==============
print("\n--- DBSCAN Clustering ---")

# Try different parameters
eps_values = [0.5, 0.7, 1.0]
min_samples_values = [5, 10]

best_dbscan_sil = -1
best_dbscan_labels = None
best_params = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters >= 2:
            mask = labels != -1
            if mask.sum() > n_clusters:
                sil_score = silhouette_score(X[mask], labels[mask])
                
                if sil_score > best_dbscan_sil:
                    best_dbscan_sil = sil_score
                    best_dbscan_labels = labels
                    best_params = (eps, min_samples)
                    
                print(f"eps={eps}, min_samples={min_samples}: Clusters={n_clusters}, Silhouette={sil_score:.4f}")

if best_params:
    print(f"\nBest DBSCAN: eps={best_params[0]}, min_samples={best_params[1]}, Silhouette={best_dbscan_sil:.4f}")

# ============== GAUSSIAN MIXTURE MODEL ==============
print("\n--- Gaussian Mixture Model ---")

# Try different covariance types
cov_types = ['full', 'tied', 'diag', 'spherical']
gmm_results = {}

for cov_type in cov_types:
    gmm = GaussianMixture(n_components=optimal_k, covariance_type=cov_type, 
                         random_state=42, n_init=3)
    labels = gmm.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    gmm_results[cov_type] = {
        'labels': labels,
        'silhouette': sil_score
    }
    print(f"Covariance={cov_type}: Silhouette={sil_score:.4f}")

# Best GMM
best_gmm = max(gmm_results.keys(), key=lambda x: gmm_results[x]['silhouette'])
gmm_labels = gmm_results[best_gmm]['labels']
print(f"\nBest GMM Covariance: {best_gmm}")

# ============== MODEL COMPARISON ==============
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison_data = [
    {'Model': 'K-Means', 'Silhouette': silhouette_score(X, kmeans_labels)},
    {'Model': 'Hierarchical', 'Silhouette': silhouette_score(X, hier_labels)},
    {'Model': 'DBSCAN', 'Silhouette': best_dbscan_sil if best_dbscan_labels is not None else 0},
    {'Model': 'GMM', 'Silhouette': silhouette_score(X, gmm_labels)}
]

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Visualize comparison
plt.figure(figsize=(10, 6))
plt.bar(comparison_df['Model'], comparison_df['Silhouette'], color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Model')
plt.ylabel('')
plt.title('Silhouette ScoreClustering Model Comparison')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/model_comparison.png', dpi=150)
plt.show()

# PCA Visualization
print("\n--- PCA Visualization ---")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# K-Means
axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='Set2', alpha=0.6)
axes[0, 0].set_title(f'K-Means (K={optimal_k})')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')

# Hierarchical
axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=hier_labels, cmap='Set2', alpha=0.6)
axes[0, 1].set_title(f'Hierarchical ({best_linkage})')
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')

# DBSCAN
if best_dbscan_labels is not None:
    axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=best_dbscan_labels, cmap='Set2', alpha=0.6)
    axes[1, 0].set_title('DBSCAN')
    axes[1, 0].set_xlabel('PC1')
    axes[1, 0].set_ylabel('PC2')

# GMM
axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='Set2', alpha=0.6)
axes[1, 1].set_title(f'GMM ({best_gmm})')
axes[1, 1].set_xlabel('PC1')
axes[1, 1].set_ylabel('PC2')

plt.tight_layout()
plt.savefig('../results/clustering_algorithms_comparison.png', dpi=150)
plt.show()

print("\nClustering experiments completed!")

