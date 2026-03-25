"""
Clustering Models Module
========================
Implements multiple unsupervised learning algorithms:
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Model (GMM)

Includes optimal cluster selection using:
- Elbow Method
- Silhouette Score
- Davies-Bouldin Index

Author: Student
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


class ClusteringModels:
    """
    A class to implement and compare multiple clustering algorithms.
    """
    
    def __init__(self, df):
        """
        Initialize clustering models with data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature-engineered data for clustering
        """
        # Filter only numeric columns
        self.df = df.select_dtypes(include=[np.number]).copy()
        self.n_samples = len(self.df)
        
        # Store evaluation results
        self.evaluation_results = {}
        
    def _evaluate_clustering(self, labels, X, model_name):
        """
        Evaluate clustering results using multiple metrics.
        
        Parameters:
        -----------
        labels : array
            Cluster labels
        X : array
            Feature data
        model_name : str
            Name of the clustering model
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        # Skip if all points are noise (-1 in DBSCAN)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return {
                'silhouette': None,
                'davies_bouldin': None,
                'calinski_harabasz': None,
                'n_clusters': len(unique_labels)
            }
        
        # Calculate metrics
        try:
            silhouette = silhouette_score(X, labels)
        except:
            silhouette = None
            
        try:
            davies_bouldin = davies_bouldin_score(X, labels)
        except:
            davies_bouldin = None
            
        try:
            calinski = calinski_harabasz_score(X, labels)
        except:
            calinski = None
        
        metrics = {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski,
            'n_clusters': len(unique_labels)
        }
        
        print(f"\n{model_name} Evaluation Metrics:")
        print(f"  Silhouette Score: {silhouette:.4f}" if silhouette else "  Silhouette Score: N/A")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}" if davies_bouldin else "  Davies-Bouldin: N/A")
        print(f"  Calinski-Harabasz Index: {calinski:.2f}" if calinski else "  Calinski-Harabasz: N/A")
        print(f"  Number of Clusters: {metrics['n_clusters']}")
        
        return metrics
    
    def _find_optimal_k_kmeans(self, X, k_range=range(2, 11)):
        """
        Find optimal K using Elbow Method and Silhouette Score.
        
        Parameters:
        -----------
        X : array
            Feature data
        k_range : range
            Range of K values to try
            
        Returns:
        --------
        tuple : (optimal_k, elbow_k, silhouette_k)
        """
        print("\n--- Finding Optimal K for K-Means ---")
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
            sil_score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(sil_score)
            print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.4f}")
        
        # Elbow method: find the "elbow" point
        # Using second derivative method
        diffs = np.diff(inertias)
        diff2 = np.diff(diffs)
        elbow_idx = np.argmax(diff2) + 2  # +2 because of double diff offset
        elbow_k = list(k_range)[elbow_idx] if elbow_idx < len(k_range) else 5
        
        # Silhouette method: find K with highest silhouette score
        silhouette_k = list(k_range)[np.argmax(silhouette_scores)]
        
        print(f"\nOptimal K Analysis:")
        print(f"  Elbow Method suggests: K = {elbow_k}")
        print(f"  Silhouette Method suggests: K = {silhouette_k}")
        
        # Use silhouette for final decision (more reliable)
        optimal_k = silhouette_k
        
        # Save elbow plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Inertia', fontsize=12)
        plt.title('Elbow Method', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Silhouette Analysis', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/kmeans_optimal_k.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n*** Selected K = {optimal_k} (based on highest Silhouette Score) ***")
        
        return optimal_k, elbow_k, silhouette_k
    
    def kmeans_clustering(self, n_clusters=None):
        """
        Perform K-Means clustering with optimal K selection.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters (if None, will find optimal)
            
        Returns:
        --------
        dict : Clustering results and metrics
        """
        print("\n" + "=" * 60)
        print("K-MEANS CLUSTERING")
        print("=" * 60)
        
        X = self.df.values
        
        # Find optimal K if not specified
        if n_clusters is None:
            optimal_k, _, _ = self._find_optimal_k_kmeans(X)
            n_clusters = optimal_k
        
        print(f"\nApplying K-Means with K = {n_clusters}")
        
        # Fit K-Means
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        labels = kmeans.fit_predict(X)
        
        # Evaluate
        metrics = self._evaluate_clustering(labels, X, "K-Means")
        
        # Store results
        results = {
            'model': kmeans,
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'n_clusters': n_clusters,
            'silhouette': metrics['silhouette'],
            'davies_bouldin': metrics['davies_bouldin'],
            'calinski_harabasz': metrics['calinski_harabasz'],
            'inertia': kmeans.inertia_
        }
        
        self.evaluation_results['kmeans'] = results
        
        return results
    
    def hierarchical_clustering(self, n_clusters=None):
        """
        Perform Hierarchical (Agglomerative) clustering.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
            
        Returns:
        --------
        dict : Clustering results and metrics
        """
        print("\n" + "=" * 60)
        print("HIERARCHICAL CLUSTERING")
        print("=" * 60)
        
        X = self.df.values
        
        # Use same K as K-Means if not specified
        if n_clusters is None:
            if 'kmeans' in self.evaluation_results:
                n_clusters = self.evaluation_results['kmeans']['n_clusters']
            else:
                n_clusters = 5
        
        print(f"\nApplying Hierarchical Clustering with K = {n_clusters}")
        
        # Try different linkage methods
        linkage_methods = ['ward', 'complete', 'average']
        best_silhouette = -1
        best_method = 'ward'
        best_labels = None
        
        for method in linkage_methods:
            hc = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=method
            )
            labels = hc.fit_predict(X)
            
            sil_score = silhouette_score(X, labels)
            print(f"  Linkage={method}: Silhouette={sil_score:.4f}")
            
            if sil_score > best_silhouette:
                best_silhouette = sil_score
                best_method = method
                best_labels = labels
        
        print(f"\nBest Linkage Method: {best_method}")
        
        labels = best_labels
        
        # Evaluate
        metrics = self._evaluate_clustering(labels, X, "Hierarchical")
        
        # Store results
        results = {
            'model': None,  # Can't store model easily
            'labels': labels,
            'n_clusters': n_clusters,
            'linkage': best_method,
            'silhouette': metrics['silhouette'],
            'davies_bouldin': metrics['davies_bouldin'],
            'calinski_harabasz': metrics['calinski_harabasz']
        }
        
        self.evaluation_results['hierarchical'] = results
        
        return results
    
    def dbscan_clustering(self, eps=None, min_samples=None):
        """
        Perform DBSCAN clustering.
        
        Parameters:
        -----------
        eps : float
            Maximum distance between two samples
        min_samples : int
            Minimum samples in a neighborhood
            
        Returns:
        --------
        dict : Clustering results and metrics
        """
        print("\n" + "=" * 60)
        print("DBSCAN CLUSTERING")
        print("=" * 60)
        
        X = self.df.values
        
        # Find optimal parameters using k-distance graph approach
        if eps is None or min_samples is None:
            # Use a range of values and find best
            print("\nFinding optimal DBSCAN parameters...")
            
            best_silhouette = -1
            best_eps = 0.5
            best_min_samples = 5
            best_labels = None
            
            # Try different parameter combinations
            eps_values = [0.3, 0.5, 0.7, 1.0]
            min_samples_values = [3, 5, 7, 10]
            
            for eps in eps_values:
                for min_samples in min_samples_values:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X)
                    
                    # Only evaluate if we have valid clusters
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters >= 2:
                        # Filter out noise points for evaluation
                        mask = labels != -1
                        if mask.sum() > n_clusters:
                            sil_score = silhouette_score(X[mask], labels[mask])
                            
                            if sil_score > best_silhouette:
                                best_silhouette = sil_score
                                best_eps = eps
                                best_min_samples = min_samples
                                best_labels = labels
            
            eps = best_eps
            min_samples = best_min_samples
            labels = best_labels
            
            print(f"Best Parameters: eps={eps}, min_samples={min_samples}")
        
        print(f"\nApplying DBSCAN with eps={eps}, min_samples={min_samples}")
        
        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Count clusters and noise points
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        print(f"\nDBSCAN Results:")
        print(f"  Number of Clusters: {n_clusters}")
        print(f"  Noise Points: {n_noise} ({n_noise/len(labels)*100:.2f}%)")
        
        # Evaluate (excluding noise points)
        mask = labels != -1
        if mask.sum() > n_clusters and n_clusters >= 2:
            metrics = self._evaluate_clustering(labels[mask], X[mask], "DBSCAN")
        else:
            metrics = {
                'silhouette': None,
                'davies_bouldin': None,
                'calinski_harabasz': None,
                'n_clusters': n_clusters
            }
            print("\nDBSCAN Evaluation Metrics:")
            print(f"  Note: Could not compute metrics due to cluster configuration")
        
        # Store results
        results = {
            'model': dbscan,
            'labels': labels,
            'n_clusters': n_clusters,
            'eps': eps,
            'min_samples': min_samples,
            'noise_points': n_noise,
            'silhouette': metrics.get('silhouette'),
            'davies_bouldin': metrics.get('davies_bouldin'),
            'calinski_harabasz': metrics.get('calinski_harabasz')
        }
        
        self.evaluation_results['dbscan'] = results
        
        return results
    
    def gmm_clustering(self, n_components=None):
        """
        Perform Gaussian Mixture Model clustering.
        
        Parameters:
        -----------
        n_components : int
            Number of Gaussian components
            
        Returns:
        --------
        dict : Clustering results and metrics
        """
        print("\n" + "=" * 60)
        print("GAUSSIAN MIXTURE MODEL (GMM) CLUSTERING")
        print("=" * 60)
        
        X = self.df.values
        
        # Use same K as K-Means if not specified
        if n_components is None:
            if 'kmeans' in self.evaluation_results:
                n_components = self.evaluation_results['kmeans']['n_clusters']
            else:
                n_components = 5
        
        print(f"\nApplying GMM with {n_components} components")
        
        # Try different covariance types
        covariance_types = ['full', 'tied', 'diag', 'spherical']
        best_silhouette = -1
        best_type = 'full'
        best_labels = None
        
        for cov_type in covariance_types:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=cov_type,
                random_state=42,
                n_init=3
            )
            labels = gmm.fit_predict(X)
            
            sil_score = silhouette_score(X, labels)
            print(f"  Covariance={cov_type}: Silhouette={sil_score:.4f}")
            
            if sil_score > best_silhouette:
                best_silhouette = sil_score
                best_type = cov_type
                best_labels = labels
        
        print(f"\nBest Covariance Type: {best_type}")
        
        labels = best_labels
        
        # Evaluate
        metrics = self._evaluate_clustering(labels, X, "GMM")
        
        # Store results
        results = {
            'model': None,
            'labels': labels,
            'n_clusters': n_components,
            'covariance_type': best_type,
            'silhouette': metrics['silhouette'],
            'davies_bouldin': metrics['davies_bouldin'],
            'calinski_harabasz': metrics['calinski_harabasz']
        }
        
        self.evaluation_results['gmm'] = results
        
        return results
    
    def compare_all_models(self):
        """
        Create a comparison summary of all clustering models.
        
        Returns:
        --------
        pd.DataFrame : Comparison of all models
        """
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name.upper(),
                'Clusters': results.get('n_clusters', 'N/A'),
                'Silhouette': f"{results.get('silhouette', 0):.4f}" if results.get('silhouette') else 'N/A',
                'Davies-Bouldin': f"{results.get('davies_bouldin', 0):.4f}" if results.get('davies_bouldin') else 'N/A',
                'Calinski-Harabasz': f"{results.get('calinski_harabasz', 0):.2f}" if results.get('calinski_harabasz') else 'N/A'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n", comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv('results/model_comparison.csv', index=False)
        
        # Create comparison visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = [r['Model'] for r in comparison_data]
        
        # Silhouette comparison
        silhouette_vals = []
        for r in comparison_data:
            val = r['Silhouette']
            silhouette_vals.append(float(val) if val != 'N/A' else 0)
        
        axes[0].bar(models, silhouette_vals, color=['blue', 'green', 'orange', 'red'])
        axes[0].set_title('Silhouette Score (Higher is Better)')
        axes[0].set_ylabel('Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Davies-Bouldin comparison
        db_vals = []
        for r in comparison_data:
            val = r['Davies-Bouldin']
            db_vals.append(float(val) if val != 'N/A' else 0)
        
        axes[1].bar(models, db_vals, color=['blue', 'green', 'orange', 'red'])
        axes[1].set_title('Davies-Bouldin Index (Lower is Better)')
        axes[1].set_ylabel('Score')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Calinski-Harabasz comparison
        ch_vals = []
        for r in comparison_data:
            val = r['Calinski-Harabasz']
            ch_vals.append(float(val) if val != 'N/A' else 0)
        
        axes[2].bar(models, ch_vals, color=['blue', 'green', 'orange', 'red'])
        axes[2].set_title('Calinski-Harabasz Index (Higher is Better)')
        axes[2].set_ylabel('Score')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return comparison_df

