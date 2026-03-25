"""
Visualization Module
====================
Handles all visualizations:
- Exploratory Data Analysis (EDA)
- Cluster Visualizations
- PCA Visualization
- t-SNE Visualization
- Business Dashboards

Author: Student
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class Visualizer:
    """
    A class to handle all visualizations for the customer segmentation project.
    """
    
    def __init__(self, df):
        """
        Initialize visualizer with data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to visualize
        """
        self.df = df.copy()
        
        # Create results directory
        import os
        os.makedirs('results', exist_ok=True)
        
    def plot_distribution(self, column, bins=30):
        """
        Plot distribution of a single column.
        
        Parameters:
        -----------
        column : str
            Column name
        bins : int
            Number of bins for histogram
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(self.df[column], bins=bins, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel(column, fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Distribution of {column}', fontsize=14)
        
        # Box plot
        axes[1].boxplot(self.df[column].dropna())
        axes[1].set_ylabel(column, fontsize=12)
        axes[1].set_title(f'Box Plot of {column}', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f'results/dist_{column}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_matrix(self):
        """
        Plot correlation matrix heatmap.
        """
        print("\n--- Correlation Matrix ---")
        
        # Select numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Plot heatmap
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Correlation Matrix Heatmap', fontsize=16)
        plt.tight_layout()
        plt.savefig('results/correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Correlation matrix saved to results/correlation_matrix.png")
        
        return corr_matrix
    
    def plot_bivariate_analysis(self, x_col, y_col):
        """
        Plot bivariate analysis between two columns.
        
        Parameters:
        -----------
        x_col : str
            X-axis column
        y_col : str
            Y-axis column
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Scatter plot
        axes[0].scatter(self.df[x_col], self.df[y_col], alpha=0.5)
        axes[0].set_xlabel(x_col, fontsize=12)
        axes[0].set_ylabel(y_col, fontsize=12)
        axes[0].set_title(f'{x_col} vs {y_col}', fontsize=14)
        
        # Hexbin plot
        axes[1].hexbin(self.df[x_col], self.df[y_col], gridsize=20, cmap='YlOrRd')
        axes[1].set_xlabel(x_col, fontsize=12)
        axes[1].set_ylabel(y_col, fontsize=12)
        axes[1].set_title(f'Density: {x_col} vs {y_col}', fontsize=14)
        plt.colorbar(axes[1].collections[0], ax=axes[1])
        
        # With regression line
        sns.regplot(x=x_col, y=y_col, data=self.df, ax=axes[2], scatter_kws={'alpha':0.3})
        axes[2].set_title(f'Regression: {x_col} vs {y_col}', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f'results/bivariate_{x_col}_vs_{y_col}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_categorical_distribution(self, column):
        """
        Plot distribution of categorical variables.
        
        Parameters:
        -----------
        column : str
            Column name
        """
        plt.figure(figsize=(10, 6))
        
        value_counts = self.df[column].value_counts()
        
        plt.bar(value_counts.index, value_counts.values, color=plt.cm.Set3.colors)
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(f'Distribution of {column}', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, (idx, val) in enumerate(zip(value_counts.index, value_counts.values)):
            plt.text(i, val + 50, f'{val}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'results/categorical_{column}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def perform_eda(self):
        """
        Perform comprehensive EDA and generate visualizations.
        """
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # 1. Distribution plots for key numeric features
        key_numeric = ['Age', 'Income', 'Spending_Score', 'Total_Spending', 
                      'Num_Purchases', 'Recency']
        
        available_numeric = [col for col in key_numeric if col in self.df.columns]
        
        print(f"\nGenerating distribution plots for: {available_numeric}")
        
        for col in available_numeric:
            if col in self.df.columns:
                self.plot_distribution(col)
        
        # 2. Correlation matrix
        print("\nGenerating correlation matrix...")
        self.plot_correlation_matrix()
        
        # 3. Bivariate analysis for key pairs
        bivariate_pairs = [
            ('Income', 'Total_Spending'),
            ('Spending_Score', 'Num_Purchases'),
            ('Age', 'Spending_Score'),
            ('Recency', 'Total_Spending')
        ]
        
        print("\nGenerating bivariate analysis...")
        for x, y in bivariate_pairs:
            if x in self.df.columns and y in self.df.columns:
                self.plot_bivariate_analysis(x, y)
        
        print("\nEDA visualizations saved to results/")
        
    def plot_cluster_results(self, df, labels, model_name):
        """
        Plot cluster results.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with features
        labels : array
            Cluster labels
        model_name : str
            Name of clustering model
        """
        print(f"\n--- Plotting {model_name} Cluster Results ---")
        
        n_clusters = len(np.unique(labels))
        
        # Select top features for visualization
        key_features = ['Income', 'Spending_Score', 'Total_Spending', 'Num_Purchases', 'Age']
        available_features = [f for f in key_features if f in df.columns]
        
        if len(available_features) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Scatter plot 1
            scatter1 = axes[0].scatter(
                df[available_features[0]], 
                df[available_features[1]], 
                c=labels, 
                cmap='Set2', 
                alpha=0.6
            )
            axes[0].set_xlabel(available_features[0], fontsize=12)
            axes[0].set_ylabel(available_features[1], fontsize=12)
            axes[0].set_title(f'{model_name} Clusters: {available_features[0]} vs {available_features[1]}', fontsize=14)
            plt.colorbar(scatter1, ax=axes[0], label='Cluster')
            
            # Scatter plot 2
            if len(available_features) >= 3:
                scatter2 = axes[1].scatter(
                    df[available_features[1]], 
                    df[available_features[2]], 
                    c=labels, 
                    cmap='Set2', 
                    alpha=0.6
                )
                axes[1].set_xlabel(available_features[1], fontsize=12)
                axes[1].set_ylabel(available_features[2], fontsize=12)
                axes[1].set_title(f'{model_name} Clusters: {available_features[1]} vs {available_features[2]}', fontsize=14)
                plt.colorbar(scatter2, ax=axes[1], label='Cluster')
            
            plt.tight_layout()
            plt.savefig(f'results/{model_name}_clusters.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3D visualization if enough features
        if len(available_features) >= 3:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(
                df[available_features[0]],
                df[available_features[1]],
                df[available_features[2]],
                c=labels,
                cmap='Set2',
                alpha=0.6
            )
            
            ax.set_xlabel(available_features[0])
            ax.set_ylabel(available_features[1])
            ax.set_zlabel(available_features[2])
            ax.set_title(f'{model_name} 3D Cluster Visualization')
            plt.colorbar(scatter, ax=ax, label='Cluster')
            
            plt.tight_layout()
            plt.savefig(f'results/{model_name}_clusters_3d.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"{model_name} cluster plots saved to results/")
    
    def plot_pca_visualization(self, df, labels, n_components=2):
        """
        Perform PCA and visualize clusters in reduced dimensions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with features
        labels : array
            Cluster labels
        n_components : int
            Number of PCA components
        """
        print("\n--- PCA Visualization ---")
        
        # Select numeric features
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Remove cluster column if present
        if 'Cluster' in numeric_df.columns:
            numeric_df = numeric_df.drop('Cluster', axis=1)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(numeric_df)
        
        # Print explained variance
        print(f"PCA Explained Variance Ratio:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {var*100:.2f}%")
        print(f"  Total: {sum(pca.explained_variance_ratio_)*100:.2f}%")
        
        # Plot 2D PCA
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(
            pca_result[:, 0], 
            pca_result[:, 1], 
            c=labels, 
            cmap='Set2', 
            alpha=0.6,
            s=50
        )
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        plt.title('PCA Visualization of Customer Clusters', fontsize=14)
        plt.colorbar(scatter, label='Cluster')
        
        # Add cluster centroids
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            centroid_x = pca_result[mask, 0].mean()
            centroid_y = pca_result[mask, 1].mean()
            plt.scatter(centroid_x, centroid_y, c='black', s=200, marker='X', 
                       edgecolors='white', linewidths=2)
            plt.annotate(f'C{cluster_id}', (centroid_x, centroid_y), 
                        fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/pca_clusters.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        
        # Cumulative variance
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        
        plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-')
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
               pca.explained_variance_ratio_, alpha=0.5)
        
        plt.xlabel('Principal Component', fontsize=12)
        plt.ylabel('Explained Variance Ratio', fontsize=12)
        plt.title('PCA Explained Variance', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/pca_explained_variance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save PCA loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=numeric_df.columns
        )
        loadings.to_csv('results/pca_loadings.csv')
        
        print("PCA visualizations saved to results/")
        
        return pca_result, pca
    
    def plot_tsne_visualization(self, df, labels, perplexity=30):
        """
        Perform t-SNE and visualize clusters in reduced dimensions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with features
        labels : array
            Cluster labels
        perplexity : int
            t-SNE perplexity parameter
        """
        print("\n--- t-SNE Visualization ---")
        
        # Select numeric features
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Remove cluster column if present
        if 'Cluster' in numeric_df.columns:
            numeric_df = numeric_df.drop('Cluster', axis=1)
        
        # Apply t-SNE (use sample if data is large)
        sample_size = min(5000, len(numeric_df))
        
        if len(numeric_df) > sample_size:
            print(f"Sampling {sample_size} points for t-SNE visualization...")
            indices = np.random.choice(len(numeric_df), sample_size, replace=False)
            sample_df = numeric_df.iloc[indices]
            sample_labels = labels[indices]
        else:
            sample_df = numeric_df
            sample_labels = labels
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                   n_iter=1000, learning_rate='auto', init='pca')
        tsne_result = tsne.fit_transform(sample_df)
        
        # Plot t-SNE
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(
            tsne_result[:, 0], 
            tsne_result[:, 1], 
            c=sample_labels, 
            cmap='Set2', 
            alpha=0.6,
            s=50
        )
        
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.title(f't-SNE Visualization of Customer Clusters (perplexity={perplexity})', fontsize=14)
        plt.colorbar(scatter, label='Cluster')
        
        plt.tight_layout()
        plt.savefig('results/tsne_clusters.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("t-SNE visualization saved to results/")
        
        return tsne_result, tsne
    
    def create_summary_dashboard(self, cluster_profiles):
        """
        Create a summary dashboard of the clustering results.
        
        Parameters:
        -----------
        cluster_profiles : dict
            Cluster profiles dictionary
        """
        print("\n--- Creating Summary Dashboard ---")
        
        n_clusters = len(cluster_profiles)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Cluster sizes
        ax1 = fig.add_subplot(2, 3, 1)
        sizes = [p['size'] for p in cluster_profiles.values()]
        clusters = [f"C{p['cluster_id']}" for p in cluster_profiles.values()]
        ax1.pie(sizes, labels=clusters, autopct='%1.1f%%', colors=plt.cm.Set2.colors)
        ax1.set_title('Cluster Size Distribution', fontsize=14)
        
        # 2. Average Total Spending by Cluster
        ax2 = fig.add_subplot(2, 3, 2)
        spending = [p['avg_total_spending'] for p in cluster_profiles.values()]
        ax2.bar(clusters, spending, color=plt.cm.Set2.colors)
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Average Total Spending')
        ax2.set_title('Avg Spending by Cluster', fontsize=14)
        
        # 3. Average Purchases by Cluster
        ax3 = fig.add_subplot(2, 3, 3)
        purchases = [p['avg_num_purchases'] for p in cluster_profiles.values()]
        ax3.bar(clusters, purchases, color=plt.cm.Set2.colors)
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Avg Number of Purchases')
        ax3.set_title('Avg Purchases by Cluster', fontsize=14)
        
        # 4. Average Income by Cluster
        ax4 = fig.add_subplot(2, 3, 4)
        income = [p['avg_income'] for p in cluster_profiles.values()]
        ax4.bar(clusters, income, color=plt.cm.Set2.colors)
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Average Income')
        ax4.set_title('Avg Income by Cluster', fontsize=14)
        
        # 5. Average Recency by Cluster
        ax5 = fig.add_subplot(2, 3, 5)
        recency = [p['avg_recency'] for p in cluster_profiles.values()]
        ax5.bar(clusters, recency, color=plt.cm.Set2.colors)
        ax5.set_xlabel('Cluster')
        ax5.set_ylabel('Average Recency (days)')
        ax5.set_title('Avg Recency by Cluster', fontsize=14)
        
        # 6. Customer Types
        ax6 = fig.add_subplot(2, 3, 6)
        types = [p.get('customer_type', 'Unknown') for p in cluster_profiles.values()]
        unique_types = list(set(types))
        type_counts = [types.count(t) for t in unique_types]
        ax6.barh(unique_types, type_counts, color=plt.cm.Set3.colors[:len(unique_types)])
        ax6.set_xlabel('Count')
        ax6.set_title('Customer Types Distribution', fontsize=14)
        
        plt.suptitle('Customer Segmentation Dashboard', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('results/summary_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Summary dashboard saved to results/")

