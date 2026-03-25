"""
Cluster Analysis Module
=======================
Performs detailed analysis and interpretation of customer clusters:
- Cluster profiling
- Behavioral characteristics
- Demographics patterns
- Customer type definition

Author: Student
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class ClusterAnalyzer:
    """
    A class to analyze and interpret customer clusters.
    """
    
    def __init__(self, df):
        """
        Initialize cluster analyzer with data containing cluster labels.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with cluster labels
        """
        self.df = df.copy()
        self.cluster_profiles = {}
        
    def get_cluster_statistics(self):
        """
        Get basic statistics for each cluster.
        
        Returns:
        --------
        dict : Statistics for each cluster
        """
        print("\n--- Cluster Statistics ---")
        
        cluster_stats = self.df.groupby('Cluster').agg({
            'Age': ['mean', 'std', 'min', 'max'],
            'Income': ['mean', 'std', 'min', 'max'],
            'Spending_Score': ['mean', 'std', 'min', 'max'],
            'Num_Purchases': ['mean', 'std', 'min', 'max'],
            'Total_Spending': ['mean', 'std', 'min', 'max'],
            'Recency': ['mean', 'std', 'min', 'max']
        })
        
        print(cluster_stats)
        
        return cluster_stats
    
    def analyze_cluster_characteristics(self):
        """
        Analyze detailed characteristics of each cluster.
        
        Returns:
        --------
        dict : Cluster profiles with characteristics
        """
        print("\n--- Analyzing Cluster Characteristics ---")
        
        n_clusters = self.df['Cluster'].nunique()
        
        for cluster_id in sorted(self.df['Cluster'].unique()):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            cluster_pct = (cluster_size / len(self.df)) * 100
            
            print(f"\n{'='*60}")
            print(f"CLUSTER {cluster_id} - {cluster_size} customers ({cluster_pct:.1f}%)")
            print(f"{'='*60}")
            
            # Calculate cluster statistics
            profile = {
                'cluster_id': cluster_id,
                'size': cluster_size,
                'percentage': cluster_pct,
                'avg_age': cluster_data['Age'].mean(),
                'avg_income': cluster_data['Income'].mean(),
                'avg_spending_score': cluster_data['Spending_Score'].mean(),
                'avg_num_purchases': cluster_data['Num_Purchases'].mean(),
                'avg_total_spending': cluster_data['Total_Spending'].mean(),
                'avg_recency': cluster_data['Recency'].mean(),
                'avg_tenure': cluster_data['Tenure_Months'].mean() if 'Tenure_Months' in cluster_data.columns else None
            }
            
            # Print key metrics
            print(f"\nDemographics:")
            print(f"  Average Age: {profile['avg_age']:.1f}")
            print(f"  Average Income: ${profile['avg_income']:.2f}")
            
            print(f"\nSpending Behavior:")
            print(f"  Spending Score: {profile['avg_spending_score']:.1f}")
            print(f"  Total Spending: ${profile['avg_total_spending']:.2f}")
            print(f"  Number of Purchases: {profile['avg_num_purchases']:.1f}")
            
            print(f"\nEngagement:")
            print(f"  Recency (days): {profile['avg_recency']:.1f}")
            if profile['avg_tenure']:
                print(f"  Tenure (months): {profile['avg_tenure']:.1f}")
            
            self.cluster_profiles[cluster_id] = profile
        
        return self.cluster_profiles
    
    def define_customer_types(self):
        """
        Define customer types based on cluster characteristics.
        
        Returns:
        --------
        dict : Customer type definitions for each cluster
        """
        print("\n--- Defining Customer Types ---")
        
        customer_types = {}
        
        # Calculate overall averages for comparison
        overall_avg_spending = self.df['Total_Spending'].mean()
        overall_avg_purchases = self.df['Num_Purchases'].mean()
        overall_avg_recency = self.df['Recency'].mean()
        overall_avg_income = self.df['Income'].mean()
        overall_avg_spending_score = self.df['Spending_Score'].mean()
        
        for cluster_id, profile in self.cluster_profiles.items():
            # Determine customer type based on characteristics
            spending_ratio = profile['avg_total_spending'] / overall_avg_spending
            purchases_ratio = profile['avg_num_purchases'] / overall_avg_purchases
            recency_ratio = profile['avg_recency'] / overall_avg_recency
            
            # Classify based on multiple factors
            if spending_ratio > 1.3 and purchases_ratio > 1.3:
                customer_type = "Premium Loyal Customers"
                description = "High spenders who purchase frequently"
            elif spending_ratio > 1.3 and purchases_ratio <= 1.0:
                customer_type = "Big Ticket Buyers"
                description = "High-value single purchases"
            elif spending_ratio < 0.7 and purchases_ratio > 1.3:
                customer_type = "Frequent Low-Spenders"
                description = "Shop often but spend less per visit"
            elif spending_ratio < 0.7 and purchases_ratio < 0.7:
                customer_type = "Budget Conscious Shoppers"
                description = "Low frequency and low spending"
            elif recency_ratio > 1.5:
                customer_type = "At-Risk Customers"
                description = "Haven't purchased in a while"
            elif spending_ratio > 1.0 and purchases_ratio < 0.7:
                customer_type = "Occasional High-Spenders"
                description = "Infrequent but valuable purchases"
            elif profile['avg_spending_score'] > overall_avg_spending_score * 1.2:
                customer_type = "High-Value Enthusiasts"
                description = "High spending potential customers"
            elif profile['avg_recency'] < overall_avg_recency * 0.5:
                customer_type = "Recent Active Customers"
                description = "Recently made purchases"
            else:
                customer_type = "Average Customers"
                description = "Typical mainstream customers"
            
            customer_types[cluster_id] = {
                'type': customer_type,
                'description': description,
                'spending_ratio': spending_ratio,
                'purchases_ratio': purchases_ratio,
                'recency_ratio': recency_ratio
            }
            
            print(f"\nCluster {cluster_id}: {customer_type}")
            print(f"  Description: {description}")
            print(f"  Spending Ratio: {spending_ratio:.2f}x average")
            print(f"  Purchases Ratio: {purchases_ratio:.2f}x average")
            print(f"  Recency Ratio: {recency_ratio:.2f}x average")
        
        # Update profiles with customer types
        for cluster_id, type_info in customer_types.items():
            self.cluster_profiles[cluster_id]['customer_type'] = type_info['type']
            self.cluster_profiles[cluster_id]['description'] = type_info['description']
        
        return customer_types
    
    def create_radar_chart(self, cluster_id):
        """
        Create a radar chart for a specific cluster.
        
        Parameters:
        -----------
        cluster_id : int
            Cluster ID to visualize
        """
        # Get features for radar chart
        features = ['Age', 'Income', 'Spending_Score', 'Num_Purchases', 'Total_Spending']
        available_features = [f for f in features if f in self.df.columns]
        
        if len(available_features) < 3:
            print("Not enough features for radar chart")
            return
        
        # Calculate normalized values for the cluster
        cluster_data = self.df[self.df['Cluster'] == cluster_id]
        
        # Normalize to 0-1 scale
        values = []
        for feat in available_features:
            feat_min = self.df[feat].min()
            feat_max = self.df[feat].max()
            if feat_max > feat_min:
                cluster_val = cluster_data[feat].mean()
                normalized = (cluster_val - feat_min) / (feat_max - feat_min)
                values.append(normalized)
            else:
                values.append(0.5)
        
        # Close the radar chart
        values += values[:1]
        
        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, len(available_features), endpoint=False).tolist()
        angles += angles[:1]
        
        return angles, values, available_features
    
    def generate_cluster_visualizations(self):
        """
        Generate visualizations for cluster analysis.
        """
        print("\n--- Generating Cluster Visualizations ---")
        
        n_clusters = self.df['Cluster'].nunique()
        
        # 1. Cluster size distribution
        plt.figure(figsize=(10, 6))
        cluster_counts = self.df['Cluster'].value_counts().sort_index()
        bars = plt.bar(cluster_counts.index, cluster_counts.values, color=plt.cm.Set2.colors)
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Number of Customers', fontsize=12)
        plt.title('Customer Distribution Across Clusters', fontsize=14)
        
        # Add count labels on bars
        for bar, count in zip(bars, cluster_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{count}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('results/cluster_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Cluster characteristics comparison
        key_features = ['Age', 'Income', 'Spending_Score', 'Total_Spending', 'Num_Purchases']
        available_features = [f for f in key_features if f in self.df.columns]
        
        if available_features:
            fig, axes = plt.subplots(2, (len(available_features) + 1) // 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, feature in enumerate(available_features):
                self.df.boxplot(column=feature, by='Cluster', ax=axes[idx])
                axes[idx].set_title(f'{feature} by Cluster')
                axes[idx].set_xlabel('Cluster')
            
            # Hide unused subplots
            for idx in range(len(available_features), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle('Cluster Characteristics Comparison', fontsize=14)
            plt.tight_layout()
            plt.savefig('results/cluster_characteristics.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Pairplot for top features
        if len(available_features) >= 3:
            plot_features = available_features[:4]
            plot_df = self.df[plot_features + ['Cluster']].copy()
            
            # Sample for faster plotting
            if len(plot_df) > 2000:
                plot_df = plot_df.sample(n=2000, random_state=42)
            
            g = sns.pairplot(plot_df, hue='Cluster', palette='Set2', diag_kind='kde')
            g.fig.suptitle('Cluster Pairwise Relationships', y=1.02)
            plt.savefig('results/cluster_pairplot.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print("Cluster visualizations saved to results/")
        
        return self.cluster_profiles
    
    def analyze_clusters(self):
        """
        Execute complete cluster analysis pipeline.
        
        Returns:
        --------
        dict : Complete cluster profiles with customer types
        """
        print("\n" + "=" * 60)
        print("CLUSTER ANALYSIS")
        print("=" * 60)
        
        # Get basic statistics
        self.get_cluster_statistics()
        
        # Analyze characteristics
        self.analyze_cluster_characteristics()
        
        # Define customer types
        self.define_customer_types()
        
        # Generate visualizations
        self.generate_cluster_visualizations()
        
        print("\n" + "=" * 60)
        print("CLUSTER ANALYSIS COMPLETE")
        print("=" * 60)
        
        # Save cluster profiles
        profiles_df = pd.DataFrame(self.cluster_profiles).T
        profiles_df.to_csv('results/cluster_profiles.csv')
        print("\nCluster profiles saved to: results/cluster_profiles.csv")
        
        return self.cluster_profiles

