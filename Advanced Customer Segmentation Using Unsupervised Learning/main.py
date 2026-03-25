"""
AI-Driven Customer Intelligence System for Strategic Business Decision Making
===============================================================================
End-to-end customer segmentation using unsupervised machine learning techniques.

This module orchestrates the complete ML pipeline for customer segmentation:
1. Data Loading & Preprocessing
2. Feature Engineering (RFM features)
3. Exploratory Data Analysis
4. Clustering Model Training (K-Means, Hierarchical, DBSCAN, GMM)
5. Optimal Cluster Selection
6. Dimensionality Reduction (PCA, t-SNE)
7. Cluster Interpretation
8. Business Insights Generation

Author: Student
Project: Customer Segmentation using Unsupervised Learning
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import custom modules
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.clustering_models import ClusteringModels
from src.cluster_analysis import ClusterAnalyzer
from src.visualization import Visualizer
from src.business_insights import BusinessInsightsGenerator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class CustomerSegmentationPipeline:
    """
    Main pipeline class for customer segmentation using unsupervised learning.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        data_path : str
            Path to the raw customer data file
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.feature_engineered_data = None
        self.cluster_results = {}
        self.best_model = None
        self.best_k = None
        self.evaluation_metrics = {}
        
        # Create results directory if not exists
        os.makedirs('results', exist_ok=True)
        
    def load_data(self, file_path=None):
        """
        Load customer data from CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to data file
        """
        if file_path:
            self.data_path = file_path
            
        print("=" * 80)
        print("STEP 1: DATA LOADING")
        print("=" * 80)
        
        if self.data_path and os.path.exists(self.data_path):
            self.raw_data = pd.read_csv(self.data_path)
        else:
            # Generate synthetic customer data for demonstration
            print("No data file found. Generating synthetic customer dataset...")
            self.raw_data = self._generate_customer_data()
            
        print(f"Dataset Shape: {self.raw_data.shape}")
        print(f"Total Records: {len(self.raw_data):,}")
        print(f"Total Features: {len(self.raw_data.columns)}")
        print("\nFirst 5 rows:")
        print(self.raw_data.head())
        print("\nData Types:")
        print(self.raw_data.dtypes)
        
        return self.raw_data
    
    def _generate_customer_data(self):
        """
        Generate synthetic customer data that mimics real-world retail data.
        Includes: Age, Gender, Income, Spending Score, Purchase History, etc.
        """
        np.random.seed(RANDOM_STATE)
        n_customers = 10000
        
        # Generate customer IDs
        customer_ids = [f'CUST_{i:06d}' for i in range(1, n_customers + 1)]
        
        # Demographics
        ages = np.random.normal(40, 15, n_customers).astype(int)
        ages = np.clip(ages, 18, 80)
        
        genders = np.random.choice(['Male', 'Female', 'Other'], n_customers, 
                                   p=[0.48, 0.49, 0.03])
        
        # Financial features
        incomes = np.random.lognormal(10.5, 0.5, n_customers)
        incomes = np.clip(incomes, 15000, 250000)
        
        # Spending score (0-100)
        spending_scores = np.random.normal(50, 25, n_customers)
        spending_scores = np.clip(spending_scores, 1, 99)
        
        # Purchase behavior
        num_purchases = np.random.poisson(15, n_customers)
        num_purchases = np.clip(num_purchases, 1, 100)
        
        # Average transaction amount
        avg_transaction = np.random.lognormal(4.5, 0.8, n_customers)
        avg_transaction = np.clip(avg_transaction, 10, 2000)
        
        # Total spending
        total_spending = num_purchases * avg_transaction
        
        # Recency (days since last purchase)
        recency = np.random.exponential(60, n_customers)
        recency = np.clip(recency, 1, 365)
        
        # Product categories
        categories = ['Electronics', 'Clothing', 'Food & Grocery', 'Home & Garden', 
                      'Sports', 'Books', 'Beauty', 'Toys']
        
        # Primary category preference
        primary_category = np.random.choice(categories, n_customers)
        secondary_category = np.random.choice(categories, n_customers)
        
        # Customer tenure (months)
        tenure = np.random.exponential(24, n_customers)
        tenure = np.clip(tenure, 1, 60)
        
        # Online/Offline preference
        online_preference = np.random.choice(['Online', 'Offline', 'Both'], n_customers,
                                              p=[0.4, 0.35, 0.25])
        
        # Discount sensitivity
        discount_sensitivity = np.random.normal(50, 20, n_customers)
        discount_sensitivity = np.clip(discount_sensitivity, 0, 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Customer_ID': customer_ids,
            'Age': ages,
            'Gender': genders,
            'Income': incomes,
            'Spending_Score': spending_scores,
            'Num_Purchases': num_purchases,
            'Avg_Transaction_Amount': avg_transaction,
            'Total_Spending': total_spending,
            'Recency': recency,
            'Tenure_Months': tenure,
            'Primary_Category': primary_category,
            'Secondary_Category': secondary_category,
            'Channel_Preference': online_preference,
            'Discount_Sensitivity': discount_sensitivity
        })
        
        # Add some missing values (real-world scenario)
        missing_indices = np.random.choice(n_customers, int(n_customers * 0.02), replace=False)
        df.loc[missing_indices, 'Income'] = np.nan
        
        # Add some outliers (real-world scenario)
        outlier_indices = np.random.choice(n_customers, int(n_customers * 0.01), replace=False)
        df.loc[outlier_indices, 'Total_Spending'] = df.loc[outlier_indices, 'Total_Spending'] * 5
        df.loc[outlier_indices, 'Num_Purchases'] = df.loc[outlier_indices, 'Num_Purchases'] * 3
        
        # Save raw data
        df.to_csv('data/raw/customer_data.csv', index=False)
        print(f"Synthetic dataset saved to: data/raw/customer_data.csv")
        
        return df
    
    def preprocess_data(self):
        """
        Preprocess the raw data: outliers, encoding handle missing values,.
        """
        print("\n" + "=" * 80)
        print("STEP 2: DATA PREPROCESSING")
        print("=" * 80)
        
        preprocessor = DataPreprocessor(self.raw_data)
        self.processed_data = preprocessor.process()
        
        print(f"\nProcessed Dataset Shape: {self.processed_data.shape}")
        print(f"Missing Values: {self.processed_data.isnull().sum().sum()}")
        
        return self.processed_data
    
    def engineer_features(self):
        """
        Perform feature engineering including RFM features and derived features.
        """
        print("\n" + "=" * 80)
        print("STEP 3: FEATURE ENGINEERING")
        print("=" * 80)
        
        engineer = FeatureEngineer(self.processed_data)
        self.feature_engineered_data = engineer.engineer_features()
        
        print(f"\nFeature Engineered Dataset Shape: {self.feature_engineered_data.shape}")
        print(f"Total Features: {len(self.feature_engineered_data.columns)}")
        print("\nNew Features Created:")
        new_features = [col for col in self.feature_engineered_data.columns 
                       if col not in self.processed_data.columns]
        for feat in new_features:
            print(f"  - {feat}")
        
        return self.feature_engineered_data
    
    def perform_eda(self):
        """
        Perform Exploratory Data Analysis and generate visualizations.
        """
        print("\n" + "=" * 80)
        print("STEP 4: EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        visualizer = Visualizer(self.feature_engineered_data)
        visualizer.perform_eda()
        
        return self.feature_engineered_data
    
    def train_clustering_models(self):
        """
        Train multiple clustering models and evaluate them.
        """
        print("\n" + "=" * 80)
        print("STEP 5: CLUSTERING MODEL TRAINING")
        print("=" * 80)
        
        # Prepare features for clustering
        clustering_data = self.feature_engineered_data.copy()
        
        # Initialize clustering models
        models = ClusteringModels(clustering_data)
        
        # 1. K-Means Clustering with Elbow Method and Silhouette Analysis
        print("\n--- K-Means Clustering ---")
        kmeans_results = models.kmeans_clustering()
        self.cluster_results['kmeans'] = kmeans_results
        
        # 2. Hierarchical Clustering
        print("\n--- Hierarchical Clustering ---")
        hier_results = models.hierarchical_clustering()
        self.cluster_results['hierarchical'] = hier_results
        
        # 3. DBSCAN Clustering
        print("\n--- DBSCAN Clustering ---")
        dbscan_results = models.dbscan_clustering()
        self.cluster_results['dbscan'] = dbscan_results
        
        # 4. Gaussian Mixture Model
        print("\n--- Gaussian Mixture Model ---")
        gmm_results = models.gmm_clustering()
        self.cluster_results['gmm'] = gmm_results
        
        # Determine best model
        self._select_best_model()
        
        return self.cluster_results
    
    def _select_best_model(self):
        """
        Select the best clustering model based on evaluation metrics.
        """
        print("\n" + "=" * 80)
        print("STEP 6: MODEL SELECTION")
        print("=" * 80)
        
        print("\nModel Evaluation Summary:")
        print("-" * 60)
        
        best_score = -1
        best_model_name = None
        
        for model_name, results in self.cluster_results.items():
            if 'silhouette' in results and results['silhouette'] is not None:
                score = results['silhouette']
                n_clusters = results.get('n_clusters', 'N/A')
                print(f"{model_name.upper():15} | Silhouette: {score:.4f} | Clusters: {n_clusters}")
                
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    self.best_k = n_clusters
        
        self.best_model = best_model_name
        print(f"\n*** BEST MODEL: {self.best_model.upper()} with {self.best_k} clusters ***")
        
    def perform_cluster_analysis(self):
        """
        Analyze and interpret each cluster.
        """
        print("\n" + "=" * 80)
        print("STEP 7: CLUSTER ANALYSIS & INTERPRETATION")
        print("=" * 80)
        
        # Get the best clustering results
        best_results = self.cluster_results.get(self.best_model, self.cluster_results['kmeans'])
        labels = best_results['labels']
        
        # Add cluster labels to data
        analysis_data = self.feature_engineered_data.copy()
        analysis_data['Cluster'] = labels
        
        # Analyze clusters
        analyzer = ClusterAnalyzer(analysis_data)
        cluster_profiles = analyzer.analyze_clusters()
        
        # Generate visualizations
        visualizer = Visualizer(self.feature_engineered_data)
        visualizer.plot_cluster_results(analysis_data, labels, self.best_model)
        
        # PCA visualization
        visualizer.plot_pca_visualization(analysis_data, labels)
        
        # t-SNE visualization
        visualizer.plot_tsne_visualization(analysis_data, labels)
        
        return cluster_profiles
    
    def generate_business_insights(self, cluster_profiles):
        """
        Generate business insights and recommendations for each segment.
        """
        print("\n" + "=" * 80)
        print("STEP 8: BUSINESS INSIGHTS GENERATION")
        print("=" * 80)
        
        best_results = self.cluster_results.get(self.best_model, self.cluster_results['kmeans'])
        labels = best_results['labels']
        
        analysis_data = self.feature_engineered_data.copy()
        analysis_data['Cluster'] = labels
        
        insights_generator = BusinessInsightsGenerator(analysis_data, cluster_profiles)
        insights = insights_generator.generate_insights()
        
        # Print insights
        insights_generator.print_insights(insights)
        
        # Save insights to report
        insights_generator.save_insights_report(insights)
        
        return insights
    
    def run_pipeline(self):
        """
        Execute the complete customer segmentation pipeline.
        """
        start_time = datetime.now()
        
        print("\n" + "=" * 80)
        print("CUSTOMER SEGMENTATION PIPELINE - STARTED")
        print("=" * 80)
        print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Execute pipeline steps
        self.load_data()
        self.preprocess_data()
        self.engineer_features()
        self.perform_eda()
        self.train_clustering_models()
        cluster_profiles = self.perform_cluster_analysis()
        insights = self.generate_business_insights(cluster_profiles)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {duration}")
        print(f"Best Model: {self.best_model.upper()}")
        print(f"Number of Clusters: {self.best_k}")
        print(f"Results saved to: results/")
        
        return {
            'cluster_results': self.cluster_results,
            'best_model': self.best_model,
            'best_k': self.best_k,
            'insights': insights,
            'duration': duration
        }


def main():
    """
    Main entry point for the customer segmentation pipeline.
    """
    # Initialize and run pipeline
    pipeline = CustomerSegmentationPipeline()
    
    # Check for custom data path
    import sys
    data_path = None
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    pipeline.data_path = data_path
    
    # Run the complete pipeline
    results = pipeline.run_pipeline()
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Best Clustering Algorithm: {results['best_model']}")
    print(f"Optimal Number of Clusters: {results['best_k']}")
    print("\nEvaluation Metrics:")
    for model_name, model_results in results['cluster_results'].items():
        if 'silhouette' in model_results:
            print(f"  {model_name}: Silhouette = {model_results['silhouette']:.4f}")
    
    return results


if __name__ == "__main__":
    results = main()

