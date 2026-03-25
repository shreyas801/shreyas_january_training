"""
Feature Engineering Module
==========================
Handles all feature engineering tasks:
- RFM features (Recency, Frequency, Monetary)
- Behavioral features
- Derived ratios and metrics

Author: Student
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    A class to perform feature engineering for customer segmentation.
    """
    
    def __init__(self, df):
        """
        Initialize the feature engineer with preprocessed data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Preprocessed customer data
        """
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
        
    def create_rfm_features(self):
        """
        Create RFM (Recency, Frequency, Monetary) features.
        
        RFM Analysis is a customer segmentation technique based on:
        - Recency: How recently a customer made a purchase
        - Frequency: How often a customer makes purchases
        - Monetary: How much money a customer spends
        """
        print("\n--- Creating RFM Features ---")
        
        # Check if required columns exist
        required_cols = ['Recency', 'Num_Purchases', 'Total_Spending']
        available_cols = [col for col in required_cols if col in self.df.columns]
        
        if len(available_cols) == len(required_cols):
            # Recency features (already have 'Recency' column)
            self.df['Recency_Score'] = pd.cut(
                self.df['Recency'], 
                bins=5, 
                labels=[5, 4, 3, 2, 1]
            ).astype(int)
            
            # Frequency features (already have 'Num_Purchases')
            self.df['Frequency_Score'] = pd.cut(
                self.df['Num_Purchases'], 
                bins=5, 
                labels=[1, 2, 3, 4, 5]
            ).astype(int)
            
            # Monetary features (already have 'Total_Spending')
            self.df['Monetary_Score'] = pd.cut(
                self.df['Total_Spending'], 
                bins=5, 
                labels=[1, 2, 3, 4, 5]
            ).astype(int)
            
            # Combined RFM Score
            self.df['RFM_Score'] = (
                self.df['Recency_Score'] + 
                self.df['Frequency_Score'] + 
                self.df['Monetary_Score']
            )
            
            print("RFM features created:")
            print("  - Recency_Score")
            print("  - Frequency_Score")
            print("  - Monetary_Score")
            print("  - RFM_Score")
        else:
            print(f"Warning: Not all RFM columns available. Found: {available_cols}")
        
        return self.df
    
    def create_behavioral_features(self):
        """
        Create behavioral features based on customer purchase patterns.
        """
        print("\n--- Creating Behavioral Features ---")
        
        # Average purchase value
        if 'Num_Purchases' in self.df.columns and 'Total_Spending' in self.df.columns:
            self.df['Avg_Purchase_Value'] = (
                self.df['Total_Spending'] / self.df['Num_Purchases']
            ).replace([np.inf, -np.inf], 0).fillna(0)
            print("  - Avg_Purchase_Value")
        
        # Spending velocity (spending per month of tenure)
        if 'Total_Spending' in self.df.columns and 'Tenure_Months' in self.df.columns:
            self.df['Spending_Velocity'] = (
                self.df['Total_Spending'] / self.df['Tenure_Months']
            ).replace([np.inf, -np.inf], 0).fillna(0)
            print("  - Spending_Velocity")
        
        # Purchase frequency rate (purchases per month)
        if 'Num_Purchases' in self.df.columns and 'Tenure_Months' in self.df.columns:
            self.df['Purchase_Frequency_Rate'] = (
                self.df['Num_Purchases'] / self.df['Tenure_Months']
            ).replace([np.inf, -np.inf], 0).fillna(0)
            print("  - Purchase_Frequency_Rate")
        
        # Customer engagement score (composite)
        engagement_cols = []
        if 'Spending_Score' in self.df.columns:
            engagement_cols.append('Spending_Score')
        if 'Num_Purchases' in self.df.columns:
            engagement_cols.append('Num_Purchases')
            
        if len(engagement_cols) > 0:
            # Normalize and combine
            self.df['Engagement_Score'] = self.df[engagement_cols].mean(axis=1)
            print("  - Engagement_Score")
        
        # Value segment indicator (high spenders)
        if 'Total_Spending' in self.df.columns:
            spending_75 = self.df['Total_Spending'].quantile(0.75)
            self.df['Is_High_Value'] = (self.df['Total_Spending'] >= spending_75).astype(int)
            print("  - Is_High_Value")
        
        # At-risk indicator (high recency = old customer = at risk)
        if 'Recency' in self.df.columns:
            recency_75 = self.df['Recency'].quantile(0.75)
            self.df['Is_At_Risk'] = (self.df['Recency'] >= recency_75).astype(int)
            print("  - Is_At_Risk")
        
        # Churn probability estimate based on recency
        if 'Recency' in self.df.columns:
            recency_mean = self.df['Recency'].mean()
            recency_std = self.df['Recency'].std()
            self.df['Churn_Risk_Score'] = (
                (self.df['Recency'] - recency_mean) / recency_std
            ).clip(0, 3) / 3  # Normalize to 0-1
            print("  - Churn_Risk_Score")
        
        return self.df
    
    def create_demographic_features(self):
        """
        Create demographic-based derived features.
        """
        print("\n--- Creating Demographic Features ---")
        
        # Age groups
        if 'Age' in self.df.columns:
            self.df['Age_Group'] = pd.cut(
                self.df['Age'],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
            )
            # Convert to numeric for clustering
            age_mapping = {'18-25': 1, '26-35': 2, '36-45': 3, '46-55': 4, '56-65': 5, '65+': 6}
            self.df['Age_Group_Encoded'] = self.df['Age_Group'].map(age_mapping)
            print("  - Age_Group, Age_Group_Encoded")
        
        # Income segments
        if 'Income' in self.df.columns:
            self.df['Income_Segment'] = pd.qcut(
                self.df['Income'],
                q=4,
                labels=['Low', 'Medium', 'High', 'Premium']
            )
            income_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Premium': 4}
            self.df['Income_Segment_Encoded'] = self.df['Income_Segment'].map(income_mapping)
            print("  - Income_Segment, Income_Segment_Encoded")
        
        # Spending propensity (ratio of spending to income)
        if 'Total_Spending' in self.df.columns and 'Income' in self.df.columns:
            self.df['Spending_Propensity'] = (
                self.df['Total_Spending'] / self.df['Income']
            ).replace([np.inf, -np.inf], 0).fillna(0)
            print("  - Spending_Propensity")
        
        return self.df
    
    def create_interaction_features(self):
        """
        Create interaction features between key variables.
        """
        print("\n--- Creating Interaction Features ---")
        
        # Spending Score * Frequency
        if 'Spending_Score' in self.df.columns and 'Num_Purchases' in self.df.columns:
            self.df['Spend_Freq_Interaction'] = (
                self.df['Spending_Score'] * self.df['Num_Purchases']
            )
            print("  - Spend_Freq_Interaction")
        
        # Income * Spending Score
        if 'Income' in self.df.columns and 'Spending_Score' in self.df.columns:
            self.df['Income_Spend_Interaction'] = (
                self.df['Income'] * self.df['Spending_Score']
            )
            print("  - Income_Spend_Interaction")
        
        # Recency * Frequency (Customer lifetime value indicator)
        if 'Recency' in self.df.columns and 'Num_Purchases' in self.df.columns:
            self.df['Recency_Freq_Product'] = (
                self.df['Recency'] * self.df['Num_Purchases']
            )
            print("  - Recency_Freq_Product")
        
        # Total Value Potential (future value indicator)
        if 'Num_Purchases' in self.df.columns and 'Avg_Transaction_Amount' in self.df.columns:
            self.df['Value_Potential'] = (
                self.df['Num_Purchases'] * self.df['Avg_Transaction_Amount']
            )
            print("  - Value_Potential")
        
        return self.df
    
    def select_features_for_clustering(self):
        """
        Select the most relevant features for clustering.
        """
        print("\n--- Selecting Features for Clustering ---")
        
        # Define columns to exclude from clustering
        exclude_cols = ['Customer_ID']
        
        # Get all numeric columns
        feature_cols = [
            col for col in self.df.columns 
            if col not in exclude_cols and self.df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
        
        # Select key features for clustering
        priority_features = [
            'Age', 'Income', 'Spending_Score', 'Num_Purchases',
            'Total_Spending', 'Recency', 'Avg_Transaction_Amount',
            'Tenure_Months', 'RFM_Score', 'Avg_Purchase_Value',
            'Spending_Velocity', 'Purchase_Frequency_Rate',
            'Engagement_Score', 'Spending_Propensity',
            'Spend_Freq_Interaction', 'Income_Spend_Interaction'
        ]
        
        # Use available priority features
        selected_features = [col for col in priority_features if col in feature_cols]
        
        print(f"Selected {len(selected_features)} features for clustering:")
        for feat in selected_features:
            print(f"  - {feat}")
        
        return self.df, selected_features
    
    def engineer_features(self):
        """
        Execute the complete feature engineering pipeline.
        
        Returns:
        --------
        pd.DataFrame
            Dataset with engineered features
        """
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        
        print(f"\nOriginal features: {len(self.original_columns)}")
        
        # Step 1: Create RFM features
        self.create_rfm_features()
        
        # Step 2: Create behavioral features
        self.create_behavioral_features()
        
        # Step 3: Create demographic features
        self.create_demographic_features()
        
        # Step 4: Create interaction features
        self.create_interaction_features()
        
        # Step 5: Select features for clustering
        self.df, selected_features = self.select_features_for_clustering()
        
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING COMPLETE")
        print("=" * 60)
        print(f"Total features after engineering: {len(self.df.columns)}")
        print(f"Features selected for clustering: {len(selected_features)}")
        
        # Save processed data
        self.df.to_csv('data/processed/feature_engineered_data.csv', index=False)
        print("\nFeature engineered data saved to: data/processed/feature_engineered_data.csv")
        
        return self.df

