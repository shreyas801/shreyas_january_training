"""
Data Preprocessing Module
==========================
Handles all data cleaning and preprocessing tasks:
- Missing value detection and treatment
- Outlier detection and treatment
- Feature scaling/normalization
- Encoding categorical variables

Author: Student
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from scipy import stats

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    A class to handle all data preprocessing tasks for customer segmentation.
    """
    
    def __init__(self, df):
        """
        Initialize the preprocessor with raw data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw customer data
        """
        self.df = df.copy()
        self.original_shape = df.shape
        self.numeric_columns = None
        self.categorical_columns = None
        self.scaler = StandardScaler()
        
    def identify_column_types(self):
        """
        Identify numeric and categorical columns.
        """
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove ID columns from features
        id_columns = [col for col in self.df.columns if 'ID' in col or 'id' in col]
        for col in id_columns:
            if col in self.numeric_columns:
                self.numeric_columns.remove(col)
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
        
        print(f"Numeric columns ({len(self.numeric_columns)}): {self.numeric_columns}")
        print(f"Categorical columns ({len(self.categorical_columns)}): {self.categorical_columns}")
        
        return self.numeric_columns, self.categorical_columns
    
    def handle_missing_values(self):
        """
        Handle missing values using appropriate strategies.
        """
        print("\n--- Handling Missing Values ---")
        
        # Check missing values
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        print("Missing values summary:")
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_df) > 0:
            print(missing_df)
        else:
            print("No missing values found.")
        
        # Handle missing values based on column type
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if col in self.numeric_columns:
                    # Use median for numeric columns (robust to outliers)
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"Filled {col} missing values with median: {median_val:.2f}")
                else:
                    # Use mode for categorical columns
                    mode_val = self.df[col].mode()[0]
                    self.df[col].fillna(mode_val, inplace=True)
                    print(f"Filled {col} missing values with mode: {mode_val}")
        
        return self.df
    
    def detect_outliers(self, method='iqr', threshold=3):
        """
        Detect outliers using IQR or Z-score method.
        
        Parameters:
        -----------
        method : str
            'iqr' for Interquartile Range or 'zscore' for Z-score
        threshold : float
            Threshold for outlier detection (1.5 for IQR, 3 for Z-score)
        """
        print("\n--- Outlier Detection ---")
        print(f"Method: {method.upper()}")
        
        outlier_summary = {}
        
        for col in self.numeric_columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            else:  # zscore
                z_scores = np.abs(stats.zscore(self.df[col]))
                outliers = z_scores > threshold
            
            outlier_count = outliers.sum()
            outlier_pct = (outlier_count / len(self.df)) * 100
            
            if outlier_count > 0:
                outlier_summary[col] = {
                    'count': outlier_count,
                    'percentage': outlier_pct,
                    'lower_bound': lower_bound if method == 'iqr' else None,
                    'upper_bound': upper_bound if method == 'iqr' else None
                }
        
        print(f"Columns with outliers: {len(outlier_summary)}")
        for col, stats_dict in outlier_summary.items():
            print(f"  {col}: {stats_dict['count']} outliers ({stats_dict['percentage']:.2f}%)")
        
        return outlier_summary
    
    def handle_outliers(self, method='capping', outlier_summary=None):
        """
        Handle outliers using specified method.
        
        Parameters:
        -----------
        method : str
            'capping' to cap outliers at bounds, 'remove' to remove rows
        outlier_summary : dict
            Dictionary of outlier information from detect_outliers
        """
        print(f"\n--- Handling Outliers (Method: {method}) ---")
        
        if outlier_summary is None:
            outlier_summary = self.detect_outliers()
        
        for col in self.numeric_columns:
            if col in outlier_summary:
                if method == 'capping':
                    stats_dict = outlier_summary[col]
                    if stats_dict['lower_bound'] is not None:
                        self.df[col] = self.df[col].clip(
                            lower=stats_dict['lower_bound'],
                            upper=stats_dict['upper_bound']
                        )
                        print(f"Capped {col} outliers within bounds")
                elif method == 'remove':
                    stats_dict = outlier_summary[col]
                    if stats_dict['lower_bound'] is not None:
                        self.df[col] = self.df[col].clip(
                            lower=stats_dict['lower_bound'],
                            upper=stats_dict['upper_bound']
                        )
        
        return self.df
    
    def encode_categorical_variables(self):
        """
        Encode categorical variables using Label Encoding and One-Hot Encoding.
        """
        print("\n--- Encoding Categorical Variables ---")
        
        # Use Label Encoding for binary categories
        # Use One-Hot Encoding for multi-class categories
        encoded_dfs = []
        
        for col in self.categorical_columns:
            unique_vals = self.df[col].nunique()
            print(f"{col}: {unique_vals} unique values")
            
            if unique_vals <= 2:
                # Binary encoding for 2 categories
                le = LabelEncoder()
                self.df[col + '_encoded'] = le.fit_transform(self.df[col])
                print(f"  -> Label Encoded: {col} -> {col}_encoded")
            else:
                # One-Hot Encoding for more than 2 categories
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                encoded_dfs.append(dummies)
                print(f"  -> One-Hot Encoded: {col}")
        
        # Add one-hot encoded columns
        for dummy_df in encoded_dfs:
            self.df = pd.concat([self.df, dummy_df], axis=1)
        
        # Remove original categorical columns
        self.df.drop(columns=self.categorical_columns, inplace=True)
        
        print(f"\nFinal columns after encoding: {len(self.df.columns)}")
        
        return self.df
    
    def scale_features(self, method='standard'):
        """
        Scale numerical features.
        
        Parameters:
        -----------
        method : str
            'standard' for StandardScaler, 'minmax' for MinMaxScaler
        """
        print(f"\n--- Feature Scaling ({method}) ---")
        
        # Update numeric columns list after encoding
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        # Scale all numeric columns
        self.df[self.numeric_columns] = self.scaler.fit_transform(self.df[self.numeric_columns])
        
        print(f"Scaled {len(self.numeric_columns)} numeric features")
        
        return self.df
    
    def process(self):
        """
        Execute the complete preprocessing pipeline.
        
        Returns:
        --------
        pd.DataFrame
            Fully preprocessed dataset ready for clustering
        """
        print("\n" + "=" * 60)
        print("DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        print(f"\nOriginal dataset shape: {self.original_shape}")
        
        # Step 1: Identify column types
        self.identify_column_types()
        
        # Step 2: Handle missing values
        self.handle_missing_values()
        
        # Step 3: Detect and handle outliers
        outlier_summary = self.detect_outliers()
        self.handle_outliers(method='capping', outlier_summary=outlier_summary)
        
        # Step 4: Encode categorical variables
        self.encode_categorical_variables()
        
        # Step 5: Scale features
        self.scale_features(method='standard')
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        print(f"Final dataset shape: {self.df.shape}")
        
        return self.df

