"""
# Data Cleaning and Preprocessing
## Step 1: Loading and Understanding the Data

This notebook covers:
- Loading the raw customer data
- Understanding data types and structure
- Identifying missing values
- Initial data quality assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load the data
print("Loading customer data...")
df = pd.read_csv('../data/raw/customer_data.csv')

# Basic information
print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"Shape: {df.shape}")
print(f"Total Records: {len(df):,}")
print(f"Total Features: {len(df.columns)}")

print("\n--- Column Information ---")
print(df.dtypes)

print("\n--- First 10 Rows ---")
print(df.head(10))

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

print("\n--- Categorical Columns ---")
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts().head())

