"""
# Exploratory Data Analysis (EDA)
## Step 2: Understanding Customer Patterns

This notebook covers:
- Univariate analysis (distribution of individual features)
- Bivariate analysis (relationships between features)
- Multivariate analysis (correlations)
- Visualizing patterns and trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load processed data
print("Loading processed data...")
df = pd.read_csv('../data/processed/feature_engineered_data.csv')

print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# ============== UNIVARIATE ANALYSIS ==============
print("\n--- Univariate Analysis ---")

# Select key numeric features
key_features = ['Age', 'Income', 'Spending_Score', 'Total_Spending', 
                'Num_Purchases', 'Recency', 'Avg_Transaction_Amount']

available_features = [f for f in key_features if f in df.columns]

# Distribution plots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, feature in enumerate(available_features):
    axes[idx].hist(df[feature], bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution of {feature}')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Frequency')

# Hide unused subplots
for idx in range(len(available_features), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('../results/eda_distributions.png', dpi=150)
plt.show()

# ============== BIVARIATE ANALYSIS ==============
print("\n--- Bivariate Analysis ---")

# Key relationships
bivariate_pairs = [
    ('Income', 'Total_Spending'),
    ('Spending_Score', 'Num_Purchases'),
    ('Age', 'Spending_Score'),
    ('Recency', 'Total_Spending')
]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, (x, y) in enumerate(bivariate_pairs):
    ax = axes[idx // 2, idx % 2]
    ax.scatter(df[x], df[y], alpha=0.5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f'{x} vs {y}')
    
    # Add correlation
    corr = df[x].corr(df[y])
    ax.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
               fontsize=12, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('../results/eda_bivariate.png', dpi=150)
plt.show()

# ============== CORRELATION ANALYSIS ==============
print("\n--- Correlation Analysis ---")

# Select numeric columns
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
           cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig('../results/eda_correlation.png', dpi=150)
plt.show()

# Print top correlations
print("\nTop Positive Correlations:")
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_pairs.append({
            'Feature 1': corr_matrix.columns[i],
            'Feature 2': corr_matrix.columns[j],
            'Correlation': corr_matrix.iloc[i, j]
        })

corr_df = pd.DataFrame(corr_pairs)
corr_df = corr_df.sort_values('Correlation', ascending=False)
print(corr_df.head(10).to_string(index=False))

print("\nEDA completed!")

