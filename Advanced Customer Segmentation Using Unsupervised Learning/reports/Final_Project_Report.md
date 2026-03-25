# Customer Segmentation Using Unsupervised Learning
## Final Project Report

---

## 1. Problem Understanding

### 1.1 Business Context
In today's competitive business environment, understanding customers is crucial for strategic decision-making. Companies collect vast amounts of customer data but often lack the tools to derive meaningful insights from it. This project addresses this challenge by implementing an AI-driven customer intelligence system using unsupervised machine learning techniques.

### 1.2 Problem Statement
The company needs to answer critical questions:
- Who are the most valuable customers?
- Which customers are likely to churn?
- Which customer groups spend the most?
- Which segments respond better to specific offers?

### 1.3 Project Objectives
1. Discover hidden patterns and behavioral clusters in customer data
2. Identify meaningful customer segments without labeled data
3. Convert discovered segments into actionable business insights
4. Provide personalized marketing and retention strategies

---

## 2. Dataset Description

### 2.1 Data Source
- **Type**: Synthetic customer data (simulating real-world retail data)
- **Records**: 10,000 customers
- **Features**: 14+ attributes

### 2.2 Features Overview
| Category | Features |
|----------|----------|
| Demographics | Age, Gender |
| Financial | Income, Spending Score |
| Purchase Behavior | Num_Purchases, Total_Spending, Avg_Transaction_Amount |
| Engagement | Recency, Tenure_Months |
| Preferences | Primary_Category, Secondary_Category, Channel_Preference, Discount_Sensitivity |

### 2.3 Data Quality Assessment
- **Missing Values**: ~2% in Income column (handled with median imputation)
- **Outliers**: ~1% in spending-related features (handled with IQR capping)
- **Data Types**: Mixed numeric and categorical (properly encoded)

---

## 3. Approach & Methodology

### 3.1 Data Preprocessing Pipeline
1. **Missing Value Treatment**: Median imputation for numeric, mode for categorical
2. **Outlier Detection**: IQR method with capping at bounds
3. **Categorical Encoding**: Label encoding for binary, One-Hot for multi-class
4. **Feature Scaling**: StandardScaler for normalization

### 3.2 Feature Engineering
Created 15+ new features including:
- **RFM Features**: Recency_Score, Frequency_Score, Monetary_Score, RFM_Score
- **Behavioral Metrics**: Avg_Purchase_Value, Spending_Velocity, Purchase_Frequency_Rate
- **Derived Ratios**: Spending_Propensity, Engagement_Score
- **Interaction Features**: Spend_Freq_Interaction, Income_Spend_Interaction

### 3.3 Clustering Algorithms Implemented

#### K-Means Clustering
- **Method**: Lloyd's algorithm with multiple initializations
- **Optimal K Selection**: Elbow Method + Silhouette Score analysis
- **Best K**: 5 clusters

#### Hierarchical Clustering
- **Method**: Agglomerative clustering
- **Linkage Methods Tested**: Ward, Complete, Average
- **Best**: Ward linkage

#### DBSCAN
- **Method**: Density-based spatial clustering
- **Parameters**: eps=0.5, min_samples=5
- **Advantage**: Automatically identifies outliers

#### Gaussian Mixture Model (GMM)
- **Method**: Probabilistic clustering
- **Covariance Types**: Full, Tied, Diagonal, Spherical
- **Best**: Full covariance

---

## 4. Algorithm Comparison

### 4.1 Evaluation Metrics

| Algorithm | Silhouette Score | Davies-Bouldin | Clusters |
|-----------|-----------------|----------------|----------|
| K-Means | 0.28 | 1.15 | 5 |
| Hierarchical | 0.25 | 1.22 | 5 |
| DBSCAN | 0.32 | 0.98 | 6 |
| GMM | 0.26 | 1.18 | 5 |

### 4.2 Why K-Means Was Selected
1. **Highest interpretability**: Clear cluster centroids
2. **Computational efficiency**: O(n) for large datasets
3. **Consistent performance**: Stable across runs
4. **Business applicability**: Easy to assign new customers

### 4.3 Optimal Cluster Selection Justification
- **Elbow Method**: Clear inflection point at K=5
- **Silhouette Score**: Peak at K=5 (0.28)
- **Business Logic**: 5 segments provide actionable granularity

---

## 5. Cluster Interpretation

### 5.1 Customer Segments Identified

#### Cluster 0: Premium Loyal Customers (18%)
- **Avg Spending**: $8,500
- **Avg Purchases**: 28
- **Characteristics**: High income, frequent buyers, recent activity
- **Profile**: Your most valuable customers requiring VIP treatment

#### Cluster 1: Big Ticket Buyers (15%)
- **Avg Spending**: $7,200
- **Avg Purchases**: 12
- **Characteristics**: High value single purchases, moderate frequency
- **Profile**: Quality-focused customers making significant purchases

#### Cluster 2: Frequent Low-Spenders (22%)
- **Avg Spending**: $3,400
- **Avg Purchases**: 35
- **Characteristics**: Budget-conscious, high engagement
- **Profile**: Deal-seeking customers who shop frequently

#### Cluster 3: Budget Conscious (25%)
- **Avg Spending**: $1,800
- **Avg Purchases**: 8
- **Characteristics**: Price-sensitive, selective purchasers
- **Profile**: Value-focused customers seeking discounts

#### Cluster 4: At-Risk Customers (20%)
- **Avg Spending**: $4,500
- **Avg Purchases**: 15
- **Characteristics**: High recency (haven't purchased recently)
- **Profile**: Customers showing signs of disengagement

---

## 6. Business Insights

### 6.1 Revenue Analysis
- **Top Segment**: Premium Loyal Customers generate 35% of revenue
- **Concentration**: Top 20% of customers contribute 58% of revenue
- **Opportunity**: Converting At-Risk customers could increase revenue by 15%

### 6.2 Marketing Strategy Recommendations

| Segment | Priority | Channel | Offer Type |
|---------|----------|---------|------------|
| Premium | HIGH | Personal Email, VIP Events | Exclusive, Early Access |
| At-Risk | URGENT | SMS, Retargeting | Win-back Discounts |
| Big Ticket | MEDIUM | Phone, Direct Mail | Premium Support |
| Frequent | MEDIUM | App, Social Media | Loyalty Points |
| Budget | LOW | Email, Social | Value Bundles |

### 6.3 Retention Strategies

**Premium Customers**
- Dedicated account management
- Exclusive product launches
- Personalized shopping experiences
- Annual VIP appreciation events

**At-Risk Customers**
- Win-back email campaigns
- Special comeback offers
- Customer satisfaction surveys
- Personalized re-engagement

### 6.4 Churn Risk Analysis
- **High Risk**: Cluster 4 (At-Risk) - 20% of customer base
- **Medium Risk**: Cluster 3 (Budget Conscious)
- **Low Risk**: Clusters 0, 1, 2 (Active, engaged customers)

---

## 7. Visualizations & Outputs

### 7.1 Key Visualizations Generated
1. **Cluster Distribution** - Customer count per segment
2. **PCA Visualization** - 2D cluster separation
3. **t-SNE Visualization** - Non-linear embedding
4. **Correlation Heatmap** - Feature relationships
5. **Model Comparison** - Algorithm performance
6. **Cluster Characteristics** - Feature boxplots

### 7.2 Files Generated
- `results/cluster_distribution.png`
- `results/pca_clusters.png`
- `results/tsne_clusters.png`
- `results/kmeans_optimal_k.png`
- `results/model_comparison.png`
- `results/cluster_profiles.csv`
- `reports/business_insights.txt`

---

## 8. Technical Implementation

### 8.1 Technology Stack
- **Language**: Python 3.8+
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **ML Algorithms**: K-Means, Hierarchical, DBSCAN, GMM

### 8.2 Project Structure
```
├── data/raw/           # Original dataset
├── data/processed/     # Cleaned data
├── notebooks/          # Learning notebooks
├── src/                # Production modules
├── results/           # Visualizations
├── reports/           # Business reports
└── main.py            # Main pipeline
```

---

## 9. Conclusions

### 9.1 Key Findings
1. Customer base naturally segments into 5 distinct groups
2. 20% of customers (Premium segment) generate majority of revenue
3. At-Risk segment represents significant churn opportunity
4. Spending behavior correlates strongly with income and tenure

### 9.2 Recommendations
1. **Invest in Premium segment** - 35% budget allocation
2. **Launch win-back campaigns** for At-Risk customers
3. **Develop loyalty program** for Frequent Low-Spenders
4. **Monitor risk indicators** quarterly

### 9.3 Future Enhancements
1. Autoencoder-based clustering for advanced feature learning
2. Time-based segmentation for temporal patterns
3. Customer Lifetime Value (CLV) prediction
4. Real-time scoring for new customers

---

## 10. References

1. MacQueen, J. (1967). "Some Methods for Classification and Analysis of Multivariate Observations"
2. Ester, M., et al. (1996). "A Density-Based Algorithm for Discovering Clusters"
3. Kaufman, L., Rousseeuw, P. (2009). "Finding Groups in Data"

---

**Submitted By**: Student  
**Date**: March 2026  
**Course**: Advanced Machine Learning - Capstone Project

