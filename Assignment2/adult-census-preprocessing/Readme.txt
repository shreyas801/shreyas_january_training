Conclusion
Missing Value Handling

For numerical features, the median method performed best because it is robust to extreme values and is not affected by outliers. Since the dataset contained skewed numerical attributes, median imputation preserved the central tendency more accurately than the mean. For categorical features, the mode method was most effective as it replaces missing values with the most frequent category without altering the distribution.

#Categorical Encoding Techniques

Different encoding techniques were suitable for different types of categorical features.

Label Encoding worked well for binary features such as gender because it does not introduce false ordering.

One-Hot Encoding performed best for nominal variables like workclass, as it avoids creating any ordinal relationship between categories.

Ordinal Encoding was appropriate for ordered features such as education level, where the natural order carries meaningful information.

Frequency Encoding was effective for high-cardinality features like occupation, as it reduced dimensionality while retaining occurrence information.

Target Encoding captured the relationship between categorical variables and the target variable by replacing categories with the mean target value, improving feature relevance.

#Feature Scaling

Among all scaling techniques, Z-score standardization was the most effective because it normalized features to have zero mean and unit variance, making it suitable for features with varying ranges and distributions. Min-Max scaling was useful for bounded values, while vector normalization helped when only the relative magnitude of features mattered. However, Z-score scaling provided the most stable and consistent results overall.

Outlier Treatment and Skewness Transformation

Outliers were handled using the Interquartile Range (IQR) method, which effectively removed extreme values without affecting the majority of the data. This improved data consistency and reduced noise. Feature scaling further helped in reducing skewness, stabilizing feature distributions, and improving model readiness.

#Final Justification

The final preprocessing choices were made to balance data integrity, interpretability, and model performance. By selecting robust missing value strategies, appropriate categorical encoding methods, effective scaling techniques, and reliable outlier treatment, the dataset was transformed into a clean and machine-learning-ready format.