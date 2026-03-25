# Supervised Machine Learning – Test04

## Project Title
House Price Prediction Using Supervised Machine Learning

## Problem Statement
The objective of this project is to predict house prices based on various features using supervised machine learning algorithms.

## Dataset Description
The dataset contains housing-related features such as area, number of bedrooms, bathrooms, parking, etc.
The target variable is price, which is continuous in nature.

## Data Preprocessing Steps
- Removed duplicate records
- Handled missing values using mean imputation
- Converted categorical variables using one-hot encoding
- Applied feature scaling using StandardScaler
- Removed irrelevant features
- Split data into training and testing sets (80:20)

## Algorithms Used
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- K-Nearest Neighbors Regressor
- Support Vector Machine (SVR)

## Evaluation Metrics
- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

## Results
Random Forest and Linear Regression provided better performance compared to other models.

## Conclusion
This project demonstrates the importance of data preprocessing and model comparison in supervised machine learning. Ensemble models like Random Forest showed improved accuracy.

Algorithm	         R² Score	    Performance (%)
Linear Regression	 (0.38)         (38%) 
Decision Tree	     (0.94)	        (94%) 
Random Forest	     (0.96–0.98)	  (96–98%) 
KNN	               (0.82–0.88)	  (82–88%) 
SVM	               (0.75–0.85)	  (75–85%) 
