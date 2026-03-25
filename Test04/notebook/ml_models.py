# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv("../dataset/train.csv")

print("\nColumns in dataset:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

# ===============================
# DATA PREPROCESSING
# ===============================

# Remove duplicates
df.drop_duplicates(inplace=True)

# Fill missing numeric values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Convert categorical columns to numbers
df = pd.get_dummies(df, drop_first=True)

# ===============================
# SPLIT FEATURES & TARGET
# ===============================
X = df.drop("TARGET(PRICE_IN_LACS)", axis=1)
y = df["TARGET(PRICE_IN_LACS)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (important for KNN & SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# MODELS
# ===============================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVM": SVR(kernel='rbf', C=100, gamma=0.1)
}

# ===============================
# TRAIN & EVALUATE
# ===============================
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    results.append([name, r2, rmse, mae])

    print(f"\n{name}")
    print("R2 Score :", r2)
    print("RMSE     :", rmse)
    print("MAE      :", mae)

# ===============================
# MODEL COMPARISON TABLE
# ===============================
results_df = pd.DataFrame(results, columns=["Model", "R2 Score", "RMSE", "MAE"])
print("\n\n===== Model Comparison =====")
print(results_df.sort_values(by="R2 Score", ascending=False))
