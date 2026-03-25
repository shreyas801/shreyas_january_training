from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def evaluate_model(model, test_df):
    X_test = test_df.drop('charges', axis=1)
    y_test = test_df['charges']

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)

    coef_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Coefficient': model.coef_
    })

    print("\nFeature Coefficients:")
    print(coef_df)
