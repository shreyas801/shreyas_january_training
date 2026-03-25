import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(df):
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['number'])

    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Features Only)")
    plt.show()

    # Scatter plot if columns exist
    if 'age' in df.columns and 'charges' in df.columns:
        sns.scatterplot(x='age', y='charges', data=df)
        plt.title("Age vs Charges")
        plt.show()
