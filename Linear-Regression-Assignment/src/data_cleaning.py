import pandas as pd

def load_and_clean(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Remove duplicates
    train_df = train_df.drop_duplicates()
    test_df = test_df.drop_duplicates()

    # Encode categorical columns (safe check)
    for df in [train_df, test_df]:
        if 'sex' in df.columns:
            df['sex'] = df['sex'].map({'male': 0, 'female': 1})
        if 'smoker' in df.columns:
            df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
        if 'region' in df.columns:
            df = pd.get_dummies(df, columns=['region'], drop_first=True)

    return train_df, test_df
