import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def handle_missing_values(df):
    df.replace('?', np.nan, inplace=True)
    df['age'] = df['age'].fillna(df['age'].median())
    df['workclass'] = df['workclass'].fillna(df['workclass'].mode()[0])
    df['occupation'] = df['occupation'].fillna(df['occupation'].mode()[0])
    return df

def fix_data_types(df):
    # df['income'] = df['income'].astype('category')
    return df

def remove_duplicates(df):
    return df.drop_duplicates()

def treat_outliers(df):
    Q1 = df['age'].quantile(0.25)
    Q3 = df['age'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['age'] >= Q1 - 1.5*IQR) & (df['age'] <= Q3 + 1.5*IQR)]
    return df

def drop_irrelevant_features(df):
    return df.drop(columns=['fnlwgt'], errors='ignore')
