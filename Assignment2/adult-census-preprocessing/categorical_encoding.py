import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def label_encoding(df, col):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    return df

def one_hot_encoding(df, col):
    return pd.get_dummies(df, columns=[col])

def ordinal_encoding(df, col):
    order = ['Preschool','HS-grad','Bachelors','Masters','Doctorate']
    oe = OrdinalEncoder(categories=[order])
    df[col] = oe.fit_transform(df[[col]])
    return df

def frequency_encoding(df, col):
    freq = df[col].value_counts()
    df[col] = df[col].map(freq)
    return df

# def target_encoding(df, col, target):
#     mean_map = df.groupby(col)[target].mean()
#     df[col] = df[col].map(mean_map)
#     return df

def target_encoding(df, col, target):
    df[target] = df[target].astype(float)   # FORCE numeric
    mean_map = df.groupby(col)[target].mean()
    df[col] = df[col].map(mean_map)
    return df
