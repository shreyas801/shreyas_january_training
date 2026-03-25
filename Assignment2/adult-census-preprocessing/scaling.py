from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, Normalizer

def min_max_scaling(df, col):
    scaler = MinMaxScaler()
    df[col] = scaler.fit_transform(df[[col]])
    return df

def max_abs_scaling(df, col):
    scaler = MaxAbsScaler()
    df[col] = scaler.fit_transform(df[[col]])
    return df

def z_score_scaling(df, col):
    scaler = StandardScaler()
    df[col] = scaler.fit_transform(df[[col]])
    return df

def vector_normalization(df, cols):
    normalizer = Normalizer()
    df[cols] = normalizer.fit_transform(df[cols])
    return df
