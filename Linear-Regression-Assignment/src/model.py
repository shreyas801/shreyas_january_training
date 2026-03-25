from sklearn.linear_model import LinearRegression

def train_model(train_df):
    X_train = train_df.drop('charges', axis=1)
    y_train = train_df['charges']

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model
