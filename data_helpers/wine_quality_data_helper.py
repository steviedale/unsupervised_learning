import pandas as pd


def load_wine_quality_data():
    df = pd.read_csv('datasets/wine_quality.csv', sep=';')
    len(df)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.columns
    feature_columns = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
    label_column = 'quality'
    X = df[feature_columns].values
    y = df[label_column].values
    n = int(0.8 * len(X))
    X_train = X[:n]
    y_train = y[:n]
    X_test = X[n:]
    y_test = y[n:]
    return X_train, y_train, X_test, y_test