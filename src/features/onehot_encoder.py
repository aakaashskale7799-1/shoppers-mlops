import pandas as pd

class OneHotEncoder:
    def __init__(self, cols):
        self.cols = cols
        self.categories_ = {}

    def fit(self, X):
        for col in self.cols:
            self.categories_[col] = X[col].astype(str).unique()
        return self

    def transform(self, X):
        X = X.copy()

        for col in self.cols:
            # Ensure string conversion
            X[col] = X[col].astype(str)

            for cat in self.categories_[col]:
                X[f"{col}_{cat}"] = (X[col] == cat).astype(int)

            X = X.drop(columns=[col])

        return X
