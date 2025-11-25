import pandas as pd

class CategoricalImputer:
    def __init__(self, cols, fill_value="Unknown"):
        self.cols = cols
        self.fill_value = fill_value

    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].fillna(self.fill_value).astype(str)
        return X
