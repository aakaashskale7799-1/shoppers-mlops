class BooleanConverter:
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].astype(bool).astype(str)
        return X
