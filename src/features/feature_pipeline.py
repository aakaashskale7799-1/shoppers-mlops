class FeaturePipeline:
    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, X):
        X_tmp = X.copy()
        for t in self.transformers:
            t.fit(X_tmp)
            X_tmp = t.transform(X_tmp)
        return self

    def transform(self, X):
        X_tmp = X.copy()
        for t in self.transformers:
            X_tmp = t.transform(X_tmp)
        return X_tmp
