import pandas as pd
from .base import BaseTransformer

class StandardScaler(BaseTransformer):
    """
    Standardizes numeric columns (z-score).
    """

    def __init__(self, cols):
        self.cols = cols
        self.means = {}
        self.stds = {}

    def fit(self, df):
        for col in self.cols:
            self.means[col] = df[col].mean()
            self.stds[col] = df[col].std()

    def transform(self, df):
        df = df.copy()
        for col in self.cols:
            df[col] = (df[col] - self.means[col]) / (self.stds[col] + 1e-6)
        return df
