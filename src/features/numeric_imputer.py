import pandas as pd
from .base import BaseTransformer

class NumericImputer(BaseTransformer):
    """
    Imputes numeric columns using mean or median.
    """

    def __init__(self, cols, strategy="median"):
        self.cols = cols
        self.strategy = strategy
        self.values = {}

    def fit(self, df: pd.DataFrame):
        for col in self.cols:
            if self.strategy == "mean":
                self.values[col] = df[col].mean()
            else:
                self.values[col] = df[col].median()

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        for col, val in self.values.items():
            df[col] = df[col].fillna(val)
        return df
