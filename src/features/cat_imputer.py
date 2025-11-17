import pandas as pd
from .base import BaseTransformer

class CategoricalImputer(BaseTransformer):
    """
    Imputes categorical columns using mode.
    """

    def __init__(self, cols):
        self.cols = cols
        self.values = {}

    def fit(self, df):
        for col in self.cols:
            self.values[col] = df[col].mode().iloc[0]

    def transform(self, df):
        df = df.copy()
        for col, val in self.values.items():
            df[col] = df[col].fillna(val)
        return df
