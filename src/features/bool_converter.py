import pandas as pd
from .base import BaseTransformer

class BooleanConverter(BaseTransformer):
    """
    Converts True/False to 1/0.
    """

    def __init__(self, cols):
        self.cols = cols

    def fit(self, df):
        return self

    def transform(self, df):
        df = df.copy()
        for col in self.cols:
            df[col] = df[col].astype(int)
        return df
