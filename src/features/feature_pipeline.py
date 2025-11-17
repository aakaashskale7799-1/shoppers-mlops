import pandas as pd
from typing import List
from .base import BaseTransformer

class FeaturePipeline:
    """
    Executes transformers sequentially (like sklearn Pipeline).
    """

    def __init__(self, transformers: List[BaseTransformer]):
        self.transformers = transformers

    def fit(self, df: pd.DataFrame):
        for t in self.transformers:
            t.fit(df)
            df = t.transform(df)
        return self

    def transform(self, df: pd.DataFrame):
        for t in self.transformers:
            df = t.transform(df)
        return df
