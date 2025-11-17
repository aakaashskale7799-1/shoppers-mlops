import pandas as pd
from .base import BaseTransformer

class OneHotEncoder(BaseTransformer):
    """
    Applies one-hot encoding to selected categorical columns.
    """

    def __init__(self, cols):
        self.cols = cols
        self.final_cols = None

    def fit(self, df):
        # Fit learns nothing except output columns
        dummies = pd.get_dummies(df[self.cols], drop_first=True)
        self.final_cols = dummies.columns.tolist()

    def transform(self, df):
        df = df.copy()
        dummies = pd.get_dummies(df[self.cols], drop_first=True)

        # Ensure consistent column order
        for col in self.final_cols:
            if col not in dummies:
                dummies[col] = 0

        df = df.drop(columns=self.cols)
        df = pd.concat([df, dummies[self.final_cols]], axis=1)
        return df
