import pandas as pd

class Predictor:
    def __init__(self, model, pipeline):
        self.model = model
        self.pipeline = pipeline

    def predict(self, df: pd.DataFrame):
        transformed = self.pipeline.transform(df)
        prob = self.model.predict_proba(transformed)[:, 1]
        pred = (prob > 0.5).astype(int)
        return pred, prob
