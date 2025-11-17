import pandas as pd
from features.feature_pipeline import FeaturePipeline
from features.numeric_imputer import NumericImputer

def test_pipeline_runs():
    df = pd.DataFrame({"a":[1,None,3]})
    pipeline = FeaturePipeline([NumericImputer(["a"])])
    pipeline.fit(df)
    out = pipeline.transform(df)
    assert out["a"].isnull().sum() == 0
