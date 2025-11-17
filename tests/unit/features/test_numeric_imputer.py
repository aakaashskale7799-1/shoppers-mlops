import pandas as pd
from features.numeric_imputer import NumericImputer

def test_numeric_imputer():
    df = pd.DataFrame({"a":[1, None, 3]})
    imputer = NumericImputer(cols=["a"])
    imputer.fit(df)
    out = imputer.transform(df)
    assert out["a"].isnull().sum() == 0
