import pandas as pd
from features.onehot_encoder import OneHotEncoder

def test_onehot_encoder():
    df = pd.DataFrame({"color":["red","blue","red"]})
    enc = OneHotEncoder(cols=["color"])
    enc.fit(df)
    out = enc.transform(df)
    assert "color_red" in out.columns or "color_blue" in out.columns
