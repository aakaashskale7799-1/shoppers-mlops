import pandas as pd
import pytest
from data.validator import DataValidator, ValidationError

def test_required_columns():
    df = pd.DataFrame({"a":[1], "b":[2]})
    v = DataValidator()

    v.check_required_columns(df, ["a","b"])

    with pytest.raises(ValidationError):
        v.check_required_columns(df, ["a","b","c"])

def test_missing_threshold():
    df = pd.DataFrame({
        "a":[1, None],
        "b":[None, None],
    })

    v = DataValidator(missing_threshold=0.5)

    with pytest.raises(ValidationError):
        v.assert_missing_below_threshold(df)
