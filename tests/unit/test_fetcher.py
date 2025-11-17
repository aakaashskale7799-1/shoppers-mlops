from data.fetcher import Fetcher
import pandas as pd

def test_fetch_dataset(tmp_path, monkeypatch):
    fetcher = Fetcher(cache_file="test_shoppers.csv")

    df = fetcher.fetch_dataset()

    assert isinstance(df, pd.DataFrame)
    assert "Revenue" in df.columns  # target column
    assert len(df) > 5000  # dataset is large
