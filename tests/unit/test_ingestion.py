import pandas as pd
from pathlib import Path
from data.ingestion import DataIngestor

def test_ingestion(tmp_path):
    file1 = tmp_path / "f1.csv"
    file2 = tmp_path / "f2.csv"

    pd.DataFrame({"x":[1,2]}).to_csv(file1, index=False)
    pd.DataFrame({"x":[3,4]}).to_csv(file2, index=False)

    ing = DataIngestor([file1, file2])
    df = ing.load()

    assert len(df) == 4
    assert "x" in df.columns
