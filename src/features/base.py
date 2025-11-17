from abc import ABC, abstractmethod
import pandas as pd

class BaseTransformer(ABC):
    """
    Abstract base class for all feature transformers.
    Each transformer must implement fit() and transform().
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
