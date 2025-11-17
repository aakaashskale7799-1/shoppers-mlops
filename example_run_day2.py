from ucimlrepo import fetch_ucirepo
from src.features.numeric_imputer import NumericImputer
from src.features.cat_imputer import CategoricalImputer
from src.features.onehot_encoder import OneHotEncoder
from src.features.bool_converter import BooleanConverter
from src.features.feature_pipeline import FeaturePipeline

# Load dataset
ds = fetch_ucirepo(id=468)
df = ds.data.features
df["Revenue"] = ds.data.targets

# Create pipeline
pipeline = FeaturePipeline([
    NumericImputer(cols=["Administrative", "Informational", "ProductRelated"], strategy="median"),
    CategoricalImputer(cols=["Month", "VisitorType"]),
    BooleanConverter(cols=["Revenue"]),
    OneHotEncoder(cols=["Month", "VisitorType"])
])

pipeline.fit(df)
out = pipeline.transform(df)

print(out.head())
print(out.shape)
