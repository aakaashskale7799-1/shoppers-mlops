from ucimlrepo import fetch_ucirepo
import pandas as pd
import joblib
# Feature engineering
from src.features.numeric_imputer import NumericImputer
from src.features.cat_imputer import CategoricalImputer
from src.features.onehot_encoder import OneHotEncoder
from src.features.feature_pipeline import FeaturePipeline
from src.features.bool_converter import BooleanConverter
# Modeling
from sklearn.ensemble import RandomForestClassifier
from src.models.trainer import ModelTrainer

def run_training():

    # STEP 1: Load dataset
    dataset = fetch_ucirepo(id=468)
    X = dataset.data.features
    y = dataset.data.targets["Revenue"].astype(int)

    # STEP 2: Feature groups
    numeric_cols = [
        "Administrative", "Administrative_Duration",
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay"
    ]
    categorical_cols = [
        "Month", "VisitorType", "OperatingSystems","Browser", "Region", "TrafficType"
    ]

    boolean_cols = ["Weekend"]
    # STEP 2: Feature Pipeline
    # STEP 3: Feature Pipeline
    pipeline = FeaturePipeline([
        NumericImputer(cols=numeric_cols),
        CategoricalImputer(cols=categorical_cols),
        BooleanConverter(cols=boolean_cols),
        OneHotEncoder(cols=categorical_cols + boolean_cols)
    ])

    pipeline.fit(X)
    X_processed = pipeline.transform(X)

    # STEP 3: Train Model
    trainer = ModelTrainer(experiment_name="online_shoppers_exp")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    trained_model, metrics = trainer.train(model, X_processed, y)

    # STEP 5: SAVE ARTIFACTS
    joblib.dump(pipeline, "feature_pipeline.pkl")
    joblib.dump(trained_model, "model.pkl")

    print("\n🎉 TRAINING COMPLETED")
    print("📊 Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    run_training()
