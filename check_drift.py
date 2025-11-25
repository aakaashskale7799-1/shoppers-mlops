import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from src.monitoring.drift import calculate_psi

# Load training data
dataset = fetch_ucirepo(id=468)
train_df = dataset.data.features

# Load pipeline
pipeline = joblib.load("feature_pipeline.pkl")

# Transform training data
train_processed = pipeline.transform(train_df)

# Simulate new batch
new_data = train_df.sample(200).copy()
new_processed = pipeline.transform(new_data)

# Calculate PSI
psi_score = calculate_psi(train_processed, new_processed)

print("PSI Score:", psi_score)

if psi_score > 0.2:
    print("⚠️ WARNING: Drift Detected. Consider investigating.")
