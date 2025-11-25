import joblib
import pandas as pd

pipeline = joblib.load("feature_pipeline.pkl")
model = joblib.load("model.pkl")

test = pd.DataFrame([{
    "Administrative": 2,
    "Administrative_Duration": 10,
    "Informational": 0,
    "Informational_Duration": 0,
    "ProductRelated": 5,
    "ProductRelated_Duration": 20,
    "BounceRates": 0.02,
    "ExitRates": 0.05,
    "PageValues": 10,
    "SpecialDay": 0,
    "Month": "Feb",
    "VisitorType": "Returning_Visitor",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 3,
    "Weekend": False
}])

X = pipeline.transform(test)
print("Transformed columns:", list(X.columns))

pred = model.predict(X)
print("Model prediction:", pred)

