import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from src.models.evaluator import Evaluator

class ModelTrainer:
    """
    Handles model training and MLflow logging.
    """

    def __init__(self, experiment_name="online_shoppers_experiment"):
        mlflow.set_experiment(experiment_name)

    def train(self, model, X, y, params=None):
        
        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        with mlflow.start_run():

            # Log hyperparameters
            if params:
                model.set_params(**params)
                for k, v in params.items():
                    mlflow.log_param(k, v)

            # Train
            model.fit(X_train, y_train)

            # Predictions
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)

            # Evaluate
            evaluator = Evaluator()
            metrics = evaluator.evaluate(y_val, y_pred, y_prob)

            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            # Log model artifact
            mlflow.sklearn.log_model(model, artifact_path="model")

            return model, metrics
