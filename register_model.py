import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Get best run by accuracy
experiment = client.get_experiment_by_name("iris_classification")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"]
)

best_run = runs[0]
print(f"Best Run ID: {best_run.info.run_id}")
print(f"Best Accuracy: {best_run.data.metrics['accuracy']}")
print(f"Model: {best_run.data.params['model_type']}")

# Register the model
model_uri = f"runs:/{best_run.info.run_id}/model"
mv = mlflow.register_model(model_uri, "IrisClassifier")
print(f"Registered model version: {mv.version}")