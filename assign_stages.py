import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

model_name = "IrisClassifier"
version = "1"

# First assign to Staging
client.set_registered_model_alias(model_name, "Staging", version)
print(f"Model version {version} assigned to Staging")

# Then promote to Production
client.set_registered_model_alias(model_name, "Production", version)
print(f"Model version {version} assigned to Production")

# Confirm
model_version = client.get_model_version(model_name, version)
print(f"Model name: {model_version.name}")
print(f"Version: {model_version.version}")
print(f"Status: {model_version.status}")