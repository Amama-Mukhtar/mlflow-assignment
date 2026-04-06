from huggingface_hub import HfApi
import os
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Train and save model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to models/model.pkl")

# Upload to Hugging Face
token = os.environ.get("HF_TOKEN")
api = HfApi()

repo_id = "Amama-Mukhtar/iris-classifier"

try:
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
    api.upload_file(
        path_or_fileobj="models/model.pkl",
        path_in_repo="model.pkl",
        repo_id=repo_id,
        token=token,
    )
    print(f"Model deployed to https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"Deployment error: {e}")