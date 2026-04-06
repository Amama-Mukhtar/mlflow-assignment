import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris_retraining")

client = MlflowClient()

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

# Train new model
with mlflow.start_run(run_name="Retrain_Run"):
    model = RandomForestClassifier(n_estimators=200, random_state=99)
    model.fit(X_train, y_train)
    new_acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("accuracy", new_acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"New model accuracy: {new_acc:.4f}")

    # Get current production model accuracy
    try:
        prod_runs = client.search_runs(
            experiment_ids=[client.get_experiment_by_name("iris_classification").experiment_id],
            order_by=["metrics.accuracy DESC"]
        )
        prod_acc = prod_runs[0].data.metrics["accuracy"]
        print(f"Production model accuracy: {prod_acc:.4f}")

        if new_acc >= prod_acc:
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri, "IrisClassifier")
            client.set_registered_model_alias("IrisClassifier", "Production", mv.version)
            print(f"New model promoted to Production! Version: {mv.version}")
        else:
            print("New model did NOT improve. Keeping existing Production model.")

    except Exception as e:
        print(f"Could not compare with production model: {e}")