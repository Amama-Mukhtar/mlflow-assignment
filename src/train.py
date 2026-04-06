import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

# Setup MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris_classification")

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Save dataset to data folder
os.makedirs("data", exist_ok=True)
X.to_csv("data/features.csv", index=False)
y.to_csv("data/labels.csv", index=False)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Model 1: Random Forest (3 runs with different n_estimators) ──
for n_estimators in [50, 100, 150]:
    with mlflow.start_run(run_name=f"RandomForest_n{n_estimators}"):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted")
        rec  = recall_score(y_test, y_pred, average="weighted")

        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"RandomForest n={n_estimators} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

# ── Model 2: Logistic Regression (3 runs with different C values) ──
for C in [0.1, 1.0, 10.0]:
    with mlflow.start_run(run_name=f"LogisticRegression_C{C}"):
        model = LogisticRegression(C=C, max_iter=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted")
        rec  = recall_score(y_test, y_pred, average="weighted")

        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", 200)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"LogisticRegression C={C} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

print("\nAll runs completed! Run 'mlflow ui' to view results.")