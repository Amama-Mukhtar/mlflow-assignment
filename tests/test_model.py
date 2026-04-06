import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def get_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_random_forest_accuracy():
    X_train, X_test, y_train, y_test = get_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc >= 0.9, f"Accuracy too low: {acc}"


def test_logistic_regression_accuracy():
    X_train, X_test, y_train, y_test = get_data()
    model = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc >= 0.9, f"Accuracy too low: {acc}"


def test_data_shape():
    X_train, X_test, y_train, y_test = get_data()
    assert X_train.shape[1] == 4, "Expected 4 features"
    assert len(X_train) > 0, "Training set is empty"
    assert len(X_test) > 0, "Test set is empty"


def test_model_predicts_valid_classes():
    X_train, X_test, y_train, y_test = get_data()
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert set(predictions).issubset({0, 1, 2}), "Invalid class predicted"