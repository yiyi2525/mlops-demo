import json
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

EXPERIMENT_NAME = "iris-classifier-demo"
REGISTERED_MODEL_NAME = "iris-random-forest"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 這裡使用 SQLite，方便本機 demo。
    # mlflow.db 會記錄 experiment、run、metrics、model metadata。
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    params = {
        "n_estimators": 100,
        "max_depth": 3,
        "random_state": 42,
    }

    model = RandomForestClassifier(**params)

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        f1_macro = float(f1_score(y_test, y_pred, average="macro"))

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train[:2]

        # 記錄模型到 MLflow，並註冊成一個 model version
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # 設定最新版本為 champion，面試可以拿這個講 model version / rollback
        client = MlflowClient()
        versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'"
        )

        latest_version = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            "champion",
            latest_version.version,
        )

        # FastAPI 服務實際載入這個檔案，讓 Docker demo 簡單穩定
        joblib.dump(model, MODEL_PATH)

        metadata = {
            "run_id": run.info.run_id,
            "experiment_name": EXPERIMENT_NAME,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "registered_model_version": latest_version.version,
            "model_alias": "champion",
            "model_type": "RandomForestClassifier",
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "features": iris.feature_names,
            "target_names": iris.target_names.tolist(),
        }

        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print("Training completed.")
        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 macro: {f1_macro:.4f}")
        print(f"Saved model to: {MODEL_PATH}")
        print(f"Registered model version: {latest_version.version}")
        print("Alias 'champion' has been assigned to the latest version.")


if __name__ == "__main__":
    main()