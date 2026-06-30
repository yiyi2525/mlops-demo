import json
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
METADATA_PATH = os.getenv("METADATA_PATH", "models/metadata.json")


class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Iris features: sepal length, sepal width, petal length, petal width",
    )


class PredictResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: Dict[str, float]
    model_run_id: Optional[str] = None
    model_version: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        app.state.metadata = json.load(f)

    yield


app = FastAPI(
    title="MLOps Demo API",
    description="A minimal FastAPI + MLflow + Docker demo for model serving.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model"),
    }


@app.get("/metadata")
def metadata():
    return app.state.metadata


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        x = np.array(payload.features).reshape(1, -1)

        model = app.state.model
        metadata = app.state.metadata

        predicted_class = int(model.predict(x)[0])
        probabilities = model.predict_proba(x)[0]

        target_names = metadata.get("target_names", [])
        predicted_label = target_names[predicted_class]

        probability_dict = {
            target_names[i]: float(probabilities[i])
            for i in range(len(target_names))
        }

        return PredictResponse(
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            probabilities=probability_dict,
            model_run_id=metadata.get("run_id"),
            model_version=str(metadata.get("registered_model_version")),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))