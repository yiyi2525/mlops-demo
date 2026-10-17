import hashlib
import json
import os

from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import joblib
import numpy as np
import psycopg
from psycopg.rows import dict_row
import redis

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from psycopg.types.json import Jsonb
from redis.exceptions import RedisError


MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
METADATA_PATH = os.getenv("METADATA_PATH", "models/metadata.json")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "mlops_demo")
DB_USER = os.getenv("DB_USER", "mlops_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_ENABLED = os.getenv(
    "DB_ENABLED",
    "false",
).lower() == "true"


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
    cache_hit: bool = False


# -------------------------
# PostgreSQL helpers
# -------------------------
def get_db_connection():
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=2,
    )



def save_prediction_history(
    features: List[float],
    predicted_class: int,
    predicted_label: str,
    model_version: str,
):

    if not DB_ENABLED:
        return
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions (
                        features,
                        predicted_class,
                        predicted_label,
                        model_version
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        Jsonb(features),
                        predicted_class,
                        predicted_label,
                        model_version,
                    ),
                )

    except psycopg.Error as e:
        # History persistence is optional for /predict.
        # Prediction should still succeed if PostgreSQL is unavailable.
        print(f"Database write failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.model_ready = False
    app.state.startup_error = None

    try:
        # 1. Load model
        app.state.model = joblib.load(MODEL_PATH)

        # 2. Load metadata
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            app.state.metadata = json.load(f)

        # 3. Sanity check
        dummy_input = np.array([[5.1, 3.5, 1.4, 0.2]])

        prediction = app.state.model.predict(dummy_input)
        probabilities = app.state.model.predict_proba(dummy_input)

        if len(prediction) != 1:
            raise RuntimeError("Model prediction sanity check failed.")

        if probabilities.shape[0] != 1:
            raise RuntimeError("Model probability sanity check failed.")

        # 4. Model is ready
        app.state.model_ready = True

    except Exception as e:
        app.state.startup_error = str(e)

    # 5. Create Redis client
    # Redis is optional.
    # Do not ping here.
    app.state.redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1,
    )

    yield


app = FastAPI(
    title="MLOps Demo API",
    description="A FastAPI model serving demo with Redis caching and PostgreSQL history.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health/live")
def liveness():
    return {
        "status": "alive"
    }


@app.get("/health/ready")
def readiness():
    """
    Kubernetes readiness probe.

    Only checks critical dependencies required for the core /predict service.
    Redis and PostgreSQL are optional for /predict, so they must not block
    Kubernetes readiness.
    """

    # Model is the critical dependency for inference.
    if not getattr(app.state, "model_ready", False):
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "error": getattr(
                    app.state,
                    "startup_error",
                    None,
                ),
            },
        )

    return {
        "status": "ready",
        "model_loaded": True,
        "model_version": str(
            app.state.metadata.get(
                "registered_model_version"
            )
        ),
    }


@app.get("/health/dependencies")
def dependency_health():
    """
    Diagnostic endpoint for optional dependencies.

    This endpoint is intentionally separate from /health/ready because
    Redis and PostgreSQL are not required for the core inference path.
    """

    # Redis
    redis_available = False
    redis_client = getattr(
        app.state,
        "redis_client",
        None,
    )

    if redis_client is not None:
        try:
            redis_client.ping()
            redis_available = True
        except RedisError:
            redis_available = False

    # PostgreSQL
    postgres_available = False

    if DB_ENABLED:
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")

            postgres_available = True

        except psycopg.Error:
            postgres_available = False

    return {
        "redis_available": redis_available,
        "postgres_enabled": DB_ENABLED,
        "postgres_available": postgres_available,
    }


@app.get("/metadata")
def metadata():
    return app.state.metadata


@app.get("/version")
def version():
    return {"version": "v1.2"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        model = app.state.model
        metadata = app.state.metadata
        redis_client = app.state.redis_client

        model_version = str(
            metadata.get("registered_model_version")
        )

        # Cache key = model version + input features
        cache_source = json.dumps(
            {
                "model_version": model_version,
                "features": payload.features,
            },
            sort_keys=True,
        )

        cache_hash = hashlib.sha256(
            cache_source.encode("utf-8")
        ).hexdigest()

        cache_key = f"prediction:{cache_hash}"

        # -------------------------
        # 1. Try Redis
        # -------------------------
        try:
            cached_value = redis_client.get(cache_key)

            if cached_value is not None:
                cached_result = json.loads(cached_value)

                # Even a cache hit is still a real prediction request,
                # so store it in prediction history.
                save_prediction_history(
                    features=payload.features,
                    predicted_class=cached_result["predicted_class"],
                    predicted_label=cached_result["predicted_label"],
                    model_version=cached_result["model_version"],
                )

                return PredictResponse(
                    **cached_result,
                    cache_hit=True,
                )

        except RedisError:
            # Redis unavailable -> fallback to model
            pass

        # -------------------------
        # 2. Model inference
        # -------------------------
        x = np.array(payload.features).reshape(1, -1)

        predicted_class = int(model.predict(x)[0])
        probabilities = model.predict_proba(x)[0]

        target_names = metadata.get("target_names", [])
        predicted_label = target_names[predicted_class]

        probability_dict = {
            target_names[i]: float(probabilities[i])
            for i in range(len(target_names))
        }

        result = {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "probabilities": probability_dict,
            "model_run_id": metadata.get("run_id"),
            "model_version": model_version,
        }

        # -------------------------
        # 3. Save prediction history
        # -------------------------
        save_prediction_history(
            features=payload.features,
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            model_version=model_version,
        )

        # -------------------------
        # 4. Write Redis cache
        # -------------------------
        try:
            redis_client.set(
                cache_key,
                json.dumps(result),
                ex=CACHE_TTL_SECONDS,
            )

        except RedisError:
            # Redis unavailable -> prediction still succeeds
            pass

        return PredictResponse(
            **result,
            cache_hit=False,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

@app.get("/predictions")
def get_predictions():
    try:
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        features,
                        predicted_class,
                        predicted_label,
                        model_version,
                        created_at
                    FROM predictions
                    ORDER BY id
                    """
                )

                rows = cur.fetchall()

        return rows

    except psycopg.Error as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database unavailable: {e}",
        )


@app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: int):
    try:
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        features,
                        predicted_class,
                        predicted_label,
                        model_version,
                        created_at
                    FROM predictions
                    WHERE id = %s
                    """,
                    (prediction_id,),
                )

                row = cur.fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="Prediction not found",
            )

        return row

    except HTTPException:
        raise

    except psycopg.Error as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database unavailable: {e}",
        )


@app.delete("/predictions/{prediction_id}", status_code=204)
def delete_prediction(prediction_id: int):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM predictions
                    WHERE id = %s
                    """,
                    (prediction_id,),
                )

                deleted_rows = cur.rowcount

        if deleted_rows == 0:
            raise HTTPException(
                status_code=404,
                detail="Prediction not found",
            )

        return None

    except HTTPException:
        raise

    except psycopg.Error as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database unavailable: {e}",
        )