import time
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:1b"


app = FastAPI(
    title="Mini LLMOps Demo",
    description="Local LLM inference service using Ollama and FastAPI",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = DEFAULT_MODEL


class GenerateResponse(BaseModel):
    model: str
    prompt: str
    response: str
    latency_seconds: float
    eval_count: Optional[int] = None
    tokens_per_second: Optional[float] = None


class BenchmarkRequest(BaseModel):
    prompt: str = "請用三句話解釋什麼是 MLOps"
    model: Optional[str] = DEFAULT_MODEL
    repeat: int = 3


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "mini-llmops-demo",
        "model": DEFAULT_MODEL,
    }


@app.post("/llm/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    start_time = time.time()

    payload = {
        "model": request.model,
        "prompt": request.prompt,
        "stream": False,
    }

    try:
        ollama_response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )
        ollama_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to call Ollama API: {str(e)}",
        )

    latency = time.time() - start_time
    result = ollama_response.json()

    eval_count = result.get("eval_count")
    eval_duration = result.get("eval_duration")

    tokens_per_second = None
    if eval_count and eval_duration:
        # Ollama 的 eval_duration 單位是 nanoseconds
        eval_seconds = eval_duration / 1_000_000_000
        if eval_seconds > 0:
            tokens_per_second = round(eval_count / eval_seconds, 2)

    return GenerateResponse(
        model=request.model,
        prompt=request.prompt,
        response=result.get("response", ""),
        latency_seconds=round(latency, 4),
        eval_count=eval_count,
        tokens_per_second=tokens_per_second,
    )


@app.post("/llm/benchmark")
def benchmark(request: BenchmarkRequest):
    if request.repeat <= 0:
        raise HTTPException(
            status_code=400,
            detail="repeat must be greater than 0",
        )

    results = []
    latencies = []
    tokens_per_second_list = []

    for i in range(request.repeat):
        start_time = time.time()

        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
        }

        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=500,
                detail=f"Benchmark failed when calling Ollama API: {str(e)}",
            )

        latency = time.time() - start_time
        result = response.json()

        eval_count = result.get("eval_count")
        eval_duration = result.get("eval_duration")

        tokens_per_second = None
        if eval_count and eval_duration:
            eval_seconds = eval_duration / 1_000_000_000
            if eval_seconds > 0:
                tokens_per_second = round(eval_count / eval_seconds, 2)

        latencies.append(latency)

        if tokens_per_second is not None:
            tokens_per_second_list.append(tokens_per_second)

        results.append({
            "round": i + 1,
            "latency_seconds": round(latency, 4),
            "eval_count": eval_count,
            "tokens_per_second": tokens_per_second,
        })

    avg_latency = sum(latencies) / len(latencies)

    avg_tokens_per_second = None
    if tokens_per_second_list:
        avg_tokens_per_second = round(
            sum(tokens_per_second_list) / len(tokens_per_second_list),
            2,
        )

    return {
        "model": request.model,
        "repeat": request.repeat,
        "results": results,
        "avg_latency_seconds": round(avg_latency, 4),
        "min_latency_seconds": round(min(latencies), 4),
        "max_latency_seconds": round(max(latencies), 4),
        "avg_tokens_per_second": avg_tokens_per_second,
    }