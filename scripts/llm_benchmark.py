import json
import requests


payload = {
    "model": "llama3.2:1b",
    "prompt": "請用三句話解釋什麼是 LLMOps。LLMOps 指的是 Large Language Model Operations。請使用繁體中文回答。",
    "repeat": 3,
}

response = requests.post(
    "http://localhost:8001/llm/benchmark",
    json=payload,
    timeout=180,
)

response.raise_for_status()

print(json.dumps(response.json(), ensure_ascii=False, indent=2))