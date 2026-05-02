import json
import requests
from config import OLLAMA_MODEL, OLLAMA_URL


def ask_ollama_json(messages, schema):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "format": schema
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()

    content = response.json()["message"]["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "error": "Invalid JSON returned by model",
            "raw_response": content
        }


def ask_ollama_text(messages):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()

    return response.json()["message"]["content"]