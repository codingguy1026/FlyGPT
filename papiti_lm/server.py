"""Loopback-only experimental OpenAI-compatible chat completion endpoint.

Start only via: python -m papiti_lm serve
Do not expose this toy language model on the public internet.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from papiti_lm import MODEL_NAME

app = FastAPI(title="Papiti scratch language model (experimental)")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1, max_length=32)
    max_tokens: int = Field(default=96, ge=1, le=8192)
    temperature: float = Field(default=0.8, ge=0, le=2)
    stream: bool = False


@lru_cache(maxsize=1)
def _loaded():
    from papiti_lm.model import load_checkpoint
    path = os.environ.get("PAPITI_LM_CHECKPOINT")
    if not path:
        raise RuntimeError("PAPITI_LM_CHECKPOINT is not configured")
    model, tokenizer, _metadata = load_checkpoint(path)
    return model, tokenizer


@app.get("/health")
def health():
    return {"status": "experimental", "model": MODEL_NAME}


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    if request.model != MODEL_NAME:
        raise HTTPException(status_code=400, detail="unknown model")
    if request.stream:
        raise HTTPException(status_code=400, detail="streaming is not supported")
    users = [msg.content for msg in request.messages if msg.role == "user"]
    if not users or not users[-1].strip() or len(users[-1]) > 2048:
        raise HTTPException(status_code=400, detail="invalid last user message")
    try:
        from papiti_lm.model import generate
        model, tokenizer = _loaded()
        # This is still single-turn generation: system/history are ignored.
        text = None
        if os.environ.get("PAPITI_LM_ARITHMETIC") == "1":
            from papiti_lm.arithmetic import try_arithmetic
            text = try_arithmetic(users[-1])
        if text is None:
            text = generate(
                model, tokenizer, users[-1], max_new_tokens=min(request.max_tokens, 128),
                temperature=request.temperature, top_k=min(24, tokenizer.vocab_size)
            )
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="model inference unavailable") from exc
    if not text.strip():
        raise HTTPException(status_code=503, detail="model produced no visible text")
    return {
        "id": "papiti-local-experiment",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }
