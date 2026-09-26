from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from connectome import FlyWireConnectome, NT_COLUMNS
from dispatcher import dispatch, dispatch_math_fast_path, is_math_fast_path
from generator_runtime import GENERATIVE_ROUTES, GeneratorRuntime
from memory_store import MemoryStore, format_recall


DATA_DIR = os.environ.get("FLYWIRE_DATA_DIR", "data/flywire_parts")

_model_override = os.environ.get("FLYGPT_MODEL_PATH")
if _model_override:
    MODEL_PATH = _model_override
else:
    _model_candidates = (
        "artifacts/fly_router_v0_2.pt",
        "artifacts/fly_router_v0_1.pt",
    )
    MODEL_PATH = next(
        (path for path in _model_candidates if Path(path).is_file()),
        _model_candidates[0],
    )

app = FastAPI(
    title="FlyGPT",
    version="0.7.0",
    description="Drosophila Connectome Chat Interface with live router streaming and compressed memory",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

_connectome: FlyWireConnectome | None = None
_connectome_lock = threading.Lock()

_router: Any | None = None
_router_lock = threading.Lock()
_router_error: str | None = None

_generator = GeneratorRuntime()
_memory = MemoryStore(os.environ.get("FLYGPT_MEMORY_PATH", "data/flygpt_memory.sqlite3"))


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class MemoryRequest(BaseModel):
    session_id: str


def get_connectome() -> FlyWireConnectome:
    global _connectome

    with _connectome_lock:
        if _connectome is None:
            _connectome = FlyWireConnectome(DATA_DIR)
        return _connectome


def get_router() -> Any | None:
    global _router, _router_error

    if _router is not None:
        return _router

    model_path = Path(MODEL_PATH)
    if not model_path.is_file():
        return None

    with _router_lock:
        if _router is not None:
            return _router

        try:
            from router_runtime import FlyRouterRuntime

            _router = FlyRouterRuntime(model_path)
            _router_error = None
        except Exception as exc:
            _router_error = f"{type(exc).__name__}: {exc}"
            return None

    return _router


def predict_route(text: str) -> dict[str, Any] | None:
    router = get_router()
    if router is None:
        return None

    try:
        return router.predict(text)
    except Exception as exc:
        global _router_error
        _router_error = f"{type(exc).__name__}: {exc}"
        return None


def extract_root_ids(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"\b\d{15,}\b", text)]


def extract_limit(text: str, default: int = 10) -> int:
    patterns = (
        r"(\d+)\s*개",
        r"(?:top|limit)\s*(\d+)",
        r"(\d+)\s*(?:connections?|partners?)",
    )

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return max(1, min(int(match.group(1)), 100))

    return default


def dominant_nt(row: dict[str, Any]) -> tuple[str, float]:
    column = max(
        NT_COLUMNS,
        key=lambda key: float(row.get(key) or 0.0),
    )
    return column.removesuffix("_avg").upper(), float(row.get(column) or 0.0)


def format_partner(row: dict[str, Any], index: int) -> str:
    nt, probability = dominant_nt(row)
    return (
        f"{index}. Partner: {row['partner_root_id']}\n"
        f"   Synapses: {row['syn_count']:,}\n"
        f"   Neuropils: {row['neuropil_count']}\n"
        f"   Dominant NT: {nt} ({probability:.3f})"
    )


def response_with_router(
    *,
    answer: str,
    response_type: str,
    data: Any,
    route: dict[str, Any] | None,
    session_id: str | None = None,
) -> dict[str, Any]:
    payload = {
        "answer": answer,
        "type": response_type,
        "data": data,
    }
    if route is not None:
        payload["router"] = route

    if session_id:
        _memory.add(session_id, "assistant", answer)

    return payload


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.get("/api/router/graph")
def router_graph():
    router = get_router()
    if router is None:
        raise HTTPException(
            status_code=503,
            detail=_router_error or f"Fly router model not found: {MODEL_PATH}",
        )
    return router.graph()


@app.get("/api/health")
def health():
    data_path = Path(DATA_DIR)
    model_path = Path(MODEL_PATH)
    has_parquet = data_path.is_file() or (
        data_path.is_dir() and any(data_path.glob("*.parquet"))
    )

    return {
        "status": "ok",
        "data_dir": str(data_path),
        "data_available": has_parquet,
        "connectome_initialized": _connectome is not None,
        "model_path": str(model_path),
        "model_available": model_path.is_file(),
        "router_initialized": _router is not None,
        "router_error": _router_error,
        "app_version": "0.7.0",
        "dispatcher_enabled": True,
        "streaming_enabled": True,
        "generator": _generator.status(),
        "memory": {
            "enabled": True,
            "path": str(_memory.path),
            "max_messages_per_session": 200,
            "capsule_batch_size": 8,
            "long_term_capsules": True,
        },
    }


@app.get("/api/memory/status")
def memory_status(session_id: str):
    return _memory.stats(session_id.strip()[:128])


@app.post("/api/memory/clear")
def memory_clear(req: MemoryRequest):
    session_id = req.session_id.strip()[:128]
    return {
        "cleared": _memory.clear(session_id),
        "session_id_present": bool(session_id),
    }


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="메시지가 비어 있습니다.")

    async def event_stream():
        try:
            route = predict_route(msg)

            if route is not None:
                route_event = {
                    "type": "route",
                    "router": {
                        key: value
                        for key, value in route.items()
                        if key != "trace"
                    },
                }
                yield json.dumps(route_event, ensure_ascii=False) + "\n"

                trace = route.get("trace") or []
                for index, frame in enumerate(trace):
                    yield json.dumps(
                        {
                            "type": "brain_step",
                            "index": index,
                            "total": len(trace),
                            "frame": frame,
                        },
                        ensure_ascii=False,
                    ) + "\n"
                    await asyncio.sleep(0.18)

            result = chat_endpoint(req)
            yield json.dumps(
                {
                    "type": "answer",
                    "payload": result,
                },
                ensure_ascii=False,
            ) + "\n"
            yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"

        except Exception as exc:
            yield json.dumps(
                {
                    "type": "error",
                    "detail": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    msg = req.message.strip()

    if not msg:
        raise HTTPException(status_code=400, detail="메시지가 비어 있습니다.")

    session_id = (req.session_id or "").strip()[:128] or None
    memory_context = _memory.recent(session_id, limit=12) if session_id else []
    if session_id:
        _memory.add(session_id, "user", msg)

    lower = msg.lower()
    root_ids = extract_root_ids(msg)
    route = predict_route(msg)

    try:
        if is_math_fast_path(msg):
            result = dispatch_math_fast_path(msg)
            router_note = (
                f"Router raw: {route['route']} · {route['confidence']:.1%}"
                if route is not None
                else "Router raw: unavailable"
            )
            answer = (
                "🪰 FlyGPT v0.7 · math fast-path\n\n"
                f"{result.answer}\n\n"
                "Decision: math fast-path\n"
                f"{router_note}"
            )
            return response_with_router(
                answer=answer,
                response_type="math_fast_path",
                data=result.to_dict(),
                route=route,
                session_id=session_id,
            )

        if "통계" in msg or "stats" in lower or "statistics" in lower:
            stats = get_connectome().stats()
            answer = (
                "📊 FlyWire v783 Dataset Statistics\n\n"
                f"Rows: {stats['rows']:,}\n"
                f"Presynaptic neurons ≈ {stats['approx_presynaptic_neurons']:,}\n"
                f"Postsynaptic neurons ≈ {stats['approx_postsynaptic_neurons']:,}\n"
                f"Neuropils: {stats['neuropils']:,}\n"
                f"Synapses: {stats['synapses']:,}\n"
                f"Parquet parts: {stats['parts']}"
            )
            return response_with_router(
                answer=answer,
                response_type="stats",
                data=stats,
                route=route,
                session_id=session_id,
            )

        if (
            "가장 강한 연결" in msg
            or "strongest" in lower
            or re.search(r"\btop\b", lower)
        ):
            limit = extract_limit(msg, 10)
            rows = get_connectome().top_connections(limit=limit)

            sections = [f"🔗 Top {len(rows)} Strongest Connections"]

            for index, row in enumerate(rows, 1):
                nt, probability = dominant_nt(row)
                sections.append(
                    f"{index}. {row['pre_pt_root_id']} → {row['post_pt_root_id']}\n"
                    f"   Synapses: {row['syn_count']:,}\n"
                    f"   Neuropils: {row['neuropil_count']}\n"
                    f"   Dominant NT: {nt} ({probability:.3f})"
                )

            return response_with_router(
                answer="\n\n".join(sections),
                response_type="top_connections",
                data=rows,
                route=route,
                session_id=session_id,
            )

        if len(root_ids) >= 2:
            pre_id, post_id = root_ids[:2]
            pair = get_connectome().pair(pre_id, post_id)

            if pair["syn_count"] == 0:
                answer = f"⚠️ {pre_id} → {post_id} 연결을 찾지 못했습니다."
            else:
                lines = [
                    "🔍 Neural Pair",
                    "",
                    f"{pre_id} → {post_id}",
                    f"Total synapses: {pair['syn_count']:,}",
                    "",
                    "By neuropil:",
                ]

                for row in pair["by_neuropil"]:
                    nt, probability = dominant_nt(row)
                    lines.append(
                        f"• {row['neuropil']}: {row['syn_count']:,} synapses, "
                        f"{nt} {probability:.3f}"
                    )

                answer = "\n".join(lines)

            return response_with_router(
                answer=answer,
                response_type="pair",
                data=pair,
                route=route,
                session_id=session_id,
            )

        if len(root_ids) == 1:
            root_id = root_ids[0]
            limit = extract_limit(msg, 5)

            if "입력" in msg or "input" in lower:
                direction = "inputs"
            elif "출력" in msg or "output" in lower:
                direction = "outputs"
            else:
                direction = "both"

            result = get_connectome().neuron(
                root_id,
                direction=direction,
                limit=limit,
            )

            sections = [f"🧠 Neuron {root_id}"]

            if "outputs" in result:
                output_lines = ["Outputs:"]
                if result["outputs"]:
                    output_lines.extend(
                        format_partner(row, index)
                        for index, row in enumerate(result["outputs"], 1)
                    )
                else:
                    output_lines.append("No outputs found.")
                sections.append("\n".join(output_lines))

            if "inputs" in result:
                input_lines = ["Inputs:"]
                if result["inputs"]:
                    input_lines.extend(
                        format_partner(row, index)
                        for index, row in enumerate(result["inputs"], 1)
                    )
                else:
                    input_lines.append("No inputs found.")
                sections.append("\n".join(input_lines))

            return response_with_router(
                answer="\n\n".join(sections),
                response_type="neuron",
                data=result,
                route=route,
                session_id=session_id,
            )

        if route is not None:
            result = dispatch(msg, route)
            model_label = route.get("model", Path(MODEL_PATH).name)

            generation = None
            memory_hits = None
            final_answer = result.answer
            mode_label = result.handler

            if result.route == "memory" and result.status != "uncertain" and session_id:
                memory_hits = _memory.search(
                    session_id,
                    msg,
                    limit=5,
                    exclude_content=msg,
                )
                final_answer = format_recall(memory_hits)
                mode_label = "memory · recall"
            elif (
                result.status == "completed"
                and result.route in GENERATIVE_ROUTES
            ):
                generation = _generator.generate(
                    msg,
                    result.route,
                    memory_context=memory_context,
                )
                if generation.used and generation.answer:
                    final_answer = generation.answer
                    mode_label = f"{result.route} · generated"
                else:
                    mode_label = f"{result.handler} · fallback"

            answer = (
                f"🪰 FlyGPT v0.7 · {mode_label}\n\n"
                f"{final_answer}\n\n"
                f"Route: {route['route']} · {route['confidence']:.1%}\n"
                f"Router: {model_label}"
            )

            data = {
                "dispatch": result.to_dict(),
                "generation": generation.to_dict() if generation is not None else None,
                "memory_hits": memory_hits,
            }

            return response_with_router(
                answer=answer,
                response_type="dispatch",
                data=data,
                route=route,
                session_id=session_id,
            )

        answer = (
            "❓ FlyGPT 라우터 모델을 찾지 못했습니다.\n\n"
            f"Expected model: {MODEL_PATH}\n\n"
            "커넥톰 질의는 계속 사용할 수 있습니다:\n"
            "데이터 통계 보여줘\n"
            "가장 강한 연결 10개 보여줘\n"
            "뉴런 720575940627737365의 출력 연결 5개\n"
            "720575940627737365 -> 720575940628914436 연결"
        )
        return response_with_router(
            answer=answer,
            response_type="help",
            data=None,
            route=None,
            session_id=session_id,
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"FlyGPT request failed: {exc}",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
    )
