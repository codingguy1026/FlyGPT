from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from connectome import MaleCNSConnectome
from dispatcher import dispatch, dispatch_math_fast_path, is_math_fast_path
from generator_runtime import GENERATIVE_ROUTES, GeneratorRuntime
from memory_store import MemoryStore, format_recall
from auth_store import AuthStore, SESSION_TTL_SECONDS


APP_VERSION = "0.8.0"

_model_override = os.environ.get("FLYGPT_MODEL_PATH")
if _model_override:
    MODEL_PATH = _model_override
else:
    _model_candidates = (
        "artifacts/fly_router_malecns_v0_4.pt",
        "artifacts/fly_router_v0_3_9.pt",
        "artifacts/fly_router_v0_3_8.pt",
        "artifacts/fly_router_v0_3_7.pt",
        "artifacts/fly_router_v0_3_6.pt",
        "artifacts/fly_router_v0_3_5.pt",
        "artifacts/fly_router_v0_3_4.pt",
        "artifacts/fly_router_v0_3_3.pt",
        "artifacts/fly_router_v0_3_2.pt",
        "artifacts/fly_router_v0_3_1.pt",
        "artifacts/fly_router_v0_3.pt",
        "artifacts/fly_router_v0_2.pt",
        "artifacts/fly_router_v0_1.pt",
    )
    MODEL_PATH = next(
        (path for path in _model_candidates if Path(path).is_file()),
        _model_candidates[0],
    )

app = FastAPI(
    title="FlyGPT",
    version=APP_VERSION,
    description="Drosophila Connectome Chat Interface with routing, generation, and local conversation memory",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

_connectome: MaleCNSConnectome | None = None
_connectome_lock = threading.Lock()

_router: Any | None = None
_router_lock = threading.Lock()
_router_error: str | None = None

_generator = GeneratorRuntime()
_memory = MemoryStore(os.environ.get("FLYGPT_MEMORY_PATH", "data/flygpt_memory.sqlite3"))
_auth = AuthStore(os.environ.get("FLYGPT_AUTH_PATH", "data/flygpt_auth.sqlite3"))

AUTH_COOKIE_NAME = "flygpt_session"
AUTH_COOKIE_SECURE = os.environ.get("FLYGPT_COOKIE_SECURE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class MemoryRequest(BaseModel):
    session_id: str


class AuthCredentials(BaseModel):
    email: str
    password: str
    display_name: str | None = None


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def _current_user(request: Request) -> dict[str, Any] | None:
    return _auth.user_for_token(request.cookies.get(AUTH_COOKIE_NAME))


def _require_user(request: Request) -> dict[str, Any]:
    user = _current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return user


def _scoped_session_id(user_id: str, session_id: str | None) -> str | None:
    raw = (session_id or "").strip()[:128]
    if not raw:
        return None
    return f"{user_id}:{raw}"


def get_connectome() -> MaleCNSConnectome:
    global _connectome

    with _connectome_lock:
        if _connectome is None:
            _connectome = MaleCNSConnectome()
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
    # MaleCNS body IDs are not fixed-width FlyWire root IDs; valid examples
    # include short values such as 12781.
    return [int(value) for value in re.findall(r"\b\d{4,16}\b", text)]


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


def dominant_nt(row: dict[str, Any]) -> str:
    return str(row.get("dominant_nt") or row.get("consensusNt") or "UNKNOWN").upper()


def format_partner(row: dict[str, Any], index: int) -> str:
    nt = dominant_nt(row)
    return (
        f"{index}. Partner: {row['partner_root_id']}\n"
        f"   Synapses: {row['syn_count']:,}\n"
        f"   Neuropils: {row['neuropil_count']}\n"
        f"   Consensus NT: {nt}"
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


@app.post("/api/auth/signup")
def auth_signup(req: AuthCredentials, response: Response):
    try:
        user = _auth.create_user(req.email, req.password, req.display_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token = _auth.create_session(user["id"])
    _set_auth_cookie(response, token)
    return {"user": user}


@app.post("/api/auth/login")
def auth_login(req: AuthCredentials, response: Response):
    user = _auth.authenticate(req.email, req.password)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="이메일 또는 비밀번호가 올바르지 않습니다.",
        )

    token = _auth.create_session(user["id"])
    _set_auth_cookie(response, token)
    return {"user": user}


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response):
    _auth.revoke_session(request.cookies.get(AUTH_COOKIE_NAME))
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        path="/",
        secure=AUTH_COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )
    return {"ok": True}


@app.get("/api/auth/me")
def auth_me(request: Request):
    return {"user": _require_user(request)}


@app.get("/api/router/graph")
def router_graph(request: Request):
    _require_user(request)
    router = get_router()
    if router is None:
        raise HTTPException(
            status_code=503,
            detail=_router_error or f"Fly router model not found: {MODEL_PATH}",
        )
    return router.graph()


@app.get("/api/health")
def health():
    model_path = Path(MODEL_PATH)
    neuprint_dataset = os.environ.get("NEUPRINT_DATASET", "male-cns:v1.0")
    neuprint_configured = bool(
        os.environ.get("NEUPRINT_TOKEN")
        or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    )

    return {
        "status": "ok",
        "connectome": {
            "provider": "neuPrint",
            "dataset": neuprint_dataset,
            "configured": neuprint_configured,
            "initialized": _connectome is not None,
        },
        "model_path": str(model_path),
        "model_available": model_path.is_file(),
        "router_initialized": _router is not None,
        "router_error": _router_error,
        "app_version": APP_VERSION,
        "dispatcher_enabled": True,
        "generator": _generator.status(),
        "memory": {
            "enabled": True,
            "path": str(_memory.path),
            "max_messages_per_session": 200,
        },
        "auth": {
            "enabled": True,
            "cookie_secure": AUTH_COOKIE_SECURE,
        },
    }


@app.get("/api/generator/status")
def generator_status():
    return _generator.status()


@app.get("/api/memory/status")
def memory_status(session_id: str, request: Request):
    user = _require_user(request)
    scoped = _scoped_session_id(user["id"], session_id)
    return _memory.stats(scoped or "")


@app.post("/api/memory/clear")
def memory_clear(req: MemoryRequest, request: Request):
    user = _require_user(request)
    session_id = _scoped_session_id(user["id"], req.session_id)
    return {
        "cleared": _memory.clear(session_id or ""),
        "session_id_present": bool(session_id),
    }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest, request: Request):
    request_started = time.perf_counter()
    user = _require_user(request)
    msg = req.message.strip()

    if not msg:
        raise HTTPException(status_code=400, detail="메시지가 비어 있습니다.")

    session_id = _scoped_session_id(user["id"], req.session_id)
    memory_context = _memory.recent(session_id, limit=12) if session_id else []
    if session_id:
        _memory.add(session_id, "user", msg)

    lower = msg.lower()
    root_ids = extract_root_ids(msg)
    router_started = time.perf_counter()
    route = predict_route(msg)
    router_ms = round((time.perf_counter() - router_started) * 1000)

    try:
        if is_math_fast_path(msg):
            result = dispatch_math_fast_path(msg)
            answer = result.answer
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
                "📊 Janelia MaleCNS v1.0 Dataset Statistics\n\n"
                f"Neurons: {stats['neurons']:,}\n"
                f"Presynaptic sites: {stats['presynaptic_sites']:,}\n"
                f"Postsynaptic sites: {stats['postsynaptic_sites']:,}\n"
                f"Primary neuropils: {stats['neuropils']:,}\n"
                f"Dataset: {stats['dataset']}"
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
                nt = dominant_nt(row)
                neuropils = row.get("neuropil_count")
                neuropil_text = str(neuropils) if neuropils is not None else "n/a (total edge)"
                sections.append(
                    f"{index}. {row['pre_pt_root_id']} → {row['post_pt_root_id']}\n"
                    f"   Synapses: {row['syn_count']:,}\n"
                    f"   Neuropils: {neuropil_text}\n"
                    f"   Consensus NT: {nt}"
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
                    nt = dominant_nt(row)
                    lines.append(
                        f"• {row['neuropil']}: {row['syn_count']:,} synapses, "
                        f"consensus NT {nt}"
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
            dispatch_started = time.perf_counter()
            result = dispatch(msg, route)
            dispatch_ms = round((time.perf_counter() - dispatch_started) * 1000)
            model_label = route.get("model", Path(MODEL_PATH).name)

            if result.status == "uncertain":
                return response_with_router(
                    answer=result.answer,
                    response_type="dispatch",
                    data={
                        "dispatch": result.to_dict(),
                        "generation": None,
                        "memory_hits": None,
                        "ui_meta": {
                            "mode": result.handler,
                            "router_model": model_label,
                            "app_version": APP_VERSION,
                        },
                        "timings": {
                            "router_ms": router_ms,
                            "dispatch_ms": dispatch_ms,
                            "generation_ms": None,
                            "total_ms": round((time.perf_counter() - request_started) * 1000),
                        },
                    },
                    route=route,
                    session_id=session_id,
                )

            memory_hits = None
            tool_context = result.tool_context

            if result.route == "memory":
                memory_hits = (
                    _memory.search(
                        session_id,
                        msg,
                        limit=5,
                        exclude_content=msg,
                    )
                    if session_id
                    else []
                )
                if memory_hits:
                    lines = ["Retrieved prior-session messages:"]
                    for item in memory_hits:
                        role = str(item.get("role", "user"))
                        content = str(item.get("content", "")).replace("\n", " ").strip()
                        lines.append(f"- {role}: {content[:700]}")
                    tool_context = "\n".join(lines)
                else:
                    tool_context = "No relevant prior-session messages were retrieved."

            elif result.route == "research":
                # Generator integration establishes the retrieval contract without pretending
                # that a live search backend already exists.
                tool_context = (
                    "No live search backend is connected in this build. "
                    "Do not invent current search results, prices, releases, news, "
                    "or other fresh external facts."
                )

            generation = _generator.generate(
                msg,
                result.route,
                memory_context=memory_context,
                tool_context=tool_context,
            )

            if generation.used and generation.answer:
                answer = generation.answer
                mode_label = f"{result.route} · generated"
            else:
                detail = generation.error or "generator is not configured"
                answer = (
                    "⚠️ 답변 생성기를 사용할 수 없습니다. "
                    "라우팅은 완료됐지만 생성 단계에서 중단됐어요. "
                    f"({detail})"
                )
                mode_label = f"{result.route} · generator_unavailable"

            total_ms = round((time.perf_counter() - request_started) * 1000)
            generation_ms = generation.latency_ms

            print(
                "[PERF] /api/chat "
                f"route={result.route} router={router_ms}ms "
                f"dispatch={dispatch_ms}ms generation={generation_ms}ms "
                f"total={total_ms}ms",
                flush=True,
            )

            data = {
                "dispatch": result.to_dict(),
                "generation": generation.to_dict(),
                "memory_hits": memory_hits,
                "timings": {
                    "router_ms": router_ms,
                    "dispatch_ms": dispatch_ms,
                    "generation_ms": generation_ms,
                    "total_ms": total_ms,
                },
                "ui_meta": {
                    "mode": mode_label,
                    "router_model": model_label,
                    "app_version": APP_VERSION,
                },
            }

            return response_with_router(
                answer=answer,
                response_type="dispatch",
                data=data,
                route=route,
                session_id=session_id,
            )

        if Path(MODEL_PATH).is_file() and _router_error:
            heading = "⚠️ FlyGPT 라우터 모델을 불러오지 못했습니다."
            detail = f"Router error: {_router_error}"
        else:
            heading = "❓ FlyGPT 라우터 모델을 찾지 못했습니다."
            detail = f"Expected model: {MODEL_PATH}"

        answer = (
            f"{heading}\n\n"
            f"{detail}\n\n"
            "커넥톰 질의는 계속 사용할 수 있습니다:\n"
            "데이터 통계 보여줘\n"
            "가장 강한 연결 10개 보여줘\n"
            "뉴런 12781의 출력 연결 5개\n"
            "12781 -> 85165 연결"
        )
        return response_with_router(
            answer=answer,
            response_type="help",
            data=None,
            route=None,
            session_id=session_id,
        )

    except (FileNotFoundError, RuntimeError) as exc:
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
