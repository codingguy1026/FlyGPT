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
from brain_runtime import FlyBrainRuntime
from dispatcher import dispatch, dispatch_math_fast_path, is_math_fast_path
from generator_runtime import GENERATIVE_ROUTES, GeneratorRuntime
from memory_store import MemoryStore, format_recall
from route_learning import RouteLearningStore
from knowledge_store import KnowledgeStore
from research_runtime import BraveResearchRuntime, validated_fact_candidates
from auth_store import AuthStore, SESSION_TTL_SECONDS


APP_VERSION = "0.12.2"


def _env_seconds(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


CHAT_REQUEST_BUDGET_SECONDS = _env_seconds(
    "FLYGPT_CHAT_BUDGET",
    22.0,
    5.0,
    28.0,
)
CHAT_RESPONSE_RESERVE_SECONDS = 0.75

_model_override = os.environ.get("FLYGPT_MODEL_PATH")
if _model_override:
    MODEL_PATH = _model_override
else:
    _model_candidates = (
        "artifacts/fly_router_malecns_v0_5.pt",
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
_brain = FlyBrainRuntime()
_memory = MemoryStore(os.environ.get("FLYGPT_MEMORY_PATH", "data/flygpt_memory.sqlite3"))
_route_learning = RouteLearningStore(
    os.environ.get("FLYGPT_ROUTE_LEARNING_PATH", "data/flygpt_route_learning.sqlite3")
)
_knowledge = KnowledgeStore(
    os.environ.get("FLYGPT_KNOWLEDGE_PATH", "data/flygpt_knowledge.sqlite3")
)
_research = BraveResearchRuntime()
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


class RouterFeedbackRequest(BaseModel):
    message: str
    route: str


class KnowledgeRememberRequest(BaseModel):
    statement: str


class KnowledgeRejectRequest(BaseModel):
    item_id: int


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


def predict_route(text: str, user_id: str | None = None) -> dict[str, Any] | None:
    router_was_loaded = _router is not None
    lookup_started = time.perf_counter()
    router = get_router()
    router_lookup_ms = round((time.perf_counter() - lookup_started) * 1000)
    if router is None:
        return None

    predict_started = time.perf_counter()
    try:
        result = router.predict(text, top_k=len(router.routes))
    except Exception as exc:
        global _router_error
        _router_error = f"{type(exc).__name__}: {exc}"
        return None
    predict_ms = round((time.perf_counter() - predict_started) * 1000)

    personalization_ms = 0
    if user_id:
        personalize_started = time.perf_counter()
        try:
            result = _route_learning.personalize(user_id, text, result)
        except Exception:
            # Personalization is an optional layer. A corrupt/locked learning
            # database must never take the frozen router or chat endpoint down.
            result["personalization"] = {
                "applied": False,
                "examples_considered": 0,
                "neighbors_used": 0,
                "blend": 0.0,
                "nearest_similarity": 0.0,
                "error": "unavailable",
            }
        personalization_ms = round((time.perf_counter() - personalize_started) * 1000)

    result["runtime_timings"] = {
        "cold_start": not router_was_loaded,
        "router_lookup_ms": router_lookup_ms,
        "model_load_ms": router_lookup_ms if not router_was_loaded else 0,
        "predict_ms": predict_ms,
        "personalization_ms": personalization_ms,
    }

    return result


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


def is_strongest_neuron_query(text: str) -> bool:
    """Detect natural-language requests for the most strongly connected neuron."""

    lower = " ".join(text.lower().split())

    korean_neuron = "뉴런" in text or "신경세포" in text
    korean_connection = any(token in text for token in ("연결", "시냅스"))
    korean_strength = any(
        token in text
        for token in (
            "가장 강",
            "제일 강",
            "가장 많이 연결",
            "제일 많이 연결",
            "연결이 가장",
            "연결이 제일",
            "연결 수가 가장",
            "연결수가 가장",
        )
    )
    if korean_neuron and korean_connection and korean_strength:
        return True

    english_patterns = (
        r"\bstrongest\s+(?:connected\s+)?neurons?\b",
        r"\bmost\s+connected\s+neurons?\b",
        r"\bneurons?\s+with\s+the\s+strongest\s+connections?\b",
    )
    return any(re.search(pattern, lower) for pattern in english_patterns)


def is_top_connection_query(text: str) -> bool:
    """Detect pair-level strongest-connection requests without hijacking generic TOP prompts."""

    if is_strongest_neuron_query(text):
        return False

    lower = " ".join(text.lower().split())
    if any(
        token in text
        for token in (
            "가장 강한 연결",
            "제일 강한 연결",
            "강한 연결 순위",
            "연결 강도 순위",
        )
    ):
        return True

    english_patterns = (
        r"\bstrongest\s+(?:neural\s+)?connections?\b",
        r"\btop\s*(?:\d+)?\s+(?:neural\s+)?connections?\b",
    )
    return any(re.search(pattern, lower) for pattern in english_patterns)


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
    timings: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    payload = {
        "answer": answer,
        "type": response_type,
        "data": data,
    }
    if route is not None:
        payload["router"] = route
    if timings is not None:
        payload["timings"] = timings

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
        "chat_request_budget_seconds": CHAT_REQUEST_BUDGET_SECONDS,
        "dispatcher_enabled": True,
        "brain": {
            "enabled": True,
            "planner": "deterministic-connectome-semantic-state",
            "generator_role": "mouth-only-language-realization",
            "semantic_contract": "mouth_only_v1",
        },
        "generator": _generator.status(),
        "memory": {
            "enabled": True,
            "path": str(_memory.path),
            "max_messages_per_session": 200,
        },
        "adaptive_learning": {
            "enabled": True,
            "account_scoped": True,
            "stores_raw_prompts": False,
            "path": str(_route_learning.path),
        },
        "knowledge": {
            "enabled": True,
            "account_scoped": True,
            "provenance_aware": True,
            "path": str(_knowledge.path),
        },
        "research": _research.status(),
        "auth": {
            "enabled": True,
            "cookie_secure": AUTH_COOKIE_SECURE,
        },
    }


@app.get("/api/router/learning/status")
def router_learning_status(request: Request):
    user = _require_user(request)
    return _route_learning.stats(user["id"])


@app.post("/api/router/learning/clear")
def router_learning_clear(request: Request):
    user = _require_user(request)
    return {
        "cleared": _route_learning.clear(user["id"]),
        "status": _route_learning.stats(user["id"]),
    }


@app.post("/api/router/learning/feedback")
def router_learning_feedback(req: RouterFeedbackRequest, request: Request):
    user = _require_user(request)
    message = req.message.strip()
    route = req.route.strip()

    if not message:
        raise HTTPException(status_code=400, detail="학습할 메시지가 비어 있습니다.")

    router = get_router()
    valid_routes = set(router.routes) if router is not None else set(GENERATIVE_ROUTES)
    if route not in valid_routes:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 라우트입니다: {route}",
        )

    learned = _route_learning.feedback(user["id"], message, route)
    return {
        "learned": learned,
        "route": route,
        "status": _route_learning.stats(user["id"]),
    }


@app.get("/api/knowledge/status")
def knowledge_status(request: Request):
    user = _require_user(request)
    return _knowledge.stats(user["id"])


@app.get("/api/knowledge/items")
def knowledge_items(request: Request, limit: int = 50):
    user = _require_user(request)
    return {
        "items": _knowledge.list_items(user["id"], limit=limit),
        "status": _knowledge.stats(user["id"]),
    }


@app.post("/api/knowledge/remember")
def knowledge_remember(req: KnowledgeRememberRequest, request: Request):
    user = _require_user(request)
    items = _knowledge.learn_from_user_text(
        user["id"],
        "기억해 " + req.statement.strip(),
    )
    return {
        "learned": bool(items),
        "items": items,
        "status": _knowledge.stats(user["id"]),
    }


@app.post("/api/knowledge/reject")
def knowledge_reject(req: KnowledgeRejectRequest, request: Request):
    user = _require_user(request)
    return {
        "rejected": _knowledge.reject(user["id"], req.item_id),
        "status": _knowledge.stats(user["id"]),
    }


@app.post("/api/knowledge/clear")
def knowledge_clear(request: Request):
    user = _require_user(request)
    return {
        "cleared": _knowledge.clear(user["id"]),
        "status": _knowledge.stats(user["id"]),
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

    try:
        knowledge_context, knowledge_hits = _knowledge.context_for_query(
            user["id"],
            msg,
            limit=6,
        )
    except Exception:
        knowledge_context, knowledge_hits = None, []

    try:
        knowledge_learned = _knowledge.learn_from_user_text(user["id"], msg)
    except Exception:
        knowledge_learned = []

    if session_id:
        _memory.add(session_id, "user", msg)

    lower = msg.lower()
    root_ids = extract_root_ids(msg)

    try:
        if is_math_fast_path(msg):
            result = dispatch_math_fast_path(msg)
            answer = result.answer
            return response_with_router(
                answer=answer,
                response_type="math_fast_path",
                data=result.to_dict(),
                route=None,
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
                route=None,
                session_id=session_id,
            )

        if is_strongest_neuron_query(msg):
            limit = extract_limit(msg, 1)
            query_started = time.perf_counter()
            rows = get_connectome().top_neurons(limit=limit)
            query_ms = round((time.perf_counter() - query_started) * 1000)
            total_ms = round((time.perf_counter() - request_started) * 1000)

            sections = [
                "🧠 MaleCNS 시냅스 수 상위 뉴런",
                "기준: 뉴런의 presynaptic + postsynaptic site 수 (pre + post)",
            ]
            if not rows:
                sections.append("조건에 맞는 뉴런을 찾지 못했습니다.")

            for index, row in enumerate(rows, 1):
                identity = row.get("type") or row.get("instance") or "untyped"
                sections.append(
                    f"{index}. Neuron {row['body_id']} · {identity}\n"
                    f"   총 시냅스 사이트: {row['total_synapses']:,}\n"
                    f"   입력(post): {row['incoming_synapses']:,} · 출력(pre): {row['outgoing_synapses']:,}\n"
                    f"   Consensus NT: {row['dominant_nt']}"
                )

            sections.append(
                "※ 빠른 조회를 위해 MaleCNS Neuron의 pre + post 사이트 수를 사용합니다. "
                "이는 생물학적 중요도·활성도나 ConnectsTo weighted degree를 뜻하지 않습니다."
            )

            print(
                "[PERF] /api/chat fastpath=top_neurons "
                f"query={query_ms}ms total={total_ms}ms",
                flush=True,
            )

            return response_with_router(
                answer="\n\n".join(sections),
                response_type="top_neurons",
                data=rows,
                route=None,
                session_id=session_id,
                timings={
                    "query_ms": query_ms,
                    "total_ms": total_ms,
                },
            )

        if is_top_connection_query(msg):
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
                route=None,
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
                route=None,
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
                route=None,
                session_id=session_id,
            )

        router_started = time.perf_counter()
        route = predict_route(msg, user["id"])
        router_ms = round((time.perf_counter() - router_started) * 1000)

        if route is not None:
            dispatch_started = time.perf_counter()
            result = dispatch(msg, route)
            dispatch_ms = round((time.perf_counter() - dispatch_started) * 1000)
            model_label = route.get("model", Path(MODEL_PATH).name)
            brain_state = _brain.plan(
                route_info=route,
                dispatch=result.to_dict(),
                message=msg,
            )

            if result.status == "uncertain":
                return response_with_router(
                    answer=result.answer,
                    response_type="dispatch",
                    data={
                        "dispatch": result.to_dict(),
                        "brain_state": brain_state,
                        "generation": None,
                        "memory_hits": None,
                        "knowledge_hits": knowledge_hits,
                        "knowledge_learned": knowledge_learned,
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

            try:
                learned_from_user = _route_learning.observe_if_confident(
                    user["id"],
                    msg,
                    route,
                )
            except Exception:
                learned_from_user = False
            route["learning_observed"] = learned_from_user

            memory_hits = None
            tool_context = result.tool_context
            research_result = None
            research_data = None
            search_ms = None
            verification_ms = None
            web_verified_facts: list[dict[str, Any]] = []

            if brain_state["retrieval"] == "memory":
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

            elif brain_state["retrieval"] == "research":
                search_started = time.perf_counter()
                research_result = _research.search(msg)
                search_ms = round((time.perf_counter() - search_started) * 1000)
                research_data = research_result.to_dict()

                if research_result.used:
                    tool_context = research_result.context()
                else:
                    detail = research_result.error or "no relevant web evidence was found"
                    tool_context = (
                        "Live web retrieval was unavailable for this request. "
                        f"Reason: {detail}. "
                        "Do not fabricate current search results, citations, prices, "
                        "releases, news, or other fresh external facts."
                    )

            brain_state = _brain.finalize(
                brain_state,
                memory_hits=memory_hits,
                research_result=research_data,
                tool_context=tool_context,
                knowledge_hits=knowledge_hits,
            )

            elapsed_before_generation = time.perf_counter() - request_started
            remaining_generation_budget = max(
                0.0,
                CHAT_REQUEST_BUDGET_SECONDS
                - elapsed_before_generation
                - CHAT_RESPONSE_RESERVE_SECONDS,
            )

            generation = _generator.generate(
                msg,
                result.route,
                memory_context=memory_context,
                tool_context=tool_context,
                knowledge_context=knowledge_context,
                brain_state=brain_state,
                budget_seconds=remaining_generation_budget,
            )

            if generation.used and generation.answer:
                answer = generation.answer
                mode_label = f"{result.route} · generated"

                if result.route == "research" and research_result is not None and research_result.used:
                    verification_started = time.perf_counter()
                    try:
                        raw_facts = _generator.extract_supported_facts(
                            query=msg,
                            answer=answer,
                            evidence_context=research_result.context(),
                            allowed_urls=[source.url for source in research_result.sources],
                        )
                        candidates = validated_fact_candidates(
                            raw_facts,
                            research_result,
                            query=msg,
                        )

                        for candidate in candidates:
                            stored = _knowledge.add(
                                user["id"],
                                candidate["statement"],
                                kind="fact",
                                source_type="web",
                                source_ref=candidate["source_ref"],
                                confidence=candidate["confidence"],
                                verified=True,
                                expires_at=candidate["expires_at"],
                                subject=candidate.get("subject"),
                                predicate=candidate.get("predicate"),
                                value=candidate.get("value"),
                                slot_key=candidate.get("slot_key"),
                            )
                            if stored is not None and stored.get("status") == "verified":
                                web_verified_facts.append(stored)
                    except Exception:
                        web_verified_facts = []
                    verification_ms = round(
                        (time.perf_counter() - verification_started) * 1000
                    )

                    if web_verified_facts:
                        mode_label = f"{result.route} · web-grounded · learned"
                    else:
                        mode_label = f"{result.route} · web-grounded"
            else:
                detail = generation.error or "generator is not configured"
                detail_lower = detail.lower()
                timed_out = (
                    "time budget exceeded" in detail_lower
                    or "timed out" in detail_lower
                    or "timeout" in detail_lower
                )
                if timed_out:
                    answer = (
                        "⏱️ 답변 생성 시간이 제한을 초과했어요. "
                        "파피티 뇌의 라우팅은 완료됐지만 LLM 발화 단계가 늦어 "
                        "연결이 끊기기 전에 중단했습니다. 다시 시도해 주세요."
                    )
                    mode_label = f"{result.route} · generator_timeout"
                else:
                    answer = (
                        "⚠️ 답변 생성기를 사용할 수 없습니다. "
                        "라우팅은 완료됐지만 생성 단계에서 중단됐어요. "
                        f"({detail})"
                    )
                    mode_label = f"{result.route} · generator_unavailable"

            total_ms = round((time.perf_counter() - request_started) * 1000)
            generation_ms = generation.latency_ms

            router_runtime = route.get("runtime_timings") or {}
            print(
                "[PERF] /api/chat "
                f"route={result.route} router={router_ms}ms "
                f"router_load={router_runtime.get('model_load_ms')}ms "
                f"router_predict={router_runtime.get('predict_ms')}ms "
                f"personalize={router_runtime.get('personalization_ms')}ms "
                f"cold_start={router_runtime.get('cold_start')} "
                f"dispatch={dispatch_ms}ms search={search_ms}ms "
                f"generation={generation_ms}ms "
                f"gen_open_wait={generation.open_wait_ms}ms "
                f"gen_body_read={generation.body_read_ms}ms "
                f"gen_json={generation.json_parse_ms}ms "
                f"req_bytes={generation.request_bytes} "
                f"resp_bytes={generation.response_bytes} "
                f"gen_budget={round(remaining_generation_budget * 1000)}ms "
                f"request_budget={round(CHAT_REQUEST_BUDGET_SECONDS * 1000)}ms "
                f"verify={verification_ms}ms total={total_ms}ms",
                flush=True,
            )

            data = {
                "dispatch": result.to_dict(),
                "brain_state": brain_state,
                "generation": generation.to_dict(),
                "memory_hits": memory_hits,
                "knowledge_hits": knowledge_hits,
                "knowledge_learned": knowledge_learned,
                "research": research_data,
                "web_verified_facts": web_verified_facts,
                "timings": {
                    "router_ms": router_ms,
                    "dispatch_ms": dispatch_ms,
                    "search_ms": search_ms,
                    "generation_ms": generation_ms,
                    "generation_open_wait_ms": generation.open_wait_ms,
                    "generation_body_read_ms": generation.body_read_ms,
                    "generation_json_parse_ms": generation.json_parse_ms,
                    "generation_request_bytes": generation.request_bytes,
                    "generation_response_bytes": generation.response_bytes,
                    "generation_budget_ms": round(remaining_generation_budget * 1000),
                    "request_budget_ms": round(CHAT_REQUEST_BUDGET_SECONDS * 1000),
                    "router_load_ms": router_runtime.get("model_load_ms"),
                    "router_predict_ms": router_runtime.get("predict_ms"),
                    "router_personalization_ms": router_runtime.get("personalization_ms"),
                    "router_cold_start": router_runtime.get("cold_start"),
                    "verification_ms": verification_ms,
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
