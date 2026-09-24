from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from connectome import FlyWireConnectome, NT_COLUMNS


DATA_DIR = os.environ.get("FLYWIRE_DATA_DIR", "data/flywire_parts")

app = FastAPI(
    title="FlyGPT",
    description="Drosophila Connectome Chat Interface",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

_connectome: FlyWireConnectome | None = None
_connectome_lock = threading.Lock()


class ChatRequest(BaseModel):
    message: str


def get_connectome() -> FlyWireConnectome:
    global _connectome

    with _connectome_lock:
        if _connectome is None:
            _connectome = FlyWireConnectome(DATA_DIR)
        return _connectome


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


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
def health():
    data_path = Path(DATA_DIR)
    has_parquet = data_path.is_file() or (
        data_path.is_dir() and any(data_path.glob("*.parquet"))
    )

    return {
        "status": "ok",
        "data_dir": str(data_path),
        "data_available": has_parquet,
        "connectome_initialized": _connectome is not None,
    }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    msg = req.message.strip()

    if not msg:
        raise HTTPException(status_code=400, detail="메시지가 비어 있습니다.")

    lower = msg.lower()
    root_ids = extract_root_ids(msg)

    try:
        connectome = get_connectome()

        if "통계" in msg or "stats" in lower or "statistics" in lower:
            stats = connectome.stats()
            answer = (
                "📊 FlyWire v783 Dataset Statistics\n\n"
                f"Rows: {stats['rows']:,}\n"
                f"Presynaptic neurons ≈ {stats['approx_presynaptic_neurons']:,}\n"
                f"Postsynaptic neurons ≈ {stats['approx_postsynaptic_neurons']:,}\n"
                f"Neuropils: {stats['neuropils']:,}\n"
                f"Synapses: {stats['synapses']:,}\n"
                f"Parquet parts: {stats['parts']}"
            )
            return {"answer": answer, "type": "stats", "data": stats}

        if (
            "가장 강한 연결" in msg
            or "strongest" in lower
            or re.search(r"\btop\b", lower)
        ):
            limit = extract_limit(msg, 10)
            rows = connectome.top_connections(limit=limit)

            sections = [f"🔗 Top {len(rows)} Strongest Connections"]

            for index, row in enumerate(rows, 1):
                nt, probability = dominant_nt(row)
                sections.append(
                    f"{index}. {row['pre_pt_root_id']} → {row['post_pt_root_id']}\n"
                    f"   Synapses: {row['syn_count']:,}\n"
                    f"   Neuropils: {row['neuropil_count']}\n"
                    f"   Dominant NT: {nt} ({probability:.3f})"
                )

            return {
                "answer": "\n\n".join(sections),
                "type": "top_connections",
                "data": rows,
            }

        if len(root_ids) >= 2:
            pre_id, post_id = root_ids[:2]
            pair = connectome.pair(pre_id, post_id)

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

            return {"answer": answer, "type": "pair", "data": pair}

        if len(root_ids) == 1:
            root_id = root_ids[0]
            limit = extract_limit(msg, 5)

            if "입력" in msg or "input" in lower:
                direction = "inputs"
            elif "출력" in msg or "output" in lower:
                direction = "outputs"
            else:
                direction = "both"

            result = connectome.neuron(
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

            return {
                "answer": "\n\n".join(sections),
                "type": "neuron",
                "data": result,
            }

        return {
            "answer": (
                "❓ 이렇게 물어볼 수 있습니다:\n\n"
                "데이터 통계 보여줘\n"
                "가장 강한 연결 10개 보여줘\n"
                "뉴런 720575940000000000 보여줘\n"
                "뉴런 720575940000000000의 출력 연결 5개\n"
                "720575940000000000 -> 720575940111111111 연결"
            ),
            "type": "help",
            "data": None,
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Connectome query failed: {exc}",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
    )
