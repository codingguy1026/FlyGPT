"use client";

import { useEffect, useMemo, useState } from "react";
import { getRouterGraph, getRouterAnnotations, type GraphResponse, type RouterResult, type NeuronAnnotationsResponse, type NeuronKind } from "@/lib/api";

const KIND_COLORS: Record<NeuronKind, string> = {
  sensory: "#65d6fa",
  motor: "#8caaff",
  interneuron: "#e6c874",
  projection: "#b58bff",
  other: "#f3a777",
  unclassified: "#8aa49a",
};

type Props = {
  router: RouterResult | null;
  pending: boolean;
  live: boolean;
  phase: string;
  onExpand: () => void;
};

// Keep this schematic layout aligned with the full BrainPanel. These coordinates
// are display coordinates, not anatomical coordinates or measured calcium data.
function nodePosition(index: number, count: number) {
  const right = index % 2 === 1;
  const rank = Math.floor(index / 2);
  const perSide = Math.max(1, Math.ceil(count / 2));
  const progress = (rank + 0.5) / perSide;
  const angle = rank * (Math.PI * (3 - Math.sqrt(5))) + (right ? 0.45 : -0.45);
  const radius = Math.sqrt(progress);
  return {
    x: (right ? 65.5 : 34.5) + Math.cos(angle) * 24.5 * radius,
    y: 50 + Math.sin(angle) * 42 * radius,
  };
}

export default function ConnectomePreview({ router, pending, live, phase, onExpand }: Props) {
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [annotations, setAnnotations] = useState<NeuronAnnotationsResponse | null>(null);
  const [error, setError] = useState(false);
  const [retry, setRetry] = useState(0);
  const [frameIndex, setFrameIndex] = useState(-1);

  useEffect(() => {
    let active = true;
    setError(false);
    getRouterGraph()
      .then((result) => { if (active) setGraph(result); })
      .catch(() => { if (active) setError(true); });
    return () => { active = false; };
  }, [retry]);

  useEffect(() => {
    if (!graph?.nodes.length) return;
    let active = true;
    // Official MaleCNS superclass annotations are optional. If neuPrint
    // is unavailable, use the neutral/unclassified color instead.
    getRouterAnnotations()
      .then((response) => { if (active) setAnnotations(response); })
      .catch(() => { if (active) setAnnotations(null); });
    return () => { active = false; };
  }, [graph]);

  useEffect(() => {
    // While streaming, show the most recently received real frame immediately.
    // Only replay older frames after the request has completed.
    if (live) {
      setFrameIndex(-1);
      return;
    }
    const frames = router?.trace ?? [];
    if (frames.length === 0) {
      setFrameIndex(-1);
      return;
    }
    setFrameIndex(0);
    let index = 0;
    const timer = window.setInterval(() => {
      if (index >= frames.length - 1) {
        window.clearInterval(timer);
        return;
      }
      index += 1;
      setFrameIndex(index);
    }, 420);
    return () => window.clearInterval(timer);
  }, [router, live]);

  const positions = useMemo(
    () => graph?.nodes.map((_, index) => nodePosition(index, graph.nodes.length)) ?? [],
    [graph],
  );
  const frame = live
    ? router?.trace?.[Math.max(0, (router.trace?.length ?? 0) - 1)] ?? null
    : frameIndex < 0 ? null : router?.trace?.[frameIndex] ?? null;
  const compatible = !!graph && !!frame && frame.activations.length === graph.nodes.length;
  const activations = compatible && frame ? frame.activations : [];
  const hasTrace = !!router?.trace?.length;
  const status = phase === "tool" ? "EXACT TOOL · NO ROUTER"
    : phase === "error" ? "REQUEST ERROR"
    : pending && phase === "routing" && compatible ? "LIVE · ROUTER FRAME"
    : pending && phase === "generation" ? "ROUTER COMPLETE · GENERATING"
    : pending ? "REQUEST IN PROGRESS"
    : compatible ? "TRACE REPLAY"
    : hasTrace ? "TRACE UNAVAILABLE"
    : "AWAITING SIGNAL";

  if (router?.category_free) {
    return (
      <section className="connectomePreview" aria-label="MaleCNS v0.5 기능 선택 결과">
        <div className="connectomePreviewTop">
          <div><span>MALECNS v0.5</span><strong>166,700-neuron policy</strong></div>
          <button type="button" onClick={onExpand} aria-label="기능 선택 상세 화면 열기">DETAIL ↗</button>
        </div>
        <div className="connectomeGraphEmpty" role="status">
          {router.selected_capabilities?.length
            ? "선택된 기능: " + router.selected_capabilities.join(", ")
            : "외부 기능을 사용하지 않기로 판단했습니다."}
          <div>이 모델은 뉴런별 실제 발화 기록을 제공하지 않습니다.</div>
        </div>
      </section>
    );
  }

  return (
    <section className="connectomePreview" aria-label="MaleCNS 라우터 신경망 미리보기">
      <div className="connectomePreviewTop">
        <div>
          <span>ROUTER CONNECTOME</span>
          <strong>Neural activity</strong>
        </div>
        <button type="button" onClick={onExpand} aria-label="Brain 상세 화면 열기">
          EXPAND ↗
        </button>
      </div>
      <div className="connectomeGraph">
        {error ? (
          <div className="connectomeGraphEmpty" role="alert">
            <span>라우터 그래프를 불러오지 못했어요.</span>
            <button type="button" onClick={() => setRetry((value) => value + 1)}>
              다시 시도
            </button>
          </div>
        ) : !graph ? (
          <div className="connectomeGraphEmpty" role="status">MaleCNS 라우터 그래프 불러오는 중...</div>
        ) : (
          <svg viewBox="0 0 100 100" role="img" aria-label={compatible ? "실제 라우터 trace의 노드 활성도" : "MaleCNS 라우터 연결 구조, 활성도 대기 중"}>
            <path
              className="connectomeOutline"
              d="M50 12 C38 7 22 14 18 29 C10 36 10 51 17 59 C14 73 25 87 40 86 C44 92 56 92 60 86 C75 87 86 73 83 59 C90 51 90 36 82 29 C78 14 62 7 50 12 Z"
            />
            {graph.edges.map((edge, index) => {
              const src = positions[edge.src];
              const dst = positions[edge.dst];
              if (!src || !dst) return null;
              // Edge lines show structural connections, not measured signal flow.
              return (
                <line
                  key={index}
                  x1={src.x} y1={src.y} x2={dst.x} y2={dst.y}
                  className="connectomeEdge"
                />
              );
            })}
            {graph.nodes.map((node, index) => {
              const pos = positions[index];
              const activity = Math.max(0, Math.min(1, activations[index] ?? 0));
              const neuronKind = node.root_id ? annotations?.by_body_id[node.root_id]?.kind : undefined;
              const categoryColor = KIND_COLORS[neuronKind ?? "unclassified"];
              return (
                <circle
                  key={node.root_id ?? index}
                  cx={pos.x} cy={pos.y}
                  r={activity > 0.08 ? 0.85 + activity * 1.4 : 0.58}
                  className={activity > 0.08 ? "connectomeNode active" : "connectomeNode"}
                  style={{
                    opacity: activity > 0.08 ? 0.55 + activity * 0.45 : 0.36,
                    fill: activity > 0.08 ? categoryColor : "#6c9a91",
                    filter: activity > 0.08 ? `drop-shadow(0 0 2px ${categoryColor})` : "none",
                  }}
                >
                  <title>{`Body ${node.root_id ?? index} · ${neuronKind ?? "unclassified"} · model trace activation ${(activity * 100).toFixed(1)}%`}</title>
                </circle>
              );
            })}
          </svg>
        )}
      </div>
      <div className="connectomeKindLegend" aria-label="공식 MaleCNS 분류별 색상">
         {annotations && ([
           ["sensory", "감각"], ["motor", "운동"], ["interneuron", "중간"],
           ["projection", "투사"], ["other", "기타"], ["unclassified", "미분류"],
         ] as const).map(([kind, label]) => (
           <span key={kind}><i style={{ backgroundColor: KIND_COLORS[kind] }} />{label}</span>
         ))}
       </div>
       <div className="connectomeTraceStatus" aria-live="polite">
        <span className={compatible && (phase === "routing" || !pending) ? "lit" : ""} />
        <div>
          <strong>{status}</strong>
          <small>
            {phase === "tool"
              ? "확정 도구 처리 · 라우터 실행 없음"
              : pending && compatible
                ? `실제 라우터 프레임 · ${frame?.stage?.replaceAll("_", " ") ?? "step"}`
                : pending
                  ? "실제 라우터 이벤트를 기다리는 중"
                  : compatible
                    ? `기록 재생 · ${frame?.stage?.replaceAll("_", " ") ?? "step"}`
                    : "질문을 보내면 기록된 활성도가 표시됩니다"}
          </small>
        </div>
      </div>
      <div className="connectomePreviewFooter">
        <span>{graph ? `${graph.n_nodes.toLocaleString()} NODES · ${graph.n_edges.toLocaleString()} EDGES` : "ROUTER SCAFFOLD"}</span>
        <span>SCHEMATIC · NOT GCaMP</span>
      </div>
    </section>
  );
}
