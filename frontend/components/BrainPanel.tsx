"use client";

import { useEffect, useMemo, useState } from "react";
import type { GraphResponse, RouterResult, TraceFrame } from "@/lib/api";
import { getRouterGraph } from "@/lib/api";
import RoutePanel from "./RoutePanel";

type MonitorPhase = "idle" | "processing" | "router_loading" | "routing" | "generation" | "tool" | "done" | "error";
type Props = {
  open: boolean;
  onClose: () => void;
  router: RouterResult | null;
  phase: MonitorPhase;
  requestId: number;
};

// Schematic layout, NOT anatomical coordinates.
function nodePosition(index: number, count: number) {
  const right = index % 2 === 1;
  const rank = Math.floor(index / 2);
  const perSide = Math.max(1, Math.ceil(count / 2));
  const angle = rank * (Math.PI * (3 - Math.sqrt(5))) + (right ? 0.45 : -0.45);
  const radius = Math.sqrt((rank + 0.5) / perSide);
  return {
    x: (right ? 65.5 : 34.5) + Math.cos(angle) * 24.5 * radius,
    y: 50 + Math.sin(angle) * 42 * radius,
  };
}

export default function BrainPanel({ open, onClose, router, phase, requestId }: Props) {
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [error, setError] = useState("");
  const [retryCount, setRetryCount] = useState(0);
  const [replayIndex, setReplayIndex] = useState<number | null>(null);
  const [selectedNode, setSelectedNode] = useState<number | null>(null);

  useEffect(() => {
    // Preload topology even while the monitor is closed so fast inferences
    // can be displayed without first waiting for a separate graph request.
    if (graph) return;
    let active = true;
    setError("");
    getRouterGraph()
      .then((result) => { if (active) setGraph(result); })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => { active = false; };
  }, [graph, retryCount]);

  // A new request starts from real incoming events, never the previous replay.
  useEffect(() => {
    setReplayIndex(null);
    setSelectedNode(null);
  }, [requestId]);

  const frames: TraceFrame[] = router?.trace ?? [];
  const currentIndex = frames.length
    ? replayIndex === null ? frames.length - 1 : Math.min(replayIndex, frames.length - 1)
    : -1;
  const activeFrame = currentIndex >= 0 ? frames[currentIndex] : null;
  const activations = activeFrame?.activations ?? [];

  const positions = useMemo(
    () => graph?.nodes.map((_, index) => nodePosition(index, graph.nodes.length)) ?? [],
    [graph],
  );

  const topNode = useMemo(() => {
    if (!activations.length) return null;
    let strongest = 0;
    for (let index = 1; index < activations.length; index++) {
      if (activations[index] > activations[strongest]) strongest = index;
    }
    return strongest;
  }, [activations]);

  const inspectedNode = selectedNode ?? topNode;
  const inspectedValue = inspectedNode === null ? 0 : activations[inspectedNode] ?? 0;
  const bodyId = inspectedNode === null ? null : graph?.nodes[inspectedNode]?.root_id ?? null;
  const meanActivity = activations.length
    ? activations.reduce((sum, value) => sum + value, 0) / activations.length
    : 0;

  const isRoutingLive = phase === "routing";
  const selectedRoute = router?.route && router.route !== "pending" ? router.route : null;
  const monitorLabel =
    phase === "tool" ? "EXACT TOOL · NO ROUTER" :
    phase === "routing" ? "LIVE · ROUTER FRAME" :
    phase === "router_loading" ? "LOADING ROUTER" :
    phase === "processing" ? "PROCESSING" :
    phase === "generation" ? "ROUTER COMPLETE · GENERATING" :
    phase === "error" ? "REQUEST ERROR" :
    frames.length ? "REPLAY · RECORDED TRACE" : "READY";

  if (!open) return null;

  return (
    <aside className="brainPanel" aria-label="MaleCNS 신경망 관제실">
      <header className="brainHeader">
        <div>
          <div className="eyebrow">MALECNS · REAL ROUTER TELEMETRY</div>
          <h2>Neural Activity Monitor</h2>
        </div>
        <div className="brainHeaderActions">
          <div className={"brainStatus" + (isRoutingLive ? " live" : "")} role="status">
            {error ? "GRAPH ERROR" : monitorLabel}
          </div>
          <button className="iconButton" onClick={onClose} type="button" aria-label="Brain 패널 닫기">×</button>
        </div>
      </header>

      <div className="brainStats">
        <span>{graph ? graph.n_nodes + " nodes" : "— nodes"}</span>
        <span>{graph ? graph.n_edges + " edges" : "— edges"}</span>
        <span>{activeFrame ? "mean relative activity " + (meanActivity * 100).toFixed(1) + "%" : graph ? graph.steps + " steps" : "— steps"}</span>
      </div>

      <div className="traceEvidence" role="status">
        {phase === "tool"
          ? "이 요청은 계산기 등의 확정 도구로 처리됐어. 라우터를 통과하지 않아 뉴런 활동을 표시하지 않아."
          : frames.length
            ? "모델이 실제 계산한 상대 활성도. 형광 칼슘 측정값이나 뇌 속 해부학적 위치는 아니야."
            : phase === "processing" || phase === "router_loading"
              ? "실제 라우터 이벤트를 기다리는 중이야. 장식용 뉴런 활동은 재생하지 않아."
              : "새로운 질문을 보내면 라우터가 계산한 단계별 활성도를 기록해."}
      </div>

      <div className={"brainStage calciumStage" + (isRoutingLive && frames.length ? " calciumLive" : "")}>
        {error ? (
          <div className="brainEmpty" role="alert">
            뉴런 연결 그래프를 불러오지 못했어. {error}
            <button type="button" className="brainRetryButton" onClick={() => setRetryCount((count) => count + 1)}>다시 시도</button>
          </div>
        ) : !graph ? (
          <div className="brainEmpty" role="status">MaleCNS 그래프 로딩 중...</div>
        ) : (
          <svg
            viewBox="0 0 100 100"
            className="brainSvg calciumSvg"
            role="img"
            aria-label="실제 라우터의 정규화된 뉴런 활성도 지도. 노드 위치는 도식적 배치"
          >
            <defs>
              <filter id="calciumGlow" x="-180%" y="-180%" width="460%" height="460%">
                <feGaussianBlur stdDeviation="1.8" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
              <filter id="calciumHotGlow" x="-260%" y="-260%" width="620%" height="620%">
                <feGaussianBlur stdDeviation="3.1" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
            </defs>
            <path
              className="calciumBrainOutline"
              d="M50 12 C38 7 22 14 18 29 C10 36 10 51 17 59 C14 73 25 87 40 86 C44 92 56 92 60 86 C75 87 86 73 83 59 C90 51 90 36 82 29 C78 14 62 7 50 12 Z"
            />
            {graph.edges.map((edge, index) => {
              const src = positions[edge.src];
              const dst = positions[edge.dst];
              if (!src || !dst) return null;
              const strength = Math.min(activations[edge.src] ?? 0, activations[edge.dst] ?? 0);
              const selected = inspectedNode !== null &&
                (edge.src === inspectedNode || edge.dst === inspectedNode);
              return (
                <line
                  key={index}
                  x1={src.x} y1={src.y} x2={dst.x} y2={dst.y}
                  className={strength > 0.18 || selected ? "calciumEdge active" : "calciumEdge"}
                  style={{ opacity: selected ? 0.63 : 0.02 + Math.min(0.26, strength * 0.22) }}
                />
              );
            })}
            {graph.nodes.map((node, index) => {
              const point = positions[index];
              const activity = activations[index] ?? 0;
              const active = activity > 0.08;
              const hot = activity > 0.48;
              const selected = inspectedNode === index;
              return (
                <g
                  key={node.root_id ?? index}
                  role="button"
                  tabIndex={0}
                  aria-label={"뉴런 " + (node.root_id ?? index) + " 선택"}
                  onClick={() => setSelectedNode(index)}
                  onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") setSelectedNode(index); }}
                  style={{ cursor: "pointer" }}
                >
                  {active && (
                    <circle
                      cx={point.x} cy={point.y}
                      r={1.9 + activity * 3.9}
                      className={hot ? "calciumHalo hot" : "calciumHalo"}
                      style={{ opacity: 0.14 + activity * 0.52, animation: "none" }}
                    />
                  )}
                  <circle
                    cx={point.x} cy={point.y}
                    r={selected ? 1.5 + activity * 1.5 : 0.44 + activity * 1.5}
                    className={hot ? "calciumNode hot" : active ? "calciumNode active" : "calciumNode"}
                    style={{ opacity: selected ? 1 : active ? 0.62 + activity * 0.38 : 0.15, stroke: selected ? "#d5ffc9" : "none", strokeWidth: selected ? 0.32 : 0 }}
                  >
                    <title>
                      Node {index} · MaleCNS body {node.root_id ?? "not mapped"} ·
                      normalized router activation {(activity * 100).toFixed(1)}%
                    </title>
                  </circle>
                </g>
              );
            })}
          </svg>
        )}

        <div className="calciumHud" aria-hidden="true">
          <span>ROUTER</span>
          <i style={{ width: (activeFrame ? Math.max(3, meanActivity * 100) : 0) + "%" }} />
        </div>
        <div className={"stageChip" + (isRoutingLive ? " live" : "")}>
          {activeFrame ? activeFrame.stage.replace("_", " ") : phase === "tool" ? "tool bypass" : "awaiting trace"}
        </div>
      </div>

      {frames.length > 0 && (
        <section className="traceControls" aria-label="Recorded router trace replay">
          <div className="traceControlHeader">
            <strong>{isRoutingLive ? "LIVE TRACE" : replayIndex === null ? "LAST FRAME" : "MANUAL REPLAY"}</strong>
            <span>{currentIndex + 1} / {frames.length}</span>
          </div>
          <input
            aria-label="뉴런 활성도 기록 단계 선택"
            type="range"
            min={0}
            max={Math.max(0, frames.length - 1)}
            value={currentIndex}
            onChange={(event) => setReplayIndex(Number(event.target.value))}
          />
          <button type="button" onClick={() => setReplayIndex(null)}>최신 단계로</button>
        </section>
      )}

      {selectedRoute && (
        <div className="traceDecision">
          <span>SELECTED ROUTE</span>
          <strong>{selectedRoute} · {(100 * (router?.confidence ?? 0)).toFixed(1)}%</strong>
        </div>
      )}
      {inspectedNode !== null && graph && (
        <div className="traceDecision">
          <span>{selectedNode !== null ? "SELECTED NEURON" : "MOST ACTIVE NODE IN FRAME"}</span>
          <strong>Node {inspectedNode} · body ID {bodyId ?? "unknown"}</strong>
          <small>Normalized activation {(inspectedValue * 100).toFixed(1)}% · selected edges are highlighted</small>
        </div>
      )}
      <RoutePanel router={router} />

      <div className="brainNote">
        실제 FlyGPT 라우터의 단계별 절댓값 활성도를 프레임별 최대값으로 정규화한 그림이야.
        그래프 배치는 설명용 도식이며 초파리의 실제 뉴런 위치나 GCaMP 칼슘 촬영을 뜻하지 않아.
        단계가 끝난 후에는 LIVE가 아닌 REPLAY로만 확인할 수 있어.
      </div>
    </aside>
  );
}
