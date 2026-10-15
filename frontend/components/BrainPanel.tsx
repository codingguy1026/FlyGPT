"use client";

import { useEffect, useMemo, useState } from "react";
import type { GraphResponse, RouterResult, TraceFrame, NeuronKind, NeuronAnnotationsResponse } from "@/lib/api";
import { getRouterGraph, getRouterAnnotations } from "@/lib/api";
import RoutePanel from "./RoutePanel";

const NEURON_KINDS: { kind: NeuronKind; label: string; color: string }[] = [
  { kind: "sensory", label: "감각", color: "#73e7f4" },
  { kind: "motor", label: "운동", color: "#8caaff" },
  { kind: "interneuron", label: "국소/중간", color: "#e6c874" },
  { kind: "projection", label: "투사/상행·하행", color: "#cf9cff" },
  { kind: "other", label: "기타", color: "#f4a574" },
  { kind: "unclassified", label: "미분류", color: "#687a6e" },
];
const KIND_COLOR: Record<NeuronKind, string> = Object.fromEntries(
  NEURON_KINDS.map(({ kind, color }) => [kind, color]),
) as Record<NeuronKind, string>;
const KIND_LABEL: Record<NeuronKind, string> = Object.fromEntries(
  NEURON_KINDS.map(({ kind, label }) => [kind, label]),
) as Record<NeuronKind, string>;

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
  const [annotations, setAnnotations] = useState<NeuronAnnotationsResponse | null>(null);
  const [annotationError, setAnnotationError] = useState("");
  const [annotationRetry, setAnnotationRetry] = useState(0);
  const [mapMode, setMapMode] = useState<"activity" | "types">("activity");
  const [kindFilter, setKindFilter] = useState<NeuronKind | "all">("all");

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

  // Neuron identities are lazily loaded from official neuPrint annotations.
  // A failed query never becomes a synthetic classification.
  useEffect(() => {
    if (!open || !graph || annotations || annotationError) return;
    let active = true;
    getRouterAnnotations()
      .then((result) => { if (active) setAnnotations(result); })
      .catch((reason: unknown) => {
        if (active) setAnnotationError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => { active = false; };
  }, [open, graph, annotations, annotationError, annotationRetry]);

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
  const annotationFor = (rootId?: string) => rootId ? annotations?.by_body_id[rootId] : undefined;
  const selectedAnnotation = bodyId ? annotationFor(bodyId) : undefined;
  const selectedCategory: NeuronKind = selectedAnnotation?.kind ?? "unclassified";
  const categoryCounts = useMemo(() => {
    const counts = Object.fromEntries(NEURON_KINDS.map(({ kind }) => [kind, 0])) as Record<NeuronKind, number>;
    graph?.nodes.forEach((node) => {
      const category = node.root_id ? annotations?.by_body_id[node.root_id]?.kind : undefined;
      counts[category ?? "unclassified"] += 1;
    });
    return counts;
  }, [graph, annotations]);

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

      <section className="neuronTypeToolbar" aria-label="뉴런 시각화 모드">
        <div className="neuronTypeMode">
          <button type="button" className={mapMode === "activity" ? "selected" : ""} onClick={() => setMapMode("activity")}>활성도</button>
          <button type="button" className={mapMode === "types" ? "selected" : ""} onClick={() => setMapMode("types")}>뉴런 종류</button>
        </div>
        {mapMode === "types" && (
          <>
            <div className="neuronTypeCoverage" role="status">
              {annotations
                ? "공식 주석 확인: " + annotations.annotated + " / " + annotations.requested + "개"
                : annotationError
                  ? "뉴런 주석 조회 실패 · 임의 분류하지 않음"
                  : "neuPrint 공식 주석 조회 중..."}
              {annotationError && (
                <button type="button" onClick={() => { setAnnotationError(""); setAnnotationRetry((n) => n + 1); }}>다시 조회</button>
              )}
            </div>
            <div className="neuronTypeLegend" aria-label="뉴런 종류별 필터">
              <button type="button" className={kindFilter === "all" ? "selected" : ""} onClick={() => setKindFilter("all")}>
                전체 {graph?.n_nodes ?? 0}
              </button>
              {NEURON_KINDS.map((item) => (
                <button
                  type="button"
                  key={item.kind}
                  disabled={!annotations}
                  className={kindFilter === item.kind ? "selected" : ""}
                  onClick={() => setKindFilter(item.kind)}
                  title={item.kind === "interneuron" ? "공식 intrinsic superclass로부터 분류한 넓은 범주" : item.label}
                >
                  <i style={{ backgroundColor: item.color }} />
                  {item.label} {annotations ? categoryCounts[item.kind] : "·"}
                </button>
              ))}
            </div>
          </>
        )}
      </section>

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
              const kind: NeuronKind = annotationFor(node.root_id)?.kind ?? "unclassified";
              const color = KIND_COLOR[kind];
              const filtered = mapMode === "types" && kindFilter !== "all" && kind !== kindFilter;
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
                  {active && mapMode === "activity" && (
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
                    className={mapMode === "types" ? "calciumNode" : hot ? "calciumNode hot" : active ? "calciumNode active" : "calciumNode"}
                    style={{ opacity: filtered ? 0.08 : selected ? 1 : mapMode === "types" ? 0.92 : active ? 0.62 + activity * 0.38 : 0.15, fill: mapMode === "types" ? color : undefined, stroke: selected ? "#d5ffc9" : "none", strokeWidth: selected ? 0.32 : 0 }}
                  >
                    <title>
                      Node {index} · MaleCNS body {node.root_id ?? "not mapped"} ·
                      normalized router activation {(activity * 100).toFixed(1)}% ·
                      class {KIND_LABEL[kind]} · official superclass {annotationFor(node.root_id)?.superclass ?? "not available"}
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
      {inspectedNode !== null && graph && (
        <section className="neuronInspector" aria-label="선택된 뉴런 주석">
          <header>
            <strong>NEURON INSPECTOR</strong>
            <span>MALECNS · BIOLOGICAL ANNOTATION</span>
          </header>
          <div className="neuronInspectorPrimary">
            <span className="neuronInspectorDot" style={{ background: KIND_COLOR[selectedCategory] }} />
            <div>
              <strong>{KIND_LABEL[selectedCategory]}</strong>
              <small>Body {bodyId ?? "not mapped"} · graph node {inspectedNode}</small>
            </div>
          </div>
          <dl>
            <div><dt>Superclass</dt><dd>{selectedAnnotation?.superclass ?? "미확인"}</dd></div>
            <div><dt>Cell type</dt><dd>{selectedAnnotation?.type ?? "미확인"}</dd></div>
            <div><dt>Instance</dt><dd>{selectedAnnotation?.instance ?? "미확인"}</dd></div>
            <div><dt>Consensus NT</dt><dd>{selectedAnnotation?.consensus_nt ?? "미확인"}</dd></div>
            <div><dt>활성도</dt><dd>{activeFrame ? (inspectedValue * 100).toFixed(1) + "% (프레임 내 정규화)" : "신호 기록 없음"}</dd></div>
          </dl>
          <p>{annotations
            ? selectedAnnotation
              ? "출처: Janelia MaleCNS v1.0 / neuPrint. 화면용 그룹은 superclass에 근거하며 이 뉴런이 FlyGPT에서 수행하는 업무를 의미하지 않아."
              : "이 body ID에 대한 neuPrint 뉴런 주석을 찾지 못했어. 추측해서 분류하지 않았어."
            : annotationError
              ? "공식 주석을 불러올 수 없어. 주석 정보 없이 뉴런 역할을 단정할 수 없어."
              : "공식 주석 조회를 기다리는 중이야."}</p>
        </section>
      )}
      <RoutePanel router={router} />

      <div className="brainNote">
        실제 FlyGPT 라우터의 단계별 절댓값 활성도를 프레임별 최대값으로 정규화한 그림이야.
        그래프 배치는 설명용 도식이며 초파리의 실제 뉴런 위치나 GCaMP 칼슘 촬영을 뜻하지 않아.
        뉴런 종류 분류는 공식 superclass를 넓은 그룹으로 표시한 것이며 FlyGPT 라우터가 학습한 기능을 나타내지 않아.
        단계가 끝난 후에는 LIVE가 아닌 REPLAY로만 확인할 수 있어.
      </div>
    </aside>
  );
}
