"use client";

import { useEffect, useMemo, useState } from "react";
import type {
  GraphResponse,
  RouterResult,
  TraceFrame,
} from "@/lib/api";
import { getRouterGraph } from "@/lib/api";
import RoutePanel from "./RoutePanel";

type Props = {
  open: boolean;
  onClose: () => void;
  router: RouterResult | null;
};

const CALCIUM_STEPS = 30;

function nodePosition(index: number, count: number) {
  const right = index % 2 === 1;
  const rank = Math.floor(index / 2);
  const perSide = Math.max(1, Math.ceil(count / 2));
  const progress = (rank + 0.5) / perSide;
  const angle =
    rank * (Math.PI * (3 - Math.sqrt(5))) + (right ? 0.45 : -0.45);
  const radius = Math.sqrt(progress);

  return {
    x: (right ? 65.5 : 34.5) + Math.cos(angle) * 24.5 * radius,
    y: 50 + Math.sin(angle) * 42 * radius,
  };
}

export default function BrainPanel({ open, onClose, router }: Props) {
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [error, setError] = useState("");
  const [frameIndex, setFrameIndex] = useState(-1);
  const [calciumStep, setCalciumStep] = useState(-1);

  useEffect(() => {
    if (!open || graph || error) return;

    getRouterGraph()
      .then(setGraph)
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : String(reason));
      });
  }, [open, graph, error]);

  useEffect(() => {
    if (!open) setCalciumStep(-1);
  }, [open]);

  useEffect(() => {
    const trace = router?.trace ?? [];
    setCalciumStep(-1);

    if (trace.length === 0) {
      setFrameIndex(-1);
      return;
    }

    setFrameIndex(0);
    let index = 0;
    const timer = window.setInterval(() => {
      index += 1;
      if (index >= trace.length) {
        window.clearInterval(timer);
        return;
      }
      setFrameIndex(index);
    }, 420);

    return () => window.clearInterval(timer);
  }, [router]);

  useEffect(() => {
    if (calciumStep < 0) return;

    if (calciumStep >= CALCIUM_STEPS) {
      const stopTimer = window.setTimeout(() => setCalciumStep(-1), 360);
      return () => window.clearTimeout(stopTimer);
    }

    const timer = window.setTimeout(
      () => setCalciumStep((current) => current + 1),
      120,
    );

    return () => window.clearTimeout(timer);
  }, [calciumStep]);

  const activeFrame: TraceFrame | null =
    frameIndex >= 0 ? router?.trace?.[frameIndex] ?? null : null;

  const positions = useMemo(
    () =>
      graph?.nodes.map((_, index) =>
        nodePosition(index, graph.nodes.length),
      ) ?? [],
    [graph],
  );

  const displayActivations = useMemo(() => {
    if (!graph) return [];

    if (calciumStep >= 0) {
      const phase = Math.min(1, calciumStep / CALCIUM_STEPS);
      const focusA = {
        x: 26 + phase * 45,
        y: 36 + Math.sin(phase * Math.PI * 2.4) * 13,
      };
      const focusB = {
        x: 76 - phase * 33,
        y: 64 + Math.cos(phase * Math.PI * 2.1) * 11,
      };

      return positions.map((point, index) => {
        const distanceTo = (x: number, y: number) => {
          const dx = point.x - x;
          const dy = point.y - y;
          return Math.sqrt(dx * dx + dy * dy);
        };

        const pulseA = Math.max(0, 1 - distanceTo(focusA.x, focusA.y) / 24);
        const pulseB = Math.max(0, 1 - distanceTo(focusB.x, focusB.y) / 20);
        const spontaneous =
          ((index * 47 + calciumStep * 29) % 101) > 86
            ? 0.35 + ((index * 13) % 17) / 40
            : 0;
        const flicker =
          0.72 + (((index * 31 + calciumStep * 17) % 29) / 29) * 0.28;

        return Math.min(
          1,
          Math.max(pulseA, pulseB * 0.82, spontaneous) * flicker,
        );
      });
    }

    return graph.nodes.map(
      (_, index) => activeFrame?.activations?.[index] ?? 0,
    );
  }, [activeFrame, calciumStep, graph, positions]);

  const meanActivity = useMemo(() => {
    if (displayActivations.length === 0) return 0;
    return (
      displayActivations.reduce((sum, value) => sum + value, 0) /
      displayActivations.length
    );
  }, [displayActivations]);

  function runCalciumDemo() {
    if (!graph) return;
    setFrameIndex(-1);
    setCalciumStep(0);
  }

  const isActive =
    calciumStep >= 0 ||
    displayActivations.some((activity) => activity > 0.08);

  if (!open) return null;

  return (
    <aside className="brainPanel">
      <header className="brainHeader">
        <div>
          <div className="eyebrow">GCaMP-STYLE MODEL VIEW</div>
          <h2>FlyGraph Calcium Map</h2>
        </div>
        <div className="brainHeaderActions">
          <button
            className={"brainSparkButton" + (calciumStep >= 0 ? " active" : "")}
            onClick={runCalciumDemo}
            type="button"
            disabled={!graph}
            title="칼슘 이미징 스타일 활성화 데모"
          >
            ◉ {calciumStep >= 0 ? "IMAGING" : "CALCIUM DEMO"}
          </button>
          <div className={"brainStatus" + (isActive ? " live" : "")}>
            {error
              ? "OFFLINE"
              : !graph
                ? "LOADING"
                : isActive
                  ? "ΔF/F LIVE"
                  : "READY"}
          </div>
          <button
            className="iconButton"
            onClick={onClose}
            type="button"
            aria-label="Brain 패널 닫기"
          >
            ×
          </button>
        </div>
      </header>

      <div className="brainStats">
        <span>{graph ? `${graph.n_nodes} nodes` : "— nodes"}</span>
        <span>{graph ? `${graph.n_edges} edges` : "— edges"}</span>
        <span>{isActive ? `ΔF/F ${(meanActivity * 100).toFixed(1)}%` : graph ? `${graph.steps} steps` : "— steps"}</span>
      </div>

      <div className={"brainStage calciumStage" + (isActive ? " calciumLive" : "")}>
        {error ? (
          <div className="brainEmpty">
            뉴런 지도를 불러오지 못했습니다.{"\n"}
            {error}
          </div>
        ) : !graph ? (
          <div className="brainEmpty">파리 뇌 지도를 불러오는 중...</div>
        ) : (
          <svg
            viewBox="0 0 100 100"
            className="brainSvg calciumSvg"
            role="img"
            aria-label="FlyGPT activation map rendered in a GCaMP calcium-imaging style"
          >
            <defs>
              <filter
                id="calciumGlow"
                x="-180%"
                y="-180%"
                width="460%"
                height="460%"
              >
                <feGaussianBlur stdDeviation="1.8" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <filter
                id="calciumHotGlow"
                x="-260%"
                y="-260%"
                width="620%"
                height="620%"
              >
                <feGaussianBlur stdDeviation="3.1" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
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

              const activity = Math.max(
                displayActivations[edge.src] ?? 0,
                displayActivations[edge.dst] ?? 0,
              );

              return (
                <line
                  key={index}
                  x1={src.x}
                  y1={src.y}
                  x2={dst.x}
                  y2={dst.y}
                  className={activity > 0.18 ? "calciumEdge active" : "calciumEdge"}
                  style={{
                    opacity: 0.025 + Math.min(0.22, activity * 0.22),
                  }}
                />
              );
            })}

            {graph.nodes.map((node, index) => {
              const point = positions[index];
              const activity = displayActivations[index] ?? 0;
              const active = activity > 0.08;
              const hot = activity > 0.48;

              return (
                <g key={node.root_id ?? index}>
                  {active && (
                    <circle
                      cx={point.x}
                      cy={point.y}
                      r={1.9 + activity * 3.9}
                      className={hot ? "calciumHalo hot" : "calciumHalo"}
                      style={{
                        opacity: 0.14 + activity * 0.52,
                        animationDelay: `-${(index % 8) * 0.08}s`,
                      }}
                    />
                  )}

                  <circle
                    cx={point.x}
                    cy={point.y}
                    r={0.44 + activity * 1.5}
                    className={hot ? "calciumNode hot" : active ? "calciumNode active" : "calciumNode"}
                    style={{
                      opacity: active ? 0.62 + activity * 0.38 : 0.22,
                    }}
                  >
                    <title>
                      Node {index} · FlyWire {node.root_id ?? "not mapped"} ·
                      relative fluorescence {(activity * 100).toFixed(1)}%
                    </title>
                  </circle>
                </g>
              );
            })}
          </svg>
        )}

        <div className="calciumHud" aria-hidden="true">
          <span>GCaMP</span>
          <i style={{ width: `${Math.max(3, meanActivity * 100)}%` }} />
        </div>

        <div className={"stageChip" + (isActive ? " live" : "")}>
          {calciumStep >= 0
            ? "calcium demo"
            : activeFrame?.stage?.replace("_", " ") ?? "idle"}
        </div>
      </div>

      <RoutePanel router={router} />

      <div className="brainNote">
        이 화면은 실제 GCaMP 칼슘 이미징의 형광 표현 방식을 참고해 FlyGPT의
        trace activation을 시각화한 것입니다. 살아있는 파리를 실시간 촬영한
        영상은 아니며, 밝아질수록 모델 활성도가 높은 것으로 표시됩니다.
      </div>
    </aside>
  );
}
