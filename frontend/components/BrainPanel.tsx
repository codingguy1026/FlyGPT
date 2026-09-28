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

const SPARK_STEPS = 26;

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
  const [sparkStep, setSparkStep] = useState(-1);

  useEffect(() => {
    if (!open || graph || error) {
      return;
    }

    getRouterGraph()
      .then(setGraph)
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : String(reason));
      });
  }, [open, graph, error]);

  useEffect(() => {
    if (!open) {
      setSparkStep(-1);
    }
  }, [open]);

  useEffect(() => {
    const trace = router?.trace ?? [];
    setSparkStep(-1);

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
    if (sparkStep < 0) return;

    if (sparkStep >= SPARK_STEPS) {
      const stopTimer = window.setTimeout(() => setSparkStep(-1), 260);
      return () => window.clearTimeout(stopTimer);
    }

    const timer = window.setTimeout(
      () => setSparkStep((current) => current + 1),
      125,
    );

    return () => window.clearTimeout(timer);
  }, [sparkStep]);

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

    if (sparkStep >= 0) {
      const phase = Math.min(1, sparkStep / SPARK_STEPS);
      const centerX = 18 + phase * 64;
      const centerY = 50 + Math.sin(phase * Math.PI * 3) * 17;

      return positions.map((point, index) => {
        const dx = point.x - centerX;
        const dy = point.y - centerY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        const wave = Math.max(0, 1 - distance / 26);
        const ring = Math.max(
          0,
          1 - Math.abs(distance - (10 + phase * 13)) / 7,
        );
        const flicker =
          0.68 + (((index * 37 + sparkStep * 19) % 31) / 31) * 0.32;

        return Math.min(1, Math.max(wave, ring * 0.72) * flicker);
      });
    }

    return graph.nodes.map(
      (_, index) => activeFrame?.activations?.[index] ?? 0,
    );
  }, [activeFrame, graph, positions, sparkStep]);

  function runSparkTest() {
    if (!graph) return;
    setFrameIndex(-1);
    setSparkStep(0);
  }

  const isSparking =
    sparkStep >= 0 ||
    displayActivations.some((activity) => activity > 0.08);

  if (!open) {
    return null;
  }

  return (
    <aside className="brainPanel">
      <header className="brainHeader">
        <div>
          <div className="eyebrow">LIVE MODEL VIEW</div>
          <h2>FlyGraph Neural Map</h2>
        </div>
        <div className="brainHeaderActions">
          <button
            className={"brainSparkButton" + (sparkStep >= 0 ? " active" : "")}
            onClick={runSparkTest}
            type="button"
            disabled={!graph}
            title="활성화 반짝임 데모"
          >
            ✦ {sparkStep >= 0 ? "SPARKING" : "SPARK TEST"}
          </button>
          <div className={"brainStatus" + (isSparking ? " live" : "")}>
            {error
              ? "OFFLINE"
              : !graph
                ? "LOADING"
                : isSparking
                  ? "LIVE"
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
        <span>{graph ? `${graph.steps} steps` : "— steps"}</span>
      </div>

      <div className={"brainStage" + (isSparking ? " sparkling" : "")}>
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
            className="brainSvg"
            role="img"
            aria-label="FlyGPT connectome scaffold activation map"
          >
            <defs>
              <filter
                id="brainGlow"
                x="-120%"
                y="-120%"
                width="340%"
                height="340%"
              >
                <feGaussianBlur stdDeviation="1.45" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <filter
                id="brainHotGlow"
                x="-180%"
                y="-180%"
                width="460%"
                height="460%"
              >
                <feGaussianBlur stdDeviation="2.4" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {graph.edges.map((edge, index) => {
              const src = positions[edge.src];
              const dst = positions[edge.dst];
              if (!src || !dst) return null;

              const a = displayActivations[edge.src] ?? 0;
              const b = displayActivations[edge.dst] ?? 0;
              const activity = Math.max(a, b);
              const hot = activity > 0.18;

              return (
                <line
                  key={index}
                  x1={src.x}
                  y1={src.y}
                  x2={dst.x}
                  y2={dst.y}
                  pathLength="1"
                  className={hot ? "brainEdge hot" : "brainEdge"}
                  style={{
                    opacity: 0.07 + Math.min(0.72, activity * 0.72),
                    strokeWidth: hot ? 0.14 + activity * 0.16 : 0.1,
                    animationDelay: `-${(index % 11) * 0.055}s`,
                  }}
                />
              );
            })}

            {graph.nodes.map((node, index) => {
              const point = positions[index];
              const activity = displayActivations[index] ?? 0;
              const active = activity > 0.08;
              const bright = activity > 0.42;

              return (
                <g key={node.root_id ?? index}>
                  {active && (
                    <circle
                      cx={point.x}
                      cy={point.y}
                      r={1.55 + activity * 2.8}
                      className={
                        bright
                          ? "brainSparkHalo bright"
                          : "brainSparkHalo"
                      }
                      style={{
                        opacity: 0.16 + activity * 0.54,
                        animationDelay: `-${(index % 9) * 0.07}s`,
                      }}
                    />
                  )}

                  {bright && (
                    <circle
                      cx={point.x}
                      cy={point.y}
                      r={0.75 + activity * 1.05}
                      className="brainSparkCore"
                      style={{
                        animationDelay: `-${(index % 7) * 0.06}s`,
                      }}
                    />
                  )}

                  <circle
                    cx={point.x}
                    cy={point.y}
                    r={0.55 + activity * 1.55}
                    className={active ? "brainNode active" : "brainNode"}
                    style={{
                      opacity: active ? 0.68 + activity * 0.32 : 0.5,
                    }}
                  >
                    <title>
                      Node {index} · FlyWire {node.root_id ?? "not mapped"} ·
                      activation {(activity * 100).toFixed(1)}%
                    </title>
                  </circle>
                </g>
              );
            })}
          </svg>
        )}

        {isSparking && <div className="brainFlashWash" aria-hidden="true" />}

        <div className={"stageChip" + (isSparking ? " live" : "")}>
          {sparkStep >= 0
            ? "signal burst"
            : activeFrame?.stage?.replace("_", " ") ?? "idle"}
        </div>
      </div>

      <RoutePanel router={router} />

      <div className="brainNote">
        ✦ SPARK TEST는 시각 효과 데모입니다. 실제 질문을 보내면 FlyGPT가
        반환한 trace activation 값으로 같은 반짝임이 재생됩니다. 실제 생물학적
        뉴런 발화 영상은 아닙니다.
      </div>
    </aside>
  );
}
