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
  liveFrame: TraceFrame | null;
  streaming: boolean;
};

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

export default function BrainPanel({
  open,
  onClose,
  router,
  liveFrame,
  streaming,
}: Props) {
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [error, setError] = useState("");
  const [frameIndex, setFrameIndex] = useState(-1);

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
    const trace = router?.trace ?? [];
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
    }, 520);

    return () => window.clearInterval(timer);
  }, [router]);

  const replayFrame: TraceFrame | null =
    frameIndex >= 0 ? router?.trace?.[frameIndex] ?? null : null;
  const activeFrame: TraceFrame | null = liveFrame ?? replayFrame;

  const positions = useMemo(
    () =>
      graph?.nodes.map((_, index) =>
        nodePosition(index, graph.nodes.length),
      ) ?? [],
    [graph],
  );

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
          <div className={streaming ? "brainStatus live" : "brainStatus"}>
            {error
              ? "OFFLINE"
              : streaming
                ? "LIVE"
                : graph
                  ? "READY"
                  : "LOADING"}
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

      <div className="brainStage">
        {error ? (
          <div className="brainEmpty">뉴런 지도를 불러오지 못했습니다.{"\n"}{error}</div>
        ) : !graph ? (
          <div className="brainEmpty">파리 뇌 지도를 불러오는 중...</div>
        ) : (
          <svg
            viewBox="0 0 100 100"
            className="brainSvg"
            role="img"
            aria-label="FlyGPT connectome scaffold activation map"
          >
            {graph.edges.map((edge, index) => {
              const src = positions[edge.src];
              const dst = positions[edge.dst];
              if (!src || !dst) return null;

              const a = activeFrame?.activations?.[edge.src] ?? 0;
              const b = activeFrame?.activations?.[edge.dst] ?? 0;
              const activity = Math.max(a, b);

              return (
                <line
                  key={index}
                  x1={src.x}
                  y1={src.y}
                  x2={dst.x}
                  y2={dst.y}
                  className="brainEdge"
                  style={{
                    opacity: 0.08 + Math.min(0.55, activity * 0.5),
                  }}
                />
              );
            })}

            {graph.nodes.map((node, index) => {
              const point = positions[index];
              const activity = activeFrame?.activations?.[index] ?? 0;
              return (
                <circle
                  key={node.root_id ?? index}
                  cx={point.x}
                  cy={point.y}
                  r={0.55 + activity * 1.7}
                  className={activity > 0.08 ? "brainNode active" : "brainNode"}
                >
                  <title>
                    Node {index} · FlyWire {node.root_id ?? "not mapped"} · activation {(activity * 100).toFixed(1)}%
                  </title>
                </circle>
              );
            })}
          </svg>
        )}

        <div className="stageChip">
          {activeFrame?.stage?.replace("_", " ") ?? "idle"}
        </div>
      </div>

      <RoutePanel router={router} />

      <div className="brainNote">
        v0.7에서는 route와 propagation frame을 서버 스트림으로 받습니다.
        연결은 FlyWire scaffold 기반이고 빛나는 정도는 FlyGPT 모델 내부
        활성도입니다. 실제 생물학적 뉴런 발화 지도는 아닙니다.
      </div>
    </aside>
  );
}
