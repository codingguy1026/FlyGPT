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

export default function BrainPanel({ open, onClose, router }: Props) {
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [error, setError] = useState("");
  const [frameIndex, setFrameIndex] = useState(-1);
  const [viewMode, setViewMode] = useState<"model" | "real">("model");
  const [warningOpen, setWarningOpen] = useState(false);

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
      setViewMode("model");
      setWarningOpen(false);
    }
  }, [open]);

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

  const activeFrame: TraceFrame | null =
    frameIndex >= 0 ? router?.trace?.[frameIndex] ?? null : null;

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
          <div className="brainStatus">
            {error ? "OFFLINE" : graph ? "READY" : "LOADING"}
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

      <div className="brainViewTabs" role="tablist" aria-label="FlyGraph 보기 방식">
        <button
          className={viewMode === "model" ? "active" : ""}
          type="button"
          role="tab"
          aria-selected={viewMode === "model"}
          onClick={() => setViewMode("model")}
        >
          MODEL MAP
        </button>
        <button
          className={viewMode === "real" ? "active real" : "real"}
          type="button"
          role="tab"
          aria-selected={viewMode === "real"}
          onClick={() => {
            if (viewMode !== "real") setWarningOpen(true);
          }}
        >
          REAL FLY BRAIN
        </button>
      </div>

      {viewMode === "model" ? (
        <div className="brainStats">
          <span>{graph ? `${graph.n_nodes} nodes` : "— nodes"}</span>
          <span>{graph ? `${graph.n_edges} edges` : "— edges"}</span>
          <span>{graph ? `${graph.steps} steps` : "— steps"}</span>
        </div>
      ) : (
        <div className="brainStats realBrainStats">
          <span>D. melanogaster</span>
          <span>microscopy</span>
          <span>CC BY-SA 4.0</span>
        </div>
      )}

      <div className={"brainStage" + (viewMode === "real" ? " realBrainMode" : "")}>
        {viewMode === "real" ? (
          <div className="realBrainFrame">
            <img
              src="https://commons.wikimedia.org/wiki/Special:FilePath/Blue_brain_104.jpg"
              alt="현미경으로 촬영한 초파리 Drosophila melanogaster 뇌와 강조된 보상 뉴런"
            />
            <div className="realBrainCaption">
              <strong>실제 초파리 뇌 현미경 이미지</strong>
              <span>보상 뉴런이 강조된 Drosophila melanogaster 뇌</span>
              <a
                href="https://commons.wikimedia.org/wiki/File:Blue_brain_104.jpg"
                target="_blank"
                rel="noreferrer"
              >
                Vincent Croset · Wikimedia Commons · CC BY-SA 4.0 ↗
              </a>
            </div>
          </div>
        ) : error ? (
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
          {viewMode === "real"
            ? "microscopy"
            : activeFrame?.stage?.replace("_", " ") ?? "idle"}
        </div>
      </div>

      <RoutePanel router={router} />

      <div className="brainNote">
        {viewMode === "real"
          ? "이 이미지는 실제 초파리 뇌의 현미경 이미지이며 FlyGPT의 라우팅 활성도와는 별개입니다."
          : "연결은 FlyWire scaffold 기반이고 빛나는 정도는 FlyGPT 모델 내부 활성도입니다. 실제 생물학적 뉴런 발화 지도는 아닙니다."}
      </div>

      {warningOpen && (
        <div className="brainWarningBackdrop" role="presentation">
          <section
            className="brainWarningDialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="real-brain-warning-title"
            aria-describedby="real-brain-warning-description"
          >
            <div className="brainWarningIcon">🧠</div>
            <div>
              <div className="eyebrow">BIOLOGICAL IMAGE WARNING</div>
              <h3 id="real-brain-warning-title">실제 초파리 뇌 이미지입니다</h3>
              <p id="real-brain-warning-description">
                현미경으로 촬영한 실제 생물학적 조직 이미지가 표시됩니다.
                이런 이미지가 불편하거나 혐오감을 줄 수 있다면 열지 않는 것을 권장해요.
              </p>
            </div>
            <div className="brainWarningActions">
              <button type="button" onClick={() => setWarningOpen(false)}>
                취소
              </button>
              <button
                className="confirm"
                type="button"
                onClick={() => {
                  setWarningOpen(false);
                  setViewMode("real");
                }}
              >
                그래도 보기
              </button>
            </div>
          </section>
        </div>
      )}
    </aside>
  );
}
