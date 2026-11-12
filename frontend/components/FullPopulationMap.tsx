"use client";

import { useEffect, useMemo, useRef, type CSSProperties } from "react";

type Phase =
  | "idle" | "processing" | "router_loading" | "routing"
  | "generation" | "tool" | "done" | "error";

type Props = {
  phase: Phase;
  pending: boolean;
  routerNodes?: number | null;
  fullPopulationActive?: boolean;
};

// The 166,700 curated MaleCNS neurons are documented in the dataset workflow.
// This is a display-only schematic. No anatomical positions, neuron identities,
// or single-cell activation traces are invented here.
const POPULATION = 166_700;
const GOLDEN = Math.PI * (3 - Math.sqrt(5));
const SIGNALS: { key: Phase; label: string; color: string }[] = [
  { key: "processing", label: "입력 처리", color: "#65d6fa" },
  { key: "routing", label: "라우터 계산", color: "#b58bff" },
  { key: "generation", label: "답변 생성", color: "#a8ef77" },
  { key: "tool", label: "도구 처리", color: "#f3c36d" },
];

function signalFor(phase: Phase) {
  if (phase === "router_loading") return SIGNALS[1];
  return SIGNALS.find((item) => item.key === phase) ?? SIGNALS[0];
}

function makeSchematicPoints() {
  const points = new Float32Array(POPULATION * 2);
  const brainCount = Math.floor(POPULATION * 0.73);

  for (let i = 0; i < POPULATION; i++) {
    let x: number;
    let y: number;
    if (i < brainCount) {
      const side = i % 2 ? -1 : 1;
      const rank = Math.floor(i / 2);
      const progress = (rank + 0.5) / Math.ceil(brainCount / 2);
      const radius = Math.sqrt(progress);
      const angle = rank * GOLDEN + side * 0.55;
      x = 0.5 + side * (0.124 + Math.cos(angle) * radius * 0.227);
      y = 0.34 + Math.sin(angle) * radius * 0.265;
    } else {
      const rank = i - brainCount;
      const radius = Math.sqrt((rank + 0.5) / (POPULATION - brainCount));
      const angle = rank * GOLDEN;
      x = 0.5 + Math.cos(angle) * radius * 0.145;
      y = 0.74 + Math.sin(angle) * radius * 0.19;
    }
    points[i * 2] = x;
    points[i * 2 + 1] = y;
  }
  return points;
}

export default function FullPopulationMap({
  phase, pending, routerNodes, fullPopulationActive = false,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const points = useMemo(makeSchematicPoints, []);
  const signal = signalFor(phase);
  const status = !pending
    ? phase === "error" ? "REQUEST ERROR" : phase === "done" ? "RESPONSE COMPLETE" : "READY"
    : phase === "router_loading" ? "LOADING ROUTER"
    : phase === "routing" ? "ROUTING"
    : phase === "generation" ? "GENERATING"
    : phase === "tool" ? "TOOL PROCESSING"
    : "PROCESSING INPUT";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const base = document.createElement("canvas");
    const baseContext = base.getContext("2d");
    if (!baseContext) return;
    let animation = 0;
    let width = 0, height = 0, ratio = 1;

    function resize() {
      width = canvas!.clientWidth;
      height = canvas!.clientHeight;
      if (!width || !height) return;
      ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas!.width = Math.round(width * ratio);
      canvas!.height = Math.round(height * ratio);
      base.width = canvas!.width;
      base.height = canvas!.height;
      baseContext!.setTransform(ratio, 0, 0, ratio, 0, 0);
      baseContext!.clearRect(0, 0, width, height);
      baseContext!.fillStyle = "rgba(109, 172, 150, 0.28)";
      for (let i = 0; i < POPULATION; i++) {
        baseContext!.fillRect(points[i * 2] * width, points[i * 2 + 1] * height, 0.62, 0.62);
      }
    }

    function draw(time: number) {
      if (!width || !height) return;
      context!.setTransform(ratio, 0, 0, ratio, 0, 0);
      context!.clearRect(0, 0, width, height);
      context!.drawImage(base, 0, 0, width, height);
      if (pending && phase !== "error" && phase !== "done") {
        const stage = SIGNALS.findIndex((item) => item.key === signal.key);
        const tick = reduced ? 0 : Math.floor(time / 130);
        context!.fillStyle = signal.color;
        context!.shadowColor = signal.color;
        context!.shadowBlur = 7;
        for (let i = 0; i < 160; i++) {
          // Hashing selects decorative sparkles, not alleged neuronal firing.
          const index = (tick * 881 + i * 9119 + stage * 1949) % POPULATION;
          const x = points[index * 2] * width;
          const y = points[index * 2 + 1] * height;
          const brightness = reduced ? 0.65 : 0.35 + 0.65 * Math.abs(Math.sin(time * 0.003 + i * 0.71));
          context!.globalAlpha = brightness;
          context!.beginPath();
          context!.arc(x, y, 0.72 + brightness * 0.85, 0, Math.PI * 2);
          context!.fill();
        }
        context!.globalAlpha = 1;
        context!.shadowBlur = 0;
      }
      if (pending && !reduced) animation = window.requestAnimationFrame(draw);
    }

    function refresh() {
      window.cancelAnimationFrame(animation);
      resize();
      animation = window.requestAnimationFrame(draw);
    }
    refresh();
    window.addEventListener("resize", refresh);
    return () => {
      window.cancelAnimationFrame(animation);
      window.removeEventListener("resize", refresh);
    };
  }, [points, pending, phase, signal.color, signal.key]);

  return (
    <section className="neuralObservatory" aria-label="MaleCNS 전체 집단 관제 시각화">
      <div className="neuralObsHead">
        <div>
          <span>MALECNS / POPULATION MAP</span>
          <strong>신경 관제실</strong>
        </div>
        <span className={"neuralObsLive" + (pending ? " active" : "")}>
          <i />{pending ? "PROCESSING" : "STANDBY"}
        </span>
      </div>
      <div className="neuralObsCanvasFrame">
        <canvas ref={canvasRef} className="neuralObsCanvas" aria-hidden="true" />
        <div className="neuralObsOverlayTop">166,700 · CURATED NEURONS</div>
        <div className="neuralObsOverlayBottom">
          <span style={{ "--signal": signal.color } as CSSProperties} />
          {status}
        </div>
      </div>
      <div className="neuralObsLegend" aria-label="처리 단계별 표시 색상">
        {SIGNALS.map((item) => (
          <span key={item.key}>
            <i style={{ backgroundColor: item.color }} />
            {item.label}
          </span>
        ))}
      </div>
      <div className="neuralObsModel">
        <span>{fullPopulationActive ? "전체 집단 모델 활성" : "전체 집단 모델 활성 여부 미확인"}</span>
        <strong>{routerNodes ? `현재 라우터 ${routerNodes.toLocaleString()}개` : "라우터 정보 대기 중"}</strong>
      </div>
      <p className="neuralObsNote">
        166,700개 점의 배치는 도식이야. 색상과 반짝임은 처리 단계 시각효과이며, 실제 뉴런 위치나 개별 뉴런 발화 측정값이 아니야.
      </p>
    </section>
  );
}
