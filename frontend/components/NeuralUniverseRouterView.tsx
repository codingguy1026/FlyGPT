"use client";

import { useMemo, useState } from "react";
import type { GraphResponse, RouterResult } from "@/lib/api";

type Props = { graph: GraphResponse; router: RouterResult | null };
type Point = { x: number; y: number; z: number };

// Deliberately schematic placement. MaleCNS anatomical coordinates are NOT
// available in the router graph, so this view must not imply real soma positions.
function makePositions(count: number): Point[] {
  const points: Point[] = [];
  const half = Math.max(1, Math.ceil(count / 2));
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const hemisphere = i % 2 ? 1 : -1;
    const rank = Math.floor(i / 2);
    const radius = Math.sqrt((rank + 0.5) / half);
    const angle = golden * rank;
    points.push({
      x: hemisphere * (0.42 + Math.cos(angle) * radius * 0.4),
      y: Math.sin(angle) * radius * 0.86,
      z: Math.cos(angle * 0.7) * radius * 0.65,
    });
  }
  return points;
}

export default function NeuralUniverseRouterView({ graph, router }: Props) {
  const [yaw, setYaw] = useState(-0.45);
  const [pitch, setPitch] = useState(0.24);
  const [zoom, setZoom] = useState(1);
  const [frameIndex, setFrameIndex] = useState<number | null>(null);
  const frames = router?.trace ?? [];
  const selected = frames.length
    ? frames[frameIndex === null ? frames.length - 1 : Math.min(frameIndex, frames.length - 1)]
    : undefined;
  const compatible = selected?.activations.length === graph.nodes.length;
  const positions = useMemo(() => makePositions(graph.nodes.length), [graph]);
  const points = useMemo(() => positions.map((point, index) => {
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const rx = point.x * cy + point.z * sy;
    const rz = -point.x * sy + point.z * cy;
    const ry = point.y * cp - rz * sp;
    const depth = point.y * sp + rz * cp;
    const perspective = 2.8 / (2.8 - depth);
    return { index, x: 50 + rx * 34 * zoom * perspective, y: 50 - ry * 34 * zoom * perspective, depth };
  }).sort((a, b) => a.depth - b.depth), [positions, yaw, pitch, zoom]);
  const byIndex = useMemo(() => {
    const array = new Array<(typeof points)[number]>(points.length);
    for (const point of points) array[point.index] = point;
    return array;
  }, [points]);

  return (
    <section className="neuralUniverseRouter" aria-label="라우터 추론 기록 3D 도식">
      <div className="neuralUniverseHeader">
        <strong>NEURAL UNIVERSE · ROUTER 3D</strong>
        <span>{compatible ? "RECORDED TRACE" : "AWAITING ROUTER TRACE"}</span>
      </div>
      <svg viewBox="0 0 100 100" role="img" aria-label="파피티 라우터의 도식적 3D 배치와 실제 계산 활성도">
        <defs>
          <radialGradient id="universeGlow">
            <stop offset="0%" stopColor="#e5fff6" />
            <stop offset="45%" stopColor="#5bf4d2" />
            <stop offset="100%" stopColor="#5bf4d2" stopOpacity="0" />
          </radialGradient>
        </defs>
        {graph.edges.slice(0, 4000).map((edge, index) => {
          const a = byIndex[edge.src], b = byIndex[edge.dst];
          if (!a || !b) return null;
          return <line key={index} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke="#4bc9b4" strokeOpacity={0.045} strokeWidth={0.09} />;
        })}
        {points.map((point) => {
          const level = compatible ? Math.max(0, Math.min(1, selected!.activations[point.index] ?? 0)) : 0;
          return <g key={point.index}>
            {level > 0.12 && <circle cx={point.x} cy={point.y} r={1 + level * 3.6}
              fill="url(#universeGlow)" opacity={0.32 + level * 0.68} />}
            <circle cx={point.x} cy={point.y} r={0.24 + level * 0.7}
              fill={level > 0.12 ? "#caffef" : "#559c92"}
              opacity={level > 0.12 ? 1 : 0.3}>
              <title>Body ID {graph.nodes[point.index]?.root_id ?? "unmapped"} · router activation {compatible ? (level * 100).toFixed(1) + "%" : "not recorded"}</title>
            </circle>
          </g>;
        })}
      </svg>
      <div className="neuralUniverseControls">
        <button type="button" onClick={() => setYaw((v) => v - 0.22)}>↶ 회전</button>
        <button type="button" onClick={() => setYaw((v) => v + 0.22)}>회전 ↷</button>
        <button type="button" onClick={() => setPitch((v) => Math.max(-1.3, v - 0.2))}>위</button>
        <button type="button" onClick={() => setPitch((v) => Math.min(1.3, v + 0.2))}>아래</button>
        <button type="button" onClick={() => setZoom((v) => Math.min(2, v + 0.15))}>＋</button>
        <button type="button" onClick={() => setZoom((v) => Math.max(0.5, v - 0.15))}>－</button>
        <button type="button" onClick={() => { setYaw(-0.45); setPitch(0.24); setZoom(1); }}>초기화</button>
      </div>
      {frames.length > 0 && <label className="neuralUniverseTimeline">
        기록 프레임 {frameIndex === null ? frames.length : frameIndex + 1} / {frames.length}
        <input type="range" min="0" max={frames.length - 1}
          value={frameIndex === null ? frames.length - 1 : Math.min(frameIndex, frames.length - 1)}
          onChange={(event) => setFrameIndex(Number(event.target.value))} />
        <button type="button" onClick={() => setFrameIndex(null)}>최신</button>
      </label>}
      <p>표시된 {graph.nodes.length.toLocaleString()}개 노드는 라우터 그래프의 도식적 배치야. 빛은 기록된 라우터 활성도만 나타내며, 실제 166,700개 MaleCNS 세포의 발화나 해부학적 좌표는 아니야.</p>
    </section>
  );
}
