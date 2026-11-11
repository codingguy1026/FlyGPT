"use client";

import { useEffect, useRef, useState, type FormEvent, type PointerEvent as ReactPointerEvent } from "react";
import { getNeuronSkeleton, type NeuronSkeletonResponse } from "@/lib/api";

type Props = { suggestedBodyId: string | null };
type Angles = { yaw: number; pitch: number };
type ViewSize = { width: number; height: number };

const DEFAULT_BODY_ID = "12781"; // Janelia MaleCNS example neuron DNge104_R.
const INITIAL_ANGLES: Angles = { yaw: -0.48, pitch: 0.28 };
const MAX_ZOOM = 4;
const MIN_ZOOM = 0.5;

export default function NeuronMorphologyExplorer({ suggestedBodyId }: Props) {
  const [entry, setEntry] = useState(suggestedBodyId ?? DEFAULT_BODY_ID);
  const [requestedId, setRequestedId] = useState(suggestedBodyId ?? DEFAULT_BODY_ID);
  const [retry, setRetry] = useState(0);
  const [skeleton, setSkeleton] = useState<NeuronSkeletonResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [angles, setAngles] = useState<Angles>(INITIAL_ANGLES);
  const [zoom, setZoom] = useState(1);
  const [size, setSize] = useState<ViewSize>({ width: 0, height: 0 });
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const lastPointer = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    if (!suggestedBodyId) return;
    setEntry(suggestedBodyId);
    setRequestedId(suggestedBodyId);
  }, [suggestedBodyId]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    setSkeleton(null);
    getNeuronSkeleton(requestedId)
      .then((result) => {
        if (active) setSkeleton(result);
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
  }, [requestedId, retry]);

  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    const update = () => {
      setSize({ width: element.clientWidth, height: element.clientHeight });
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const element = canvas.current;
    if (!element || !size.width || !size.height) return;
    const ctx = element.getContext("2d");
    if (!ctx) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    element.width = Math.round(size.width * ratio);
    element.height = Math.round(size.height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const width = size.width, height = size.height;
    ctx.fillStyle = "#07131b";
    ctx.fillRect(0, 0, width, height);
    const bg = ctx.createRadialGradient(width / 2, height / 2, 1, width / 2, height / 2, Math.max(width, height) * .6);
    bg.addColorStop(0, "rgba(39,107,107,.20)");
    bg.addColorStop(1, "rgba(7,19,27,0)");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    if (!skeleton || skeleton.segments.length === 0) return;
    const low = skeleton.bounds.min, high = skeleton.bounds.max;
    const center = low.map((n, i) => (n + high[i]) / 2);
    const scale = 2 / Math.max(1, ...low.map((n, i) => high[i] - n));
    const base = Math.min(width * .41, height * .44) * zoom;
    const cosy = Math.cos(angles.yaw), siny = Math.sin(angles.yaw);
    const cosp = Math.cos(angles.pitch), sinp = Math.sin(angles.pitch);

    function project(x: number, y: number, z: number): [number, number, number] {
      const nx = (x - center[0]) * scale;
      const ny = (y - center[1]) * scale;
      const nz = (z - center[2]) * scale;
      const rx = nx * cosy + nz * siny;
      const rz = -nx * siny + nz * cosy;
      const ry = ny * cosp - rz * sinp;
      const depth = ny * sinp + rz * cosp;
      const perspective = 2.8 / Math.max(0.7, 2.8 - depth);
      return [width / 2 + rx * perspective * base, height / 2 - ry * perspective * base, depth];
    }

    ctx.lineCap = "round";
    for (const [ax, ay, az, bx, by, bz] of skeleton.segments) {
      const a = project(ax, ay, az);
      const b = project(bx, by, bz);
      const depth = Math.max(0, Math.min(1, ((a[2] + b[2]) / 2 + 1) / 2));
      ctx.strokeStyle = `rgba(109, 238, 213, ${(0.34 + depth * 0.56).toFixed(3)})`;
      ctx.lineWidth = 0.8 + depth * .7;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    }
  }, [skeleton, angles, zoom, size]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const id = entry.trim();
    if (!/^\d{1,16}$/.test(id) || Number(id) <= 0) {
      setError("숫자로 된 올바른 Body ID를 입력해 줘.");
      return;
    }
    setAngles(INITIAL_ANGLES);
    setZoom(1);
    setRequestedId(id);
    setRetry((value) => value + 1);
  }

  function pointerDown(event: ReactPointerEvent<HTMLCanvasElement>) {
    lastPointer.current = { x: event.clientX, y: event.clientY };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function pointerMove(event: ReactPointerEvent<HTMLCanvasElement>) {
    const previous = lastPointer.current;
    if (!previous) return;
    const dx = event.clientX - previous.x;
    const dy = event.clientY - previous.y;
    lastPointer.current = { x: event.clientX, y: event.clientY };
    setAngles((current) => ({
      yaw: current.yaw + dx * 0.01,
      pitch: Math.max(-1.5, Math.min(1.5, current.pitch + dy * 0.01)),
    }));
  }

  return (
    <div className="neuronLab">
      <div className="neuronLabIntro">
        <div>
          <span>ANATOMY, NOT SIMULATED ACTIVITY</span>
          <p>실제 MaleCNS 뉴런의 가지를 3차원 좌표로 탐색해. 드래그해서 돌리고, 확대해서 살펴봐!</p>
        </div>
        <a href="https://male-cns.janelia.org/download/" target="_blank" rel="noreferrer">공식 데이터 ↗</a>
      </div>

      <form className="neuronLabSearch" onSubmit={submit}>
        <label htmlFor="neuronBodyId">Neuron Body ID</label>
        <input
          id="neuronBodyId"
          inputMode="numeric"
          pattern="[0-9]+"
          maxLength={16}
          value={entry}
          onChange={(event) => setEntry(event.target.value)}
          placeholder="예: 12781 (DNge104_R)"
        />
        <button type="submit" disabled={loading}>불러오기</button>
      </form>

      <div className="neuronLabStage">
        <canvas
          ref={canvas}
          className="neuronLabCanvas"
          role="img"
          aria-label="공식 neuPrint 뉴런의 3차원 골격을 회전 가능한 원근 투영으로 표시"
          onPointerDown={pointerDown}
          onPointerMove={pointerMove}
          onPointerUp={() => { lastPointer.current = null; }}
          onPointerCancel={() => { lastPointer.current = null; }}
          onWheel={(event) => {
            event.preventDefault();
            setZoom((current) => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, current - Math.sign(event.deltaY) * .12)));
          }}
        />
        {(loading || error || (skeleton && skeleton.total_segments === 0)) && (
          <div className="neuronLabOverlay" role={error ? "alert" : "status"}>
            {loading ? "neuPrint에서 실제 뉴런 골격을 가져오는 중..." : error
              ? <><strong>골격을 가져올 수 없어.</strong><small>{error}</small><button type="button" onClick={() => setRetry((value) => value + 1)}>다시 시도</button></>
              : "이 Body ID에는 표시할 골격 선분이 없어."}
          </div>
        )}
        <span className="neuronLabCorner">3D COORDINATES · PERSPECTIVE VIEW</span>
      </div>

      <div className="neuronLabControls" aria-label="3차원 뉴런 뷰 조작">
        <button type="button" onClick={() => setAngles((a) => ({ ...a, yaw: a.yaw - .24 }))}>↶ 왼쪽</button>
        <button type="button" onClick={() => setAngles((a) => ({ ...a, yaw: a.yaw + .24 }))}>↷ 오른쪽</button>
        <button type="button" onClick={() => setZoom((v) => Math.min(MAX_ZOOM, v * 1.25))}>＋ 확대</button>
        <button type="button" onClick={() => setZoom((v) => Math.max(MIN_ZOOM, v / 1.25))}>－ 축소</button>
        <button type="button" onClick={() => { setAngles(INITIAL_ANGLES); setZoom(1); }}>초기화</button>
      </div>

      <div className="neuronLabStats">
        <div><span>BODY ID</span><strong>{skeleton?.body_id ?? requestedId}</strong></div>
        <div><span>VISIBLE SEGMENTS</span><strong>{skeleton ? skeleton.shown_segments.toLocaleString() : "—"}</strong></div>
        <div><span>TOTAL SEGMENTS</span><strong>{skeleton ? skeleton.total_segments.toLocaleString() : "—"}</strong></div>
        <div><span>ZOOM</span><strong>{zoom.toFixed(1)}×</strong></div>
      </div>
      <p className="neuronLabEvidence">
        출처: {skeleton ? `${skeleton.dataset} / ${skeleton.source}` : "Janelia MaleCNS v1.0 / neuPrint"}.
        {skeleton?.sampled ? " 속도를 위해 실제 선분 일부만 균일 간격으로 표시해. " : " "}
        이것은 한 뉴런의 해부학적 골격이지 초파리 뇌 전체 표면, 실제 신경 발화나 FlyGPT 라우터의 활동 화면이 아니야.
        표시에는 별도의 neuPrint 토큰이 필요해.
      </p>
    </div>
  );
}
