"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./FlyGPTHome.module.css";

type Point = { x: number; y: number; z: number };
type Branch = { points: Point[]; signal: boolean; brightness: number };

function createSpecimen() {
  let seed = 1843;
  const random = () => {
    seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const branches: Branch[] = [];
  const particles: Point[] = [];
  const cloud = (rx: number, ry: number, rz: number): Point => {
    const angle = random() * 2 * Math.PI;
    const height = random() * 2 - 1;
    const radius = Math.cbrt(random());
    const ring = Math.sqrt(1 - height * height) * radius;
    return { x: Math.cos(angle) * ring * rx, y: height * radius * ry, z: Math.sin(angle) * ring * rz };
  };

  // Bilateral, procedural anatomy for the public illustration.
  for (const side of [-1, 1]) {
    for (let i = 0; i < 140; i++) {
      const angle = random() * 2 * Math.PI;
      const end = cloud(54, 40, 39);
      end.x += 89 * side;
      end.y += Math.sin(angle) * 14;
      const points = [
        { x: side * (3 + random() * 29), y: 24 + random() * 51, z: (random() - 0.5) * 40 },
        { x: side * (28 + random() * 23), y: 10 + Math.cos(angle) * 43, z: Math.sin(angle) * 37 },
        { x: end.x * 0.81, y: end.y - 15 - random() * 27, z: end.z * 1.2 },
        end,
        { x: end.x + side * (4 + random() * 15), y: end.y + (random() - 0.5) * 27, z: end.z },
      ];
      branches.push({ points, signal: i % 13 === 0, brightness: random() });
    }

    for (let i = 0; i < 850; i++) {
      const p = cloud(53, 40, 44);
      p.x += side * 84;
      p.y -= 9 + (1 - Math.abs(p.x) / 140) * 25;
      particles.push(p);
    }
  }

  for (let i = 0; i < 360; i++) {
    const p = cloud(59, 39, 40);
    particles.push({ ...p, y: p.y + 14 });
  }

  return { branches, particles };
}

export default function ConnectomeSpecimen() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewRef = useRef<"map" | "signal">("map");
  const [view, setView] = useState<"map" | "signal">("map");

  function changeView(next: typeof view) {
    viewRef.current = next;
    setView(next);
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;

    const { branches, particles } = createSpecimen();
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let width = 0;
    let height = 0;
    let frame = 0;
    let visible = true;
    let yaw = 0;
    let pitch = 0;
    let targetX = 0;
    let targetY = 0;

    function paint(now: number) {
      if (!context || !width || !height) return;
      yaw += (targetX - yaw) * 0.075;
      pitch += (targetY - pitch) * 0.075;
      const still = motion.matches;
      const t = still ? 0 : now * 0.001;
      const theta = still ? 0.16 : 0.16 + yaw + Math.sin(t * 0.32) * 0.13;
      const phi = still ? -0.1 : pitch - 0.1;
      const cos = Math.cos(theta), sin = Math.sin(theta);
      const tiltCos = Math.cos(phi), tiltSin = Math.sin(phi);
      const scale = Math.min(width / 350, height / 193);
      const signalView = viewRef.current === "signal";
      const project = ({ x, y, z }: Point) => {
        const dx = x * cos - z * sin;
        const depth = x * sin + z * cos;
        const perspective = 330 / (330 - depth);
        return {
          x: width / 2 + dx * scale * perspective,
          y: height * 0.48 + (y * tiltCos - depth * tiltSin) * scale * perspective,
        };
      };

      context.clearRect(0, 0, width, height);
      context.globalCompositeOperation = "lighter";
      for (const { points, brightness, signal } of branches) {
        const projected = points.map(project);
        const [first, second, third, fourth, last] = projected;
        const highlighted = signalView && signal;
        context.beginPath();
        context.moveTo(first.x, first.y);
        context.bezierCurveTo(second.x, second.y, third.x, third.y, fourth.x, fourth.y);
        context.quadraticCurveTo(fourth.x, fourth.y, last.x, last.y);
        context.strokeStyle = highlighted
          ? `rgba(255,190,107,${0.27 + brightness * 0.34})`
          : `rgba(149,221,101,${(0.025 + brightness * 0.14) * (signalView ? 0.48 : 1)})`;
        context.lineWidth = highlighted ? 0.95 : 0.46;
        context.stroke();

        if (highlighted) {
          const p = (t * 0.48 + brightness) % 1;
          const q = 1 - p;
          const x = q ** 3 * first.x + 3 * q ** 2 * p * second.x + 3 * q * p ** 2 * third.x + p ** 3 * fourth.x;
          const y = q ** 3 * first.y + 3 * q ** 2 * p * second.y + 3 * q * p ** 2 * third.y + p ** 3 * fourth.y;
          context.fillStyle = "#ffd99e";
          context.fillRect(x, y, 2.4, 2.4);
        }
      }

      for (let i = 0; i < particles.length; i++) {
        const { x, y } = project(particles[i]);
        const bright = i % 31 === 0;
        const size = (bright ? 1.65 : 0.65) * Math.max(0.8, scale);
        context.fillStyle = bright ? "rgba(225,255,169,.68)" : "rgba(129,206,85,.22)";
        context.fillRect(x, y, size, size);
      }
      context.globalCompositeOperation = "source-over";
    }

    function tick(now: number) {
      frame = 0;
      if (!visible) return;
      paint(now);
      if (!motion.matches) frame = window.requestAnimationFrame(tick);
    }

    function requestDraw() {
      if (!frame && visible) frame = window.requestAnimationFrame(tick);
    }

    function resize() {
      if (!canvas || !context) return;
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      requestDraw();
    }

    function move(event: PointerEvent) {
      if (!canvas || motion.matches || event.pointerType === "touch") return;
      const bounds = canvas.getBoundingClientRect();
      targetX = ((event.clientX - bounds.left) / bounds.width - 0.5) * 0.8;
      targetY = ((event.clientY - bounds.top) / bounds.height - 0.5) * 0.45;
    }

    function reset() {
      targetX = 0;
      targetY = 0;
    }

    const resizeObserver = new ResizeObserver(resize);
    const intersectionObserver = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      requestDraw();
    });
    const figure = canvas.parentElement;
    resizeObserver.observe(canvas);
    intersectionObserver.observe(canvas);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerleave", reset);
    figure?.addEventListener("click", requestDraw);
    motion.addEventListener("change", requestDraw);
    resize();

    return () => {
      window.cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerleave", reset);
      figure?.removeEventListener("click", requestDraw);
      motion.removeEventListener("change", requestDraw);
    };
  }, []);

  return (
    <figure id="connectome" className={styles.specimen} aria-labelledby="specimen-caption">
      <div className={styles.specimenHeader}>
        <div className={styles.specimenHeading}>
          <span className={styles.specimenCross} aria-hidden="true">⊹</span>
          <div><strong>MaleCNS</strong><span>DROSOPHILA MELANOGASTER</span></div>
        </div>
        <div className={styles.viewerOptions} role="group" aria-label="시각화 보기">
          <button type="button" aria-pressed={view === "map"} onClick={() => changeView("map")}>연결망</button>
          <button type="button" aria-pressed={view === "signal"} onClick={() => changeView("signal")}>신호 경로</button>
        </div>
      </div>
      <div className={styles.specimenAxes} aria-hidden="true" />
      <canvas ref={canvasRef} className={styles.canvas} aria-hidden="true" />
      <figcaption id="specimen-caption" className={styles.specimenCaption}>
        <span><i /> 커넥톰 개념도</span>
        <span aria-live="polite">{view === "map" ? "뉴런 연결 보기" : "예시 신호 경로 보기"}</span>
      </figcaption>
    </figure>
  );
}
