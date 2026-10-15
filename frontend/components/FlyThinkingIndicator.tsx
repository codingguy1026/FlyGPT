"use client";

import type { TraceFrame } from "@/lib/api";

export type FlyThinkingPhase =
  | "idle" | "processing" | "router_loading" | "routing"
  | "generation" | "tool" | "done" | "error";

type Props = {
  phase: FlyThinkingPhase;
  frame: TraceFrame | null;
};

// Copy always follows real SSE events. The dancing flies themselves are
// decorative: they are NOT a neuron count, a progress percentage, or telemetry.
function describeThinking(phase: FlyThinkingPhase, frame: TraceFrame | null) {
  switch (phase) {
    case "router_loading":
      return {
        label: "ROUTER STARTING",
        headline: "초파리 신경망 준비 중…",
        detail: "실제 라우터를 불러오는 중입니다",
      };
    case "routing": {
      if (!frame) {
        return {
          label: "ROUTING",
          headline: "뉴런 신호를 기다리는 중…",
          detail: "라우터의 실제 실행 기록을 기다리는 중입니다",
        };
      }
      const nodes = frame.activations.length;
      const index = /^step_(\d+)$/.exec(frame.stage);
      return {
        label: "LIVE NEURON TRACE",
        headline: frame.stage === "input" ? "뉴런 입력 신호 계산 중…" : "뉴런 신호 전파 중…",
        detail: `${nodes.toLocaleString("ko-KR")}개 라우터 노드 · ${index ? `전파 ${index[1]}단계` : "입력 단계"} · 실제 실행 기록`,
      };
    }
    case "generation":
      return {
        label: "RESPONSE GENERATION",
        headline: "답변을 문장으로 만드는 중…",
        detail: "라우터 작업 이후 생성기가 답변을 작성하는 중입니다",
      };
    case "tool":
      return {
        label: "EXACT TOOL",
        headline: "계산 도구 처리 중…",
        detail: "뉴런 라우터를 사용하지 않는 작업입니다",
      };
    default:
      return {
        label: "REQUEST IN PROGRESS",
        headline: "질문 처리 중…",
        detail: "서버의 다음 처리 단계를 기다리는 중입니다",
      };
  }
}

export default function FlyThinkingIndicator({ phase, frame }: Props) {
  const copy = describeThinking(phase, frame);

  return (
    <div className="flyThinking" role="status" aria-live="polite" aria-atomic="true">
      <div className="flyThinkingFlight" aria-hidden="true">
        <svg className="flyThinkingTrail" viewBox="0 0 136 70" fill="none" focusable="false">
          <path d="M5 49 C26 12 37 68 65 36 S106 4 130 40"
            stroke="currentColor" strokeWidth="1.2" strokeDasharray="3 7" strokeLinecap="round" />
          <circle cx="69" cy="33" r="2.6" fill="currentColor" />
          <circle cx="116" cy="47" r="1.5" fill="currentColor" />
        </svg>
        <span className="flyThinkingBug flyThinkingBugOne">🪰</span>
        <span className="flyThinkingBug flyThinkingBugTwo">🪰</span>
        <span className="flyThinkingSpark flyThinkingSparkOne" />
        <span className="flyThinkingSpark flyThinkingSparkTwo" />
      </div>
      <div className="flyThinkingContent">
        <span className="flyThinkingEyebrow">
          <i className="flyThinkingSignal" aria-hidden="true" />
          {copy.label}
        </span>
        <strong>{copy.headline}</strong>
        <span className="flyThinkingDetail">{copy.detail}</span>
        <span className="flyThinkingQuip">파리가 날아다니고 있습니다… 🪰</span>
      </div>
    </div>
  );
}
