"use client";

import type { TraceFrame } from "@/lib/api";

export type FlyThinkingPhase =
  | "idle" | "processing" | "router_loading" | "routing"
  | "generation" | "tool" | "done" | "error";

type Props = { phase: FlyThinkingPhase; frame: TraceFrame | null };

function describeThinking(phase: FlyThinkingPhase, frame: TraceFrame | null) {
  if (phase === "router_loading") return { title: "신경망 준비 중…", detail: "라우터를 불러오는 중입니다", step: 0 };
  if (phase === "routing") {
    const match = frame && /^step_(\d+)$/.exec(frame.stage);
    return {
      title: frame?.stage === "input" ? "입력 신호 분석 중…" : "뉴런 신호 분석 중…",
      detail: frame ? `${frame.activations.length.toLocaleString("ko-KR")}개 라우터 노드 · ${match ? `전파 ${match[1]}단계` : "입력 단계"}` : "실제 실행 기록을 기다리는 중입니다",
      step: 0,
    };
  }
  if (phase === "generation") return { title: "답변 생성 중…", detail: "생성기가 답변을 작성하고 있어요", step: 1 };
  if (phase === "tool") return { title: "도구 처리 중…", detail: "계산 도구를 실행하고 있어요", step: 1 };
  return { title: "요청 처리 중…", detail: "서버의 응답을 기다리는 중입니다", step: 0 };
}

export default function FlyThinkingIndicator({ phase, frame }: Props) {
  const current = describeThinking(phase, frame);
  return (
    <div className="flyThinking" role="status" aria-live="polite" aria-atomic="true">
      <div className="flyThinkingHeader">
        <span className="flyThinkingBrand"><span className="flyThinkingSignal" aria-hidden="true" /> FLY</span>
        <span className="flyThinkingState">응답 처리 중</span>
      </div>
      <div className="flyThinkingSteps" aria-hidden="true">
        <div className={`flyThinkingStep ${current.step >= 0 ? "isActive" : ""}`}>
          <span className="flyThinkingStepTrack" /><span>요청 분석</span>
        </div>
        <div className={`flyThinkingStep ${current.step >= 1 ? "isActive" : ""}`}>
          <span className="flyThinkingStepTrack" /><span>답변 생성</span>
        </div>
      </div>
      <div className="flyThinkingStatus">
        <span className="flyThinkingSpinner" aria-hidden="true" />
        <div><strong>{current.title}</strong><span className="flyThinkingDetail">{current.detail}</span></div>
      </div>
      <div className="flyThinkingProgress" aria-hidden="true"><span /></div>
    </div>
  );
}
