"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import BrainPanel from "./BrainPanel";
import {
  clearMemory,
  getHealth,
  sendChat,
  type HealthResponse,
  type RouterResult,
} from "@/lib/api";
import { getMemorySessionId } from "@/lib/session";

type MessageMeta = {
  text: string;
  tone?: "good" | "info" | "warn" | "muted";
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  meta?: MessageMeta[];
};

const INITIAL_MESSAGE: Message = {
  id: "welcome",
  role: "assistant",
  content:
    "안녕하세요! 무엇을 같이 풀어볼까요?\n" +
    "질문은 FlyWire 기반 라우터를 거쳐 가장 알맞은 처리 경로로 전달됩니다.",
};

function makeId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function FlyGPTApp() {
  const [messages, setMessages] = useState<Message[]>([INITIAL_MESSAGE]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [brainOpen, setBrainOpen] = useState(false);
  const [router, setRouter] = useState<RouterResult | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setSessionId(getMemorySessionId());

    getHealth()
      .then((result) => {
        setHealth(result);
        setHealthError(false);
      })
      .catch(() => {
        setHealthError(true);
      });
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  const online = !healthError && health?.status === "ok";

  const modelLabel = useMemo(() => {
    const raw = health?.model_path;
    if (!raw) return "router loading";
    return raw.split("/").pop() || raw;
  }, [health]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const text = input.trim();
    if (!text || sending || !sessionId) return;

    setInput("");
    setSending(true);
    setMessages((current) => [
      ...current,
      {
        id: makeId("user"),
        role: "user",
        content: text,
      },
    ]);

    try {
      const response = await sendChat(text, sessionId);
      setRouter(response.router ?? null);
      const meta: MessageMeta[] = [];

      if (response.router) {
        meta.push({
          text: `${response.router.route} · ${(response.router.confidence * 100).toFixed(1)}%`,
          tone: response.router.confidence >= 0.55 ? "good" : "warn",
        });

        if (response.router.model) {
          meta.push({
            text: response.router.model
              .replace("fly_router_", "")
              .replace(".pt", ""),
            tone: "muted",
          });
        }
      }

      const generation = response.data?.generation;
      if (generation?.used) {
        meta.push({
          text: `GEN · ${generation.model ?? generation.provider ?? "ready"}`,
          tone: "info",
        });

        if (generation.latency_ms != null) {
          meta.push({
            text: `${generation.latency_ms} ms`,
            tone: "muted",
          });
        }
      } else if (generation && !generation.used) {
        meta.push({
          text: "fallback",
          tone: "warn",
        });
      }

      setMessages((current) => [
        ...current,
        {
          id: makeId("assistant"),
          role: "assistant",
          content: response.answer,
          meta,
        },
      ]);
    } catch (reason: unknown) {
      const detail =
        reason instanceof Error ? reason.message : String(reason);

      setMessages((current) => [
        ...current,
        {
          id: makeId("error"),
          role: "assistant",
          content: `⚠️ 요청 실패\n${detail}`,
        },
      ]);
    } finally {
      setSending(false);
    }
  }

  async function handleClearMemory() {
    if (!sessionId) return;

    const confirmed = window.confirm(
      "이 브라우저 세션의 FlyGPT 대화 기억을 지울까요?",
    );
    if (!confirmed) return;

    try {
      const result = await clearMemory(sessionId);
      setMessages([
        INITIAL_MESSAGE,
        {
          id: makeId("memory-cleared"),
          role: "assistant",
          content: `🧠 기억을 정리했습니다. DB에서 ${result.cleared}개 메시지를 삭제했어요.`,
        },
      ]);
      setRouter(null);
    } catch (reason: unknown) {
      const detail =
        reason instanceof Error ? reason.message : String(reason);
      setMessages((current) => [
        ...current,
        {
          id: makeId("clear-error"),
          role: "assistant",
          content: `⚠️ 기억 초기화 실패\n${detail}`,
        },
      ]);
    }
  }

  const quickPrompts = [
    ["✦", "일반", "왜 하늘은 파란색이야?"],
    ["∑", "수학", "삼각형 넓이는 어떻게 구해?"],
    ["</>", "코드", "파이썬 리스트 정렬 알려줘"],
    ["↺", "기억", "내가 방금 뭐라고 했지?"],
  ];

  return (
    <main className={brainOpen ? "appShell brainOpen" : "appShell"}>
      <section className="chatPanel">
        <header className="chatHeader">
          <div className="brand">
            <div className="logo" aria-hidden="true">
              <span className="logoGlyph">🪰</span>
            </div>
            <div className="brandText">
              <div className="titleRow">
                <h1>FlyGPT</h1>
                <span className="alphaBadge">LAB · v0.7</span>
              </div>
              <div className="subtitle">Connectome-routed AI workspace</div>
            </div>
          </div>

          <div className="headerActions">
            <button
              type="button"
              className="ghostButton memoryButton"
              onClick={handleClearMemory}
              title="현재 세션 기억 지우기"
            >
              <span aria-hidden="true">↺</span>
              Memory
            </button>
            <button
              type="button"
              className={brainOpen ? "ghostButton active" : "ghostButton"}
              onClick={() => setBrainOpen((value) => !value)}
            >
              <span aria-hidden="true">◉</span>
              Brain
            </button>
            <div className={online ? "status online" : "status offline"}>
              <span className="statusDot" />
              {online ? "Live" : healthError ? "Offline" : "Checking"}
            </div>
          </div>
        </header>

        <div className="systemStrip" aria-label="시스템 상태">
          <div className="systemChip">
            <span className="systemIcon">↯</span>
            <span className="systemText">
              <small>BACKEND</small>
              <strong>{health?.app_version ?? "…"}</strong>
            </span>
          </div>
          <div className="systemChip">
            <span className="systemIcon">◈</span>
            <span className="systemText">
              <small>ROUTER</small>
              <strong>{modelLabel}</strong>
            </span>
          </div>
          <div className="systemChip">
            <span className="systemIcon">◇</span>
            <span className="systemText">
              <small>MEMORY</small>
              <strong>{health?.memory?.enabled ? "Ready" : health ? "Off" : "…"}</strong>
            </span>
          </div>
          <div className="systemChip">
            <span className="systemIcon">✦</span>
            <span className="systemText">
              <small>GENERATOR</small>
              <strong>
                {health?.generator?.configured
                  ? health.generator.model ?? "Ready"
                  : health
                    ? "Fallback"
                    : "…"}
              </strong>
            </span>
          </div>
        </div>

        <section className="messages" aria-live="polite">
          <div className="conversation">
            {messages.map((message) => (
              <article key={message.id} className={`message ${message.role}`}>
                <div className="avatar" aria-hidden="true">
                  {message.role === "assistant" ? "🪰" : "Y"}
                </div>
                <div className="messageBody">
                  <div className="messageRole">
                    {message.role === "assistant" ? "FlyGPT" : "You"}
                  </div>
                  <div className="bubble">
                    <div className="messageContent">{message.content}</div>
                    {message.meta && message.meta.length > 0 && (
                      <div className="messageMeta">
                        {message.meta.map((item, index) => (
                          <span
                            className={`messageMetaChip ${item.tone ?? "muted"}`}
                            key={`${message.id}-meta-${index}`}
                          >
                            {item.text}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </article>
            ))}

            {sending && (
              <article className="message assistant">
                <div className="avatar" aria-hidden="true">🪰</div>
                <div className="messageBody">
                  <div className="messageRole">FlyGPT</div>
                  <div className="bubble loadingBubble">
                    <span className="thinkingDot" />
                    <span className="thinkingDot" />
                    <span className="thinkingDot" />
                    <span className="thinkingText">Routing through the fly brain</span>
                  </div>
                </div>
              </article>
            )}

            <div ref={bottomRef} />
          </div>
        </section>

        <footer className="composerArea">
          <div className="composerInner">
            <div className="quickPrompts" aria-label="빠른 테스트">
              {quickPrompts.map(([icon, label, prompt]) => (
                <button
                  key={label}
                  type="button"
                  onClick={() => setInput(prompt)}
                  disabled={sending}
                >
                  <span>{icon}</span>
                  {label}
                </button>
              ))}
            </div>

            <form className="composer" onSubmit={handleSubmit}>
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }}
                placeholder="FlyGPT에게 메시지 보내기"
                autoComplete="off"
                disabled={sending}
                aria-label="FlyGPT 메시지"
                rows={1}
              />
              <button
                className="sendButton"
                type="submit"
                disabled={sending || !input.trim() || !sessionId}
                aria-label="메시지 보내기"
              >
                {sending ? "…" : "↑"}
              </button>
            </form>

            <div className="footerLine">
              <span>Enter 전송 · Shift + Enter 줄바꿈</span>
              {router ? (
                <span className="routeMini">
                  <i />
                  {router.route} {(router.confidence * 100).toFixed(1)}%
                </span>
              ) : (
                <span>local session</span>
              )}
            </div>
          </div>
        </footer>
      </section>

      <BrainPanel
        open={brainOpen}
        onClose={() => setBrainOpen(false)}
        router={router}
      />
    </main>
  );
}
