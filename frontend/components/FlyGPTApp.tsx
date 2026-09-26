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

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

const INITIAL_MESSAGE: Message = {
  id: "welcome",
  role: "assistant",
  content:
    "안녕하세요! FlyGPT v0.6 Frontend Alpha입니다.\n" +
    "FlyWire 라우터, 수식 fast-path, 로컬 대화 기억을 새 Next.js UI에서 사용할 수 있어요.",
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
      setMessages((current) => [
        ...current,
        {
          id: makeId("assistant"),
          role: "assistant",
          content: response.answer,
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

  return (
    <main className={brainOpen ? "appShell brainOpen" : "appShell"}>
      <section className="chatPanel">
        <header className="chatHeader">
          <div className="brand">
            <div className="logo">🪰</div>
            <div className="brandText">
              <div className="titleRow">
                <h1>FlyGPT</h1>
                <span className="alphaBadge">v0.6 ALPHA</span>
              </div>
              <div className="subtitle">
                FlyWire router + memory + Next.js frontend
              </div>
            </div>
          </div>

          <div className="headerActions">
            <button
              type="button"
              className="ghostButton memoryButton"
              onClick={handleClearMemory}
              title="현재 세션 기억 지우기"
            >
              🧠 Memory
            </button>
            <button
              type="button"
              className={brainOpen ? "ghostButton active" : "ghostButton"}
              onClick={() => setBrainOpen((value) => !value)}
            >
              🧬 Brain
            </button>
            <div className={online ? "status online" : "status offline"}>
              <span className="statusDot" />
              {online ? "ONLINE" : healthError ? "OFFLINE" : "CHECKING"}
            </div>
          </div>
        </header>

        <div className="systemStrip">
          <div>
            <span className="stripLabel">BACKEND</span>
            <strong>{health?.app_version ?? "…"}</strong>
          </div>
          <div>
            <span className="stripLabel">ROUTER</span>
            <strong>{modelLabel}</strong>
          </div>
          <div>
            <span className="stripLabel">MEMORY</span>
            <strong>{health?.memory?.enabled ? "ON" : health ? "OFF" : "…"}</strong>
          </div>
          <div>
            <span className="stripLabel">GENERATOR</span>
            <strong>
              {health?.generator?.configured
                ? health.generator.model ?? "ON"
                : health
                  ? "FALLBACK"
                  : "…"}
            </strong>
          </div>
        </div>

        <section className="messages" aria-live="polite">
          {messages.map((message) => (
            <article
              key={message.id}
              className={`message ${message.role}`}
            >
              <div className="avatar">
                {message.role === "assistant" ? "🪰" : "👤"}
              </div>
              <div className="bubble">{message.content}</div>
            </article>
          ))}

          {sending && (
            <article className="message assistant">
              <div className="avatar">🪰</div>
              <div className="bubble loadingBubble">
                <span className="thinkingDot" />
                <span className="thinkingDot" />
                <span className="thinkingDot" />
                FlyGPT 처리 중
              </div>
            </article>
          )}

          <div ref={bottomRef} />
        </section>

        <footer className="composerArea">
          <form className="composer" onSubmit={handleSubmit}>
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="FlyGPT에게 물어보기..."
              autoComplete="off"
              disabled={sending}
              aria-label="FlyGPT 메시지"
            />
            <button
              type="submit"
              disabled={sending || !input.trim() || !sessionId}
            >
              {sending ? "…" : "전송"}
            </button>
          </form>
          <div className="footerLine">
            <span>FlyGPT v0.6 Frontend Alpha</span>
            {router && (
              <span className="routeMini">
                {router.route} {(router.confidence * 100).toFixed(1)}%
              </span>
            )}
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
