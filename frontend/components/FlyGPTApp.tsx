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
    "안녕하세요! FlyGPT v0.7 Frontend Alpha입니다.\n" +
    "FlyWire 라우터 + 로컬 기억 + 생성기 상태를 한 화면에서 보고, 라우팅 결과까지 확인할 수 있어요.",
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

  return (
    <main className={brainOpen ? "appShell brainOpen" : "appShell"}>
      <section className="chatPanel">
        <header className="chatHeader">
          <div className="brand">
            <div className="logo">🪰</div>
            <div className="brandText">
              <div className="titleRow">
                <h1>FlyGPT</h1>
                <span className="alphaBadge">v0.7 ALPHA</span>
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
          <div className="quickPrompts" aria-label="빠른 테스트">
            {[
              ["땡큐", "땡큐"],
              ["17 × 23", "17 × 23은?"],
              ["Python", "파이썬 리스트 정렬 알려줘"],
              ["Memory", "내가 방금 뭐라고 했지?"],
            ].map(([label, prompt]) => (
              <button
                key={label}
                type="button"
                onClick={() => setInput(prompt)}
                disabled={sending}
              >
                {label}
              </button>
            ))}
          </div>
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
            <span>FlyGPT v0.7 Frontend Alpha</span>
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
