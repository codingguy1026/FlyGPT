"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import BrainPanel from "./BrainPanel";
import {
  clearMemory,
  getHealth,
  getMemoryStatus,
  sendChatStream,
  type HealthResponse,
  type MemoryStatus,
  type RouterResult,
  type TraceFrame,
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
    "안녕하세요! FlyGPT v0.7 Live Brain입니다.\n" +
    "라우터 전파 단계가 실시간으로 스트리밍되고, 오래된 대화는 memory capsule로 압축됩니다.",
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
  const [liveFrame, setLiveFrame] = useState<TraceFrame | null>(null);
  const [streamStage, setStreamStage] = useState("idle");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [memoryStatus, setMemoryStatus] = useState<MemoryStatus | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const bottomRef = useRef<HTMLDivElement | null>(null);

  async function refreshMemoryStatus(id: string) {
    try {
      setMemoryStatus(await getMemoryStatus(id));
    } catch {
      // Memory status is supplemental. Chat should keep working.
    }
  }

  useEffect(() => {
    const id = getMemorySessionId();
    setSessionId(id);
    refreshMemoryStatus(id);

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
    setLiveFrame(null);
    setStreamStage("routing");
    setMessages((current) => [
      ...current,
      {
        id: makeId("user"),
        role: "user",
        content: text,
      },
    ]);

    try {
      await sendChatStream(text, sessionId, (event) => {
        if (event.type === "route") {
          setRouter(event.router);
          setStreamStage("route");
          return;
        }

        if (event.type === "brain_step") {
          setLiveFrame(event.frame);
          setStreamStage(event.frame.stage);
          return;
        }

        if (event.type === "answer") {
          setRouter(event.payload.router ?? null);
          setMessages((current) => [
            ...current,
            {
              id: makeId("assistant"),
              role: "assistant",
              content: event.payload.answer,
            },
          ]);
          setStreamStage("answer");
          return;
        }

        if (event.type === "error") {
          throw new Error(event.detail);
        }

        if (event.type === "done") {
          setStreamStage("done");
        }
      });

      await refreshMemoryStatus(sessionId);
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
      setStreamStage("error");
    } finally {
      setSending(false);
    }
  }

  async function handleClearMemory() {
    if (!sessionId) return;

    const confirmed = window.confirm(
      "이 브라우저 세션의 단기기억과 장기기억 capsule을 모두 지울까요?",
    );
    if (!confirmed) return;

    try {
      const result = await clearMemory(sessionId);
      setMessages([
        INITIAL_MESSAGE,
        {
          id: makeId("memory-cleared"),
          role: "assistant",
          content: `🧠 기억을 정리했습니다. DB에서 ${result.cleared}개 기록을 삭제했어요.`,
        },
      ]);
      setRouter(null);
      setLiveFrame(null);
      setStreamStage("idle");
      await refreshMemoryStatus(sessionId);
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

  const memoryLabel = memoryStatus
    ? `${memoryStatus.messages} msgs · ${memoryStatus.capsules} capsules`
    : health?.memory?.enabled
      ? "ON"
      : health
        ? "OFF"
        : "…";

  return (
    <main className={brainOpen ? "appShell brainOpen" : "appShell"}>
      <section className="chatPanel">
        <header className="chatHeader">
          <div className="brand">
            <div className="logo">🪰</div>
            <div className="brandText">
              <div className="titleRow">
                <h1>FlyGPT</h1>
                <span className="alphaBadge">v0.7 LIVE</span>
              </div>
              <div className="subtitle">
                streamed FlyGraph + compressed long-term memory
              </div>
            </div>
          </div>

          <div className="headerActions">
            <button
              type="button"
              className="ghostButton memoryButton"
              onClick={handleClearMemory}
              title="현재 세션의 단기/장기 기억 지우기"
            >
              🧠 Memory
            </button>
            <button
              type="button"
              className={brainOpen ? "ghostButton active" : "ghostButton"}
              onClick={() => setBrainOpen((value) => !value)}
            >
              {sending ? "📡 Brain LIVE" : "🧬 Brain"}
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
            <strong>{memoryLabel}</strong>
          </div>
          <div>
            <span className="stripLabel">STREAM</span>
            <strong>{sending ? streamStage.toUpperCase() : health?.streaming_enabled ? "READY" : "…"}</strong>
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
                {streamStage === "routing"
                  ? "라우터 깨우는 중"
                  : `Live Brain · ${streamStage.replace("_", " ")}`}
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
              {sending ? "LIVE" : "전송"}
            </button>
          </form>
          <div className="footerLine">
            <span>FlyGPT v0.7 · Live Brain + Long Memory</span>
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
        liveFrame={liveFrame}
        streaming={sending}
      />
    </main>
  );
}
