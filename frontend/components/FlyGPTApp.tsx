"use client";

import {
  type CSSProperties,
  type FormEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import BrainPanel from "./BrainPanel";
import {
  clearMemory,
  getHealth,
  sendChat,
  type HealthResponse,
  type RouterResult,
} from "@/lib/api";
import {
  createMemorySessionId,
  getMemorySessionId,
} from "@/lib/session";

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

type IconName =
  | "home"
  | "search"
  | "library"
  | "shirt"
  | "settings"
  | "plus"
  | "menu"
  | "share"
  | "more"
  | "copy"
  | "like"
  | "dislike"
  | "refresh"
  | "mic"
  | "send"
  | "sparkles"
  | "code"
  | "image";

const INITIAL_MESSAGE: Message = {
  id: "welcome",
  role: "assistant",
  content:
    "안녕하세요! 🪰 FlyGPT예요.\n궁금한 걸 물어보세요. 질문은 FlyWire 기반 라우터를 거쳐 가장 알맞은 처리 경로로 전달돼요.",
};

const PRESETS = [
  { id: "default", label: "기본", badge: "" },
  { id: "dev", label: "개발자", badge: "</>" },
  { id: "baseball", label: "야구", badge: "⚾" },
  { id: "space", label: "우주", badge: "✦" },
  { id: "glasses", label: "안경", badge: "⌐■-■" },
  { id: "angel", label: "천사", badge: "◯" },
  { id: "devil", label: "악마", badge: "▲" },
  { id: "cyborg", label: "사이보그", badge: "⌁" },
] as const;

const ACCENTS = [
  "#5f72ff",
  "#5470f7",
  "#46d9b7",
  "#ffd253",
  "#ff8eb7",
  "#202536",
  "#f4f6fb",
];

const BACKGROUNDS = [
  "linear-gradient(145deg,#f8faff,#e8eeff)",
  "linear-gradient(145deg,#8fd0ff,#e9f8ff)",
  "linear-gradient(145deg,#111a46,#253785)",
  "linear-gradient(145deg,#ffd9e7,#d6d0ff)",
  "linear-gradient(145deg,#111a42,#0c1230)",
];

function makeId(prefix: string) {
  return prefix + "-" + Date.now() + "-" + Math.random().toString(16).slice(2);
}

function getTitle(text: string) {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return "새 대화";
  return clean.length > 30 ? clean.slice(0, 30) + "…" : clean;
}

function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };

  if (name === "home") {
    return (
      <svg {...common}>
        <path d="M3 10.7 12 3l9 7.7" />
        <path d="M5.5 9.4V21h13V9.4" />
        <path d="M9.5 21v-6h5v6" />
      </svg>
    );
  }
  if (name === "search") {
    return (
      <svg {...common}>
        <circle cx="10.5" cy="10.5" r="6.5" />
        <path d="m15.5 15.5 5 5" />
      </svg>
    );
  }
  if (name === "library") {
    return (
      <svg {...common}>
        <path d="M4 5.5h4v14H4zM10 4h4v15.5h-4zM16 7h4v12.5h-4z" />
      </svg>
    );
  }
  if (name === "shirt") {
    return (
      <svg {...common}>
        <path d="m8.5 4 3.5 2 3.5-2 4 2.6-2.1 4-2.1-1V21H8.7V9.6l-2.1 1-2.1-4z" />
      </svg>
    );
  }
  if (name === "settings") {
    return (
      <svg {...common}>
        <circle cx="12" cy="12" r="3.2" />
        <path d="M19.4 15a1.8 1.8 0 0 0 .4 2l.1.1-2.8 2.8-.1-.1a1.8 1.8 0 0 0-2-.4 1.8 1.8 0 0 0-1.1 1.6v.2H10v-.2A1.8 1.8 0 0 0 8.9 19a1.8 1.8 0 0 0-2 .4l-.1.1L4 16.7l.1-.1a1.8 1.8 0 0 0 .4-2 1.8 1.8 0 0 0-1.6-1.1h-.2V9.6h.2a1.8 1.8 0 0 0 1.6-1.1 1.8 1.8 0 0 0-.4-2L4 6.4l2.8-2.8.1.1a1.8 1.8 0 0 0 2 .4A1.8 1.8 0 0 0 10 2.5v-.2h3.9v.2A1.8 1.8 0 0 0 15 4.1a1.8 1.8 0 0 0 2-.4l.1-.1 2.8 2.8-.1.1a1.8 1.8 0 0 0-.4 2 1.8 1.8 0 0 0 1.6 1.1h.2v3.9H21A1.8 1.8 0 0 0 19.4 15Z" />
      </svg>
    );
  }
  if (name === "plus") {
    return (
      <svg {...common}>
        <path d="M12 5v14M5 12h14" />
      </svg>
    );
  }
  if (name === "menu") {
    return (
      <svg {...common}>
        <path d="M4 7h16M4 12h16M4 17h16" />
      </svg>
    );
  }
  if (name === "share") {
    return (
      <svg {...common}>
        <path d="M12 16V4" />
        <path d="m8 8 4-4 4 4" />
        <path d="M5 12v8h14v-8" />
      </svg>
    );
  }
  if (name === "more") {
    return (
      <svg {...common}>
        <circle cx="5" cy="12" r="1" fill="currentColor" stroke="none" />
        <circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" />
        <circle cx="19" cy="12" r="1" fill="currentColor" stroke="none" />
      </svg>
    );
  }
  if (name === "copy") {
    return (
      <svg {...common}>
        <rect x="8" y="8" width="11" height="11" rx="2" />
        <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
      </svg>
    );
  }
  if (name === "like") {
    return (
      <svg {...common}>
        <path d="M7 21H4V10h3zM7 11l4-8 1.8.5a2 2 0 0 1 1.3 2.4L13 10h5.5a2 2 0 0 1 2 2.3l-1 6A3 3 0 0 1 16.6 21H7z" />
      </svg>
    );
  }
  if (name === "dislike") {
    return (
      <svg {...common}>
        <path d="M7 3H4v11h3zM7 13l4 8 1.8-.5a2 2 0 0 0 1.3-2.4L13 14h5.5a2 2 0 0 0 2-2.3l-1-6A3 3 0 0 0 16.6 3H7z" />
      </svg>
    );
  }
  if (name === "refresh") {
    return (
      <svg {...common}>
        <path d="M20 7v5h-5" />
        <path d="M18.5 16a8 8 0 1 1 .6-7.2L20 12" />
      </svg>
    );
  }
  if (name === "mic") {
    return (
      <svg {...common}>
        <rect x="8.5" y="3" width="7" height="12" rx="3.5" />
        <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3" />
      </svg>
    );
  }
  if (name === "send") {
    return (
      <svg {...common}>
        <path d="M12 19V5M6.5 10.5 12 5l5.5 5.5" />
      </svg>
    );
  }
  if (name === "sparkles") {
    return (
      <svg {...common}>
        <path d="m12 3 1.2 3.8L17 8l-3.8 1.2L12 13l-1.2-3.8L7 8l3.8-1.2zM5 14l.8 2.2L8 17l-2.2.8L5 20l-.8-2.2L2 17l2.2-.8zM19 14l.7 1.8 1.8.7-1.8.7L19 19l-.7-1.8-1.8-.7 1.8-.7z" />
      </svg>
    );
  }
  if (name === "code") {
    return (
      <svg {...common}>
        <path d="m9 7-5 5 5 5M15 7l5 5-5 5M13.5 4l-3 16" />
      </svg>
    );
  }
  if (name === "image") {
    return (
      <svg {...common}>
        <rect x="3" y="4" width="18" height="16" rx="3" />
        <circle cx="9" cy="9" r="1.4" />
        <path d="m5 18 5-5 3.2 3.2 2-2L19 18" />
      </svg>
    );
  }
  return null;
}

function FlyMark({
  accent,
  background,
  preset,
  compact = false,
}: {
  accent: string;
  background: string;
  preset: string;
  compact?: boolean;
}) {
  const badge = PRESETS.find((item) => item.id === preset)?.badge ?? "";
  const style = {
    "--fly-accent": accent,
    "--fly-bg": background,
  } as CSSProperties;

  return (
    <div className={compact ? "flyMark compact" : "flyMark"} style={style} aria-label="FlyGPT">
      <span className="antenna antennaLeft"><i /></span>
      <span className="antenna antennaRight"><i /></span>
      <span className="wing wingLeft" />
      <span className="wing wingRight" />
      <span className="flyWord">
        <b>FLY</b>
        <b>GPT</b>
      </span>
      {badge && <span className={"presetBadge preset-" + preset}>{badge}</span>}
      <span className="spark sparkOne">✦</span>
      <span className="spark sparkTwo">✦</span>
    </div>
  );
}

export default function FlyGPTApp() {
  const [messages, setMessages] = useState<Message[]>([INITIAL_MESSAGE]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [brainOpen, setBrainOpen] = useState(false);
  const [customizerOpen, setCustomizerOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [router, setRouter] = useState<RouterResult | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [conversationTitle, setConversationTitle] = useState("새 대화");
  const [preset, setPreset] = useState("default");
  const [accent, setAccent] = useState(ACCENTS[0]);
  const [avatarBackground, setAvatarBackground] = useState(BACKGROUNDS[0]);
  const [avatarReady, setAvatarReady] = useState(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setSessionId(getMemorySessionId());

    try {
      const saved = window.localStorage.getItem("flygpt-avatar-v0.7");
      if (saved) {
        const parsed = JSON.parse(saved) as {
          preset?: string;
          accent?: string;
          background?: string;
        };
        if (parsed.preset) setPreset(parsed.preset);
        if (parsed.accent) setAccent(parsed.accent);
        if (parsed.background) setAvatarBackground(parsed.background);
      }
    } catch {
      // Keep defaults when local preferences cannot be read.
    }

    if (window.innerWidth >= 1100) {
      setCustomizerOpen(true);
    }
    setAvatarReady(true);

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
    if (!avatarReady) return;

    try {
      window.localStorage.setItem(
        "flygpt-avatar-v0.7",
        JSON.stringify({ preset, accent, background: avatarBackground }),
      );
    } catch {
      // Cosmetic preference persistence is optional.
    }
  }, [preset, accent, avatarBackground, avatarReady]);

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

    if (conversationTitle === "새 대화") {
      setConversationTitle(getTitle(text));
    }

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
          text:
            response.router.route +
            " · " +
            (response.router.confidence * 100).toFixed(1) +
            "%",
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
          text: "GEN · " + (generation.model ?? generation.provider ?? "ready"),
          tone: "info",
        });

        if (generation.latency_ms != null) {
          meta.push({
            text: generation.latency_ms + " ms",
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
          content: "⚠️ 요청 실패\n" + detail,
        },
      ]);
    } finally {
      setSending(false);
    }
  }

  function handleNewChat() {
    setSessionId(createMemorySessionId());
    setMessages([INITIAL_MESSAGE]);
    setInput("");
    setRouter(null);
    setConversationTitle("새 대화");
    setBrainOpen(false);
    setMobileNavOpen(false);
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
          content:
            "🧠 기억을 정리했습니다. DB에서 " +
            result.cleared +
            "개 메시지를 삭제했어요.",
        },
      ]);
      setRouter(null);
      setConversationTitle("새 대화");
    } catch (reason: unknown) {
      const detail =
        reason instanceof Error ? reason.message : String(reason);
      setMessages((current) => [
        ...current,
        {
          id: makeId("clear-error"),
          role: "assistant",
          content: "⚠️ 기억 초기화 실패\n" + detail,
        },
      ]);
    }
  }

  function openBrain() {
    setBrainOpen(true);
    setCustomizerOpen(false);
    setMobileNavOpen(false);
  }

  function openCustomizer() {
    setCustomizerOpen(true);
    setBrainOpen(false);
    setMobileNavOpen(false);
  }

  function randomizeAvatar() {
    const nextPreset = PRESETS[Math.floor(Math.random() * PRESETS.length)];
    const nextAccent = ACCENTS[Math.floor(Math.random() * ACCENTS.length)];
    const nextBackground =
      BACKGROUNDS[Math.floor(Math.random() * BACKGROUNDS.length)];
    setPreset(nextPreset.id);
    setAccent(nextAccent);
    setAvatarBackground(nextBackground);
  }

  return (
    <main className={"appShell" + (customizerOpen ? " withCustomizer" : "")}>
      <aside className={"sidebar" + (mobileNavOpen ? " mobileOpen" : "")}>
        <div className="sidebarTop">
          <div className="sidebarBrand">
            <FlyMark
              accent={accent}
              background={avatarBackground}
              preset={preset}
              compact
            />
          </div>

          <button className="newChatButton" type="button" onClick={handleNewChat}>
            <Icon name="plus" size={18} />
            <span>새 대화</span>
          </button>

          <nav className="mainNav" aria-label="주요 메뉴">
            <button className="navItem active" type="button" onClick={() => setMobileNavOpen(false)}>
              <Icon name="home" />
              <span>홈</span>
            </button>
            <button className="navItem" type="button" onClick={openBrain}>
              <Icon name="search" />
              <span>탐색하기</span>
            </button>
            <button className="navItem mutedNav" type="button" title="대화 라이브러리 연결 준비 중">
              <Icon name="library" />
              <span>라이브러리</span>
            </button>
            <button className="navItem" type="button" onClick={openCustomizer}>
              <Icon name="shirt" />
              <span>아바타 꾸미기</span>
            </button>
            <button
              className="navItem"
              type="button"
              onClick={() => {
                setSettingsOpen(true);
                setMobileNavOpen(false);
              }}
            >
              <Icon name="settings" />
              <span>설정</span>
            </button>
          </nav>
        </div>

        <div className="recentSection">
          <div className="recentHeading">
            <span>최근 대화</span>
            <span className="recentHint">현재 세션</span>
          </div>
          <button className="recentItem active" type="button">
            <span className="recentBubble">◯</span>
            <span className="recentTitle">{conversationTitle}</span>
            <span className="recentTime">지금</span>
          </button>
        </div>

        <div className="sidebarProfile">
          <div className="profileAvatar">드</div>
          <div className="profileText">
            <strong>드가이</strong>
            <span>FlyGPT Lab</span>
          </div>
          <button className="iconOnly" type="button" onClick={() => setSettingsOpen(true)} aria-label="프로필 메뉴">
            <Icon name="more" />
          </button>
        </div>
      </aside>

      <section className="workspace">
        <header className="chatHeader">
          <div className="headerTitle">
            <button
              className="mobileMenuButton"
              type="button"
              aria-label="메뉴 열기"
              onClick={() => setMobileNavOpen((value) => !value)}
            >
              <Icon name="menu" />
            </button>
            <h1>{conversationTitle}</h1>
            <span className="titleChevron">⌄</span>
          </div>
          <div className="headerActions">
            <span className={"livePill " + (online ? "online" : "offline")}>
              <i />
              {online ? "Live" : healthError ? "Offline" : "Checking"}
            </span>
            <button className="roundAction" type="button" aria-label="공유">
              <Icon name="share" size={19} />
            </button>
            <button className="roundAction" type="button" aria-label="더 보기">
              <Icon name="more" size={20} />
            </button>
          </div>
        </header>

        <section className="messages" aria-live="polite">
          <div className="conversation">
            {messages.map((message) => (
              <article
                key={message.id}
                className={"message " + message.role}
              >
                {message.role === "assistant" && (
                  <div className="assistantAvatar" aria-hidden="true">
                    <FlyMark
                      accent={accent}
                      background={avatarBackground}
                      preset={preset}
                      compact
                    />
                  </div>
                )}

                <div className="messageBody">
                  <div className="bubble">
                    <div className="messageContent">{message.content}</div>
                    {message.meta && message.meta.length > 0 && (
                      <div className="messageMeta">
                        {message.meta.map((item, index) => (
                          <span
                            className={"messageMetaChip " + (item.tone ?? "muted")}
                            key={message.id + "-meta-" + index}
                          >
                            {item.text}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {message.role === "assistant" && (
                    <div className="messageActions">
                      <button
                        type="button"
                        aria-label="답변 복사"
                        onClick={() => navigator.clipboard?.writeText(message.content)}
                      >
                        <Icon name="copy" size={17} />
                      </button>
                      <button type="button" aria-label="좋아요">
                        <Icon name="like" size={17} />
                      </button>
                      <button type="button" aria-label="싫어요">
                        <Icon name="dislike" size={17} />
                      </button>
                      <button type="button" aria-label="다시 보기" onClick={openBrain}>
                        <Icon name="refresh" size={17} />
                      </button>
                      <button type="button" aria-label="더 보기">
                        <Icon name="more" size={17} />
                      </button>
                    </div>
                  )}
                </div>
              </article>
            ))}

            {sending && (
              <article className="message assistant">
                <div className="assistantAvatar" aria-hidden="true">
                  <FlyMark
                    accent={accent}
                    background={avatarBackground}
                    preset={preset}
                    compact
                  />
                </div>
                <div className="messageBody">
                  <div className="bubble loadingBubble">
                    <span className="thinkingDot" />
                    <span className="thinkingDot" />
                    <span className="thinkingDot" />
                    <span className="thinkingText">파리 뇌가 길을 찾는 중…</span>
                  </div>
                </div>
              </article>
            )}

            <div ref={bottomRef} />
          </div>
        </section>

        <footer className="composerArea">
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
              placeholder="무엇이든 물어보세요..."
              autoComplete="off"
              disabled={sending}
              aria-label="FlyGPT 메시지"
              rows={1}
            />

            <div className="composerBottom">
              <div className="composerTools">
                <button className="toolCircle" type="button" title="첨부 기능 준비 중">
                  <Icon name="plus" size={19} />
                </button>
                <button
                  className="toolChip"
                  type="button"
                  onClick={() => setInput((value) => value || "다음 주제를 조사해줘: ")}
                >
                  <Icon name="search" size={16} />
                  검색
                </button>
                <button className="toolChip disabledTool" type="button" title="이미지 입력은 백엔드 연결 준비 중">
                  <Icon name="image" size={16} />
                  이미지
                </button>
                <button
                  className="toolChip"
                  type="button"
                  onClick={() => setInput((value) => value || "다음 코드를 도와줘: ")}
                >
                  <Icon name="code" size={16} />
                  코딩
                </button>
                <button className="toolChip desktopTool" type="button" onClick={openBrain}>
                  <Icon name="sparkles" size={16} />
                  심층 연구
                </button>
              </div>

              <div className="composerSend">
                <button className="toolCircle micButton" type="button" title="음성 입력 준비 중">
                  <Icon name="mic" size={18} />
                </button>
                <button
                  className="sendButton"
                  type="submit"
                  disabled={sending || !input.trim() || !sessionId}
                  aria-label="메시지 보내기"
                >
                  {sending ? "…" : <Icon name="send" size={20} />}
                </button>
              </div>
            </div>
          </form>

          <div className="composerFoot">
            <span>
              {health?.app_version ?? "FlyGPT v0.7"} · {modelLabel}
            </span>
            {router && (
              <span className="routeMini">
                <i />
                {router.route} {(router.confidence * 100).toFixed(1)}%
              </span>
            )}
          </div>
        </footer>
      </section>

      {customizerOpen && (
        <aside className="customizerPanel">
          <header className="customizerHeader">
            <h2>FlyGPT 아바타 꾸미기</h2>
            <button
              className="roundAction"
              type="button"
              aria-label="아바타 패널 닫기"
              onClick={() => setCustomizerOpen(false)}
            >
              ×
            </button>
          </header>

          <div className="avatarPreview" style={{ background: avatarBackground }}>
            <FlyMark
              accent={accent}
              background={avatarBackground}
              preset={preset}
            />
            <div className="previewControls">
              <button type="button" onClick={randomizeAvatar}>
                랜덤 꾸미기
              </button>
              <button className="shuffleButton" type="button" onClick={randomizeAvatar} aria-label="랜덤">
                ↻
              </button>
            </div>
          </div>

          <div className="customizerTabs">
            <button className="active" type="button">스타일</button>
            <button type="button">액세서리</button>
            <button type="button">날개</button>
            <button type="button">색상</button>
            <button type="button">배경</button>
          </div>

          <section className="customizerSection">
            <h3>프리셋</h3>
            <div className="presetGrid">
              {PRESETS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={"presetOption " + (preset === item.id ? "selected" : "")}
                  onClick={() => setPreset(item.id)}
                >
                  <span className="presetThumb">
                    <FlyMark
                      accent={accent}
                      background={avatarBackground}
                      preset={item.id}
                      compact
                    />
                  </span>
                  <span>{item.label}</span>
                </button>
              ))}
            </div>
          </section>

          <section className="customizerSection">
            <h3>색상</h3>
            <div className="swatchRow">
              {ACCENTS.map((value) => (
                <button
                  key={value}
                  type="button"
                  className={"colorSwatch " + (accent === value ? "selected" : "")}
                  style={{ background: value }}
                  onClick={() => setAccent(value)}
                  aria-label={"색상 " + value}
                />
              ))}
            </div>
          </section>

          <section className="customizerSection">
            <h3>배경</h3>
            <div className="backgroundRow">
              {BACKGROUNDS.map((value, index) => (
                <button
                  key={value}
                  type="button"
                  className={"backgroundSwatch " + (avatarBackground === value ? "selected" : "")}
                  style={{ background: value }}
                  onClick={() => setAvatarBackground(value)}
                  aria-label={"배경 " + (index + 1)}
                />
              ))}
            </div>
          </section>
        </aside>
      )}

      <BrainPanel
        open={brainOpen}
        onClose={() => setBrainOpen(false)}
        router={router}
      />

      {mobileNavOpen && (
        <button
          className="mobileScrim"
          type="button"
          aria-label="메뉴 닫기"
          onClick={() => setMobileNavOpen(false)}
        />
      )}

      {settingsOpen && (
        <div className="modalBackdrop" role="presentation" onMouseDown={() => setSettingsOpen(false)}>
          <section className="settingsCard" role="dialog" aria-modal="true" aria-label="FlyGPT 설정" onMouseDown={(event) => event.stopPropagation()}>
            <div className="settingsHeader">
              <div>
                <span>FLYGPT SETTINGS</span>
                <h2>설정</h2>
              </div>
              <button className="roundAction" type="button" onClick={() => setSettingsOpen(false)} aria-label="설정 닫기">
                ×
              </button>
            </div>

            <div className="settingsRows">
              <div>
                <span>백엔드</span>
                <strong>{online ? "연결됨" : "연결 확인 필요"}</strong>
              </div>
              <div>
                <span>라우터</span>
                <strong>{modelLabel}</strong>
              </div>
              <div>
                <span>메모리</span>
                <strong>{health?.memory?.enabled ? "사용 중" : "상태 확인 중"}</strong>
              </div>
              <div>
                <span>Generator</span>
                <strong>{health?.generator?.model ?? (health?.generator?.configured ? "Ready" : "Fallback")}</strong>
              </div>
            </div>

            <button className="dangerButton" type="button" onClick={handleClearMemory}>
              현재 세션 기억 초기화
            </button>
          </section>
        </div>
      )}
    </main>
  );
}
