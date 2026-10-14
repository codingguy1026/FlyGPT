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
import { Icon, FlyOrb } from "./FlightIdentity";
import { THEMES } from "@/lib/flight-theme";
import {
  activeChatKey, formatChatTime, INITIAL_MESSAGE, loadChatHistory,
  makeId, MAX_STORED_CHATS, titleFrom, writeChatHistory,
  type ChatRecord, type Message,
} from "@/lib/chat-history";
import { buildMessageMeta, formatRouterModel } from "@/lib/chat-presentation";

import type { AuthUser } from "@/lib/auth";
import {
  clearKnowledge,
  clearMemory,
  clearRouteLearning,
  confirmRouteLearning,
  getHealth,
  getKnowledgeStatus,
  getRouteLearningStatus,
  sendChat,
  type HealthResponse,
  type KnowledgeStatus,
  type RouteLearningStatus,
  type RouterResult,
} from "@/lib/api";
import {
  createMemorySessionId,
  getMemorySessionId,
  setMemorySessionId,
} from "@/lib/session";

export default function FlyGPTApp({
  user,
  onSignOut,
}: {
  user: AuthUser;
  onSignOut: () => void | Promise<void>;
}) {
  const [messages, setMessages] = useState<Message[]>([INITIAL_MESSAGE]);
  const [chatHistory, setChatHistory] = useState<ChatRecord[]>([]);
  const [activeChatId, setActiveChatId] = useState("");
  const [historyReady, setHistoryReady] = useState(false);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [brainOpen, setBrainOpen] = useState(false);
  const [telemetryOpen, setTelemetryOpen] = useState(false);
  const [railOpen, setRailOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [router, setRouter] = useState<RouterResult | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [learningStatus, setLearningStatus] = useState<RouteLearningStatus | null>(null);
  const [knowledgeStatus, setKnowledgeStatus] = useState<KnowledgeStatus | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [conversationTitle, setConversationTitle] = useState("Untitled flight");
  const [themeId, setThemeId] = useState<(typeof THEMES)[number]["id"]>("bio");
  const [mode, setMode] = useState<"ask" | "code" | "research">("ask");
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const storedHistory = loadChatHistory(user.id);
    const fallbackSessionId = getMemorySessionId(user.id);
    let nextSessionId = fallbackSessionId;
    let nextChatId = makeId("chat");

    try {
      const storedActiveId = window.localStorage.getItem(activeChatKey(user.id));
      const storedActiveChat = storedHistory.find((chat) => chat.id === storedActiveId);

      if (storedActiveChat) {
        nextChatId = storedActiveChat.id;
        nextSessionId = storedActiveChat.sessionId;
        setMessages(storedActiveChat.messages.length ? storedActiveChat.messages : [INITIAL_MESSAGE]);
        setConversationTitle(storedActiveChat.title);
      }
    } catch {
      // Fall back to a fresh local conversation shell.
    }

    setChatHistory(storedHistory);
    setActiveChatId(nextChatId);
    setSessionId(nextSessionId);
    setMemorySessionId(user.id, nextSessionId);
    setHistoryReady(true);

    try {
      const savedTheme = window.localStorage.getItem("flygpt-original-theme");
      if (savedTheme && THEMES.some((theme) => theme.id === savedTheme)) {
        setThemeId(savedTheme as (typeof THEMES)[number]["id"]);
      }
    } catch {
      // Theme persistence is optional.
    }

    getRouteLearningStatus()
      .then(setLearningStatus)
      .catch(() => setLearningStatus(null));

    getKnowledgeStatus()
      .then(setKnowledgeStatus)
      .catch(() => setKnowledgeStatus(null));
  }, [user.id]);

  useEffect(() => {
    // Periodic heartbeat keeps the status honest after a connection drops.
    let active = true;
    const refreshHealth = () => {
      getHealth()
        .then((result) => {
          if (!active) return;
          setHealth(result);
          setHealthError(false);
        })
        .catch(() => {
          if (active) { setHealth(null); setHealthError(true); }
        });
    };
    refreshHealth();
    const timer = window.setInterval(refreshHealth, 30_000);
    return () => { active = false; window.clearInterval(timer); };
  }, [user.id]);

  useEffect(() => {
    try {
      window.localStorage.setItem("flygpt-original-theme", themeId);
    } catch {
      // Ignore storage failures.
    }
  }, [themeId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  useEffect(() => {
    if (!historyReady || !activeChatId || !sessionId) return;

    try {
      window.localStorage.setItem(activeChatKey(user.id), activeChatId);
    } catch {
      // Active-chat persistence is best-effort.
    }

    const hasUserMessage = messages.some((message) => message.role === "user");
    const alreadyStored = chatHistory.some((chat) => chat.id === activeChatId);

    if (!hasUserMessage && !alreadyStored) return;

    const record: ChatRecord = {
      id: activeChatId,
      title: conversationTitle,
      sessionId,
      messages,
      updatedAt: Date.now(),
    };

    setChatHistory((current) => {
      const next = [
        record,
        ...current.filter((chat) => chat.id !== activeChatId),
      ].slice(0, MAX_STORED_CHATS);
      writeChatHistory(user.id, next);
      return next;
    });
  }, [
    activeChatId,
    conversationTitle,
    historyReady,
    messages,
    sessionId,
  ]);

  const theme = useMemo(
    () => THEMES.find((item) => item.id === themeId) ?? THEMES[0],
    [themeId],
  );

  const themeVars = {
    "--accent": theme.accent,
    "--accent-2": theme.accent2,
    "--accent-rgb": theme.glow,
  } as CSSProperties;

  const online = !healthError && health?.status === "ok";

  const modelLabel = useMemo(
    () => formatRouterModel(health?.model_path),
    [health],
  );

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = input.trim();
    if (!text || sending || !sessionId) return;

    if (conversationTitle === "Untitled flight") {
      setConversationTitle(titleFrom(text));
    }

    setInput("");
    setSending(true);
    setMessages((current) => [
      ...current,
      { id: makeId("user"), role: "user", content: text },
    ]);

    try {
      const response = await sendChat(text, sessionId);
      setRouter(response.router ?? null);

      const meta = buildMessageMeta(response);
      setMessages((current) => [
        ...current,
        {
          id: makeId("assistant"),
          role: "assistant",
          content: response.answer,
          meta,
          learning: response.router
            ? {
                prompt: text,
                route: response.router.route,
              }
            : undefined,
        },
      ]);

      if (response.router?.learning_observed) {
        getRouteLearningStatus().then(setLearningStatus).catch(() => undefined);
      }
      if (
        (response.data?.knowledge_learned?.length ?? 0) > 0 ||
        (response.data?.web_verified_facts?.length ?? 0) > 0
      ) {
        getKnowledgeStatus().then(setKnowledgeStatus).catch(() => undefined);
      }
    } catch (reason: unknown) {
      const detail = reason instanceof Error ? reason.message : String(reason);
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

  async function confirmLearning(messageId: string, learning?: Message["learning"]) {
    if (!learning || learning.confirmed) return;

    try {
      const result = await confirmRouteLearning(learning.prompt, learning.route);
      setLearningStatus(result.status);
      setMessages((current) =>
        current.map((message) =>
          message.id === messageId
            ? {
                ...message,
                learning: message.learning
                  ? { ...message.learning, confirmed: true }
                  : message.learning,
                meta: [
                  ...(message.meta ?? []),
                  { text: "CONFIRMED", tone: "good" as const },
                ],
              }
            : message,
        ),
      );
    } catch {
      // Explicit route feedback is optional; keep chat usable if it fails.
    }
  }

  async function resetRouteLearning() {
    try {
      const result = await clearRouteLearning();
      setLearningStatus(result.status);
      setRouter(null);
    } catch {
      // Keep settings usable even if the backend is unavailable.
    }
  }

  async function resetKnowledge() {
    const confirmed = window.confirm(
      "이 계정의 장기 지식(검증 상태와 정정 이력 포함)을 모두 지울까요?",
    );
    if (!confirmed) return;

    try {
      const result = await clearKnowledge();
      setKnowledgeStatus(result.status);
    } catch {
      // Keep settings usable even if the backend is unavailable.
    }
  }

  function newFlight() {
    if (sending) return;
    const nextSessionId = createMemorySessionId(user.id);
    const nextChatId = makeId("chat");

    setActiveChatId(nextChatId);
    setSessionId(nextSessionId);
    setMessages([INITIAL_MESSAGE]);
    setInput("");
    setRouter(null);
    setConversationTitle("Untitled flight");
    setRailOpen(false);

    try {
      window.localStorage.setItem(activeChatKey(user.id), nextChatId);
    } catch {
      // Active-chat persistence is best-effort.
    }
  }

  function openFlight(chat: ChatRecord) {
    if (sending) return;

    setActiveChatId(chat.id);
    setSessionId(chat.sessionId);
    setMemorySessionId(user.id, chat.sessionId);
    setMessages(chat.messages.length ? chat.messages : [INITIAL_MESSAGE]);
    setConversationTitle(chat.title);
    setRouter(null);
    setInput("");
    setRailOpen(false);

    try {
      window.localStorage.setItem(activeChatKey(user.id), chat.id);
    } catch {
      // Active-chat persistence is best-effort.
    }
  }

  async function resetMemory() {
    if (!sessionId || sending) return;
    const confirmed = window.confirm("현재 비행 세션의 기억을 초기화할까요?");
    if (!confirmed) return;

    try {
      const result = await clearMemory(sessionId);
      setMessages([
        INITIAL_MESSAGE,
        {
          id: makeId("cleared"),
          role: "assistant",
          content: "🧠 메모리에서 " + result.cleared + "개 메시지를 정리했어요.",
        },
      ]);
      setRouter(null);
      setConversationTitle("Untitled flight");
    } catch (reason: unknown) {
      const detail = reason instanceof Error ? reason.message : String(reason);
      setMessages((current) => [
        ...current,
        {
          id: makeId("clear-error"),
          role: "assistant",
          content: "⚠️ 메모리 초기화 실패\n" + detail,
        },
      ]);
    }
  }

  function applyMode(next: "ask" | "code" | "research") {
    setMode(next);
    if (next === "code" && !input) setInput("다음 코드를 도와줘: ");
    if (next === "research" && !input) setInput("다음 주제를 깊게 조사해줘: ");
  }

  const topRoutes = router?.top_routes?.slice(0, 4) ?? [];

  return (
    <main className={"flightApp" + (sending ? " isThinking" : "")} style={themeVars}>
      <aside className={"flightRail" + (railOpen ? " open" : "")}>
        <div className="railBrandFull">
          <div className="railBrand">
            <img
              className="railBrandLogo"
              src="/flygpt-logo.webp"
              alt="FlyGPT logo"
            />
          </div>
          <div className="railBrandCopy">
            <strong>FlyGPT</strong>
            <span>MALECNS CONSOLE</span>
          </div>
        </div>

        <button className="newFlightButton" type="button" onClick={newFlight} disabled={sending}>
          <Icon name="plus" size={18} />
          <span>새 채팅</span>
        </button>

        <nav className="railNav" aria-label="FlyGPT navigation">
          <button className="railButton active" type="button" title="홈" aria-current="page">
            <Icon name="home" size={18} />
            <span>홈</span>
          </button>
          <button className="railButton" type="button" title="FlyGraph" onClick={() => { setBrainOpen(true); setRailOpen(false); }}>
            <Icon name="brain" size={18} />
            <span>FlyGraph</span>
          </button>
          <button className="railButton" type="button" title="테마" onClick={() => { setTelemetryOpen(true); setRailOpen(false); }}>
            <Icon name="palette" size={18} />
            <span>Telemetry</span>
          </button>
        </nav>

        <section className="historySection" aria-label="채팅 기록">
          <div className="historyLabel">
            <span>RECENT FLIGHTS</span>
            <b>{chatHistory.length}</b>
          </div>

          <div className="historyList">
            {chatHistory.length === 0 ? (
              <p className="historyEmpty">대화를 시작하면 여기에 기록돼요.</p>
            ) : (
              chatHistory.map((chat) => (
                <button
                  key={chat.id}
                  className={"historyItem" + (chat.id === activeChatId ? " active" : "")}
                  type="button"
                  onClick={() => openFlight(chat)}
                  disabled={sending}
                  title={chat.title}
                >
                  <span>{chat.title}</span>
                  <small>{formatChatTime(chat.updatedAt)}</small>
                </button>
              ))
            )}
          </div>
        </section>

        <div className="railAccount">
          <div className="railAccountAvatar" aria-hidden="true">
            {user.display_name.slice(0, 1).toUpperCase()}
          </div>
          <div className="railAccountCopy">
            <strong>{user.display_name}</strong>
            <span>{user.email}</span>
          </div>
          <button
            className="railSignOut"
            type="button"
            onClick={() => void onSignOut()}
            title="로그아웃"
          >
            로그아웃
          </button>
        </div>

        <button className="railButton railSettings" type="button" title="설정" onClick={() => { setSettingsOpen(true); setRailOpen(false); }}>
          <Icon name="settings" size={18} />
          <span>설정</span>
        </button>
      </aside>

      {railOpen && <button className="railScrim" type="button" aria-label="메뉴 닫기" onClick={() => setRailOpen(false)} />}

      <section className="flightWorkspace">
        <header className="flightHeader">
          <div className="flightTitleWrap">
            <button className="mobileMenu" type="button" aria-label="메뉴 열기" onClick={() => setRailOpen(true)}>
              <Icon name="menu" />
            </button>
            <div className="flightTitle">
              <span className="flightEyebrow">MALECNS / LIVE SESSION</span>
              <h1>{conversationTitle}</h1>
            </div>
          </div>

          <div className="headerTelemetry">
            <span className={"connectionChip " + (online ? "online" : "offline")}>
              <i />
              {online ? "online" : healthError ? "offline" : "checking"}
            </span>
            {router && (
              <button className="routeChip" type="button" onClick={() => setTelemetryOpen(true)}>
                <span>{router.route}</span>
                <strong>{router.semantic_override ? "rule" : (router.confidence * 100).toFixed(0) + "%"}</strong>
              </button>
            )}
            <button className="telemetryToggle" type="button" onClick={() => setTelemetryOpen((value) => !value)}>
              <Icon name="spark" size={18} />
              <span>Telemetry</span>
            </button>
          </div>
        </header>

        <div className="labStatusStrip" aria-label="MaleCNS system status">
          <div>
            <span>DATASET</span>
            <strong>{health?.connectome?.dataset ?? (healthError ? "unavailable" : "checking")}</strong>
          </div>
          <div>
            <span>ROUTER</span>
            <strong>{router?.n_nodes ? router.n_nodes + " nodes" : health?.model_available ? "checkpoint ready" : "standby"}</strong>
          </div>
          <div>
            <span>TRACE</span>
            <strong>{router?.steps ? router.steps + " propagation steps" : "awaiting signal"}</strong>
          </div>
          <div>
            <span>GENERATOR</span>
            <strong>{health?.generator?.model ?? (health?.generator?.configured ? "ready" : healthError ? "unavailable" : "checking")}</strong>
          </div>
          <div className={"labPulse" + (sending ? " active" : "")}>
            <i />
            <span>{sending ? "SIGNAL MOVING" : online ? "SYSTEM NOMINAL" : "BACKEND CHECK"}</span>
          </div>
        </div>

        <div className="airCanvas">
          <div className="airCurrent currentOne" />
          <div className="airCurrent currentTwo" />

          <section className="messageViewport" aria-label="대화 내용">
            <div className="flightConversation" role="log" aria-live="polite" aria-relevant="additions">
              {messages.map((message, index) => (
                <article key={message.id} className={"flightMessage " + message.role}>
                  {message.role === "assistant" && (
                    <div className="messageBeacon">
                      <FlyOrb small />
                      {index < messages.length - 1 && <span className="beaconTrail" />}
                    </div>
                  )}

                  <div className="messagePayload">
                    <div className="messageBubble">
                      <div className="messageText">{message.content}</div>

                      {message.meta && message.meta.length > 0 && (
                        <div className="messageMeta">
                          {message.meta.map((item, metaIndex) => (
                            <span
                              key={message.id + "-" + metaIndex}
                              className={"metaChip " + (item.tone ?? "muted")}
                            >
                              {item.text}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>

                    {message.role === "assistant" && (
                      <div className="payloadActions">
                        <button type="button" title="복사" onClick={() => navigator.clipboard?.writeText(message.content)}>
                          <Icon name="copy" size={16} />
                        </button>
                        <button
                          type="button"
                          title={message.learning?.confirmed ? "학습 완료" : "이 라우트가 맞았다고 학습시키기"}
                          onClick={() => void confirmLearning(message.id, message.learning)}
                          disabled={!message.learning || message.learning.confirmed}
                        >
                          <Icon name="thumb" size={16} />
                        </button>
                        <button type="button" title="경로 보기" onClick={() => setTelemetryOpen(true)}>
                          <Icon name="retry" size={16} />
                        </button>
                      </div>
                    )}
                  </div>
                </article>
              ))}

              {sending && (
                <article className="flightMessage assistant">
                  <div className="messageBeacon">
                    <FlyOrb small />
                  </div>
                  <div className="messagePayload">
                    <div className="messageBubble loadingPayload">
                      <span />
                      <span />
                      <span />
                      <b>route tracing</b>
                    </div>
                  </div>
                </article>
              )}

              <div ref={bottomRef} />
            </div>
          </section>

          <footer className="launchDock">
            <form className="launchComposer" onSubmit={handleSubmit}>
              <div className="modeStrip">
                <button className={mode === "ask" ? "active" : ""} type="button" onClick={() => applyMode("ask")}>
                  <Icon name="spark" size={14} />
                  Ask
                </button>
                <button className={mode === "code" ? "active" : ""} type="button" onClick={() => applyMode("code")}>
                  <Icon name="code" size={14} />
                  Code
                </button>
                <button className={mode === "research" ? "active" : ""} type="button" onClick={() => applyMode("research")}>
                  <Icon name="search" size={14} />
                  Research
                </button>
              </div>

              <div className="launchInputRow">
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      if (!event.nativeEvent.isComposing) {
                        event.preventDefault();
                        event.currentTarget.form?.requestSubmit();
                      }
                    }
                  }}
                  placeholder="Signal something to FlyGPT…"
                  rows={1}
                  disabled={sending}
                  aria-label="FlyGPT message"
                />
                <button
                  className="launchButton"
                  type="submit"
                  disabled={sending || !input.trim() || !sessionId}
                  aria-label="메시지 보내기"
                >
                  {sending ? <span className="launchSpinner" /> : <Icon name="send" size={20} />}
                </button>
              </div>
            </form>

            <div className="dockStatus">
              <span>{health?.app_version ? "FlyGPT v" + health.app_version : "FlyGPT · " + (healthError ? "offline" : "checking")}</span>
              <span>•</span>
              <span>{health?.connectome?.dataset ?? "dataset pending"}</span>
              <span>•</span>
              <span>{modelLabel}</span>
              {router && (
                <>
                  <span>•</span>
                  <span>{router.route}</span>
                </>
              )}
            </div>
          </footer>
        </div>
      </section>

      <aside className={"telemetryDock" + (telemetryOpen ? " open" : "")}>
        <div className="telemetryHeader">
          <div>
            <span>LIVE TELEMETRY</span>
            <h2>MaleCNS Live Trace</h2>
          </div>
          <button className="dockClose" type="button" aria-label="Telemetry 닫기" onClick={() => setTelemetryOpen(false)}>
            <Icon name="close" />
          </button>
        </div>

        <div className="pulseStage">
          <div className="pulseHalo haloOne" />
          <div className="pulseHalo haloTwo" />
          <div className="pulseHalo haloThree" />
          <FlyOrb />
          <div className="pulseLabel">
            <span>CURRENT ROUTE</span>
            <strong>{router?.route ?? "awaiting signal"}</strong>
            <b>{router ? (router.semantic_override ? "semantic rule" : (router.confidence * 100).toFixed(1) + "%") : "—"}</b>
          </div>
        </div>

        <section className="telemetrySection">
          <div className="sectionTitle">
            <span>ROUTE MIX</span>
            <button type="button" onClick={() => setBrainOpen(true)}>open brain ↗</button>
          </div>

          <div className="routeMix">
            {topRoutes.length === 0 ? (
              <p className="telemetryEmpty">질문을 보내면 라우터 분포가 여기에 나타나요.</p>
            ) : (
              topRoutes.map((item) => (
                <div className="routeMixRow" key={item.route}>
                  <div>
                    <span>{item.route}</span>
                    <strong>{(item.confidence * 100).toFixed(1)}%</strong>
                  </div>
                  <i><b style={{ width: Math.max(3, item.confidence * 100) + "%" }} /></i>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="telemetrySection systemGrid">
          <div>
            <span>BACKEND</span>
            <strong>{health?.app_version ? "v" + health.app_version : "checking"}</strong>
          </div>
          <div>
            <span>MEMORY</span>
            <strong>{health?.memory?.enabled ? "ready" : "pending"}</strong>
          </div>
          <div>
            <span>GENERATOR</span>
            <strong>{health?.generator?.model ?? (health?.generator?.configured ? "ready" : "fallback")}</strong>
          </div>
          <div>
            <span>ROUTER</span>
            <strong>{modelLabel}</strong>
          </div>
        </section>

        <section className="telemetrySection">
          <div className="sectionTitle">
            <span>FLIGHT COLOR</span>
            <small>{theme.name}</small>
          </div>
          <div className="themeSwatches">
            {THEMES.map((item) => (
              <button
                key={item.id}
                className={themeId === item.id ? "selected" : ""}
                type="button"
                aria-label={item.name}
                style={{ background: "linear-gradient(135deg," + item.accent + "," + item.accent2 + ")" }}
                onClick={() => setThemeId(item.id)}
              />
            ))}
          </div>
        </section>

        <div className="telemetryActions">
          <button type="button" onClick={newFlight} disabled={sending}>
            <Icon name="plus" size={17} />
            New flight
          </button>
          <button type="button" onClick={resetMemory} disabled={sending}>
            <Icon name="retry" size={17} />
            Clear memory
          </button>
        </div>
      </aside>

      {telemetryOpen && <button className="telemetryScrim" type="button" aria-label="Telemetry 닫기" onClick={() => setTelemetryOpen(false)} />}

      <BrainPanel open={brainOpen} onClose={() => setBrainOpen(false)} router={router} />

      {settingsOpen && (
        <div className="settingsBackdrop" onMouseDown={() => setSettingsOpen(false)}>
          <section className="settingsModal" role="dialog" aria-modal="true" onMouseDown={(event) => event.stopPropagation()}>
            <div className="settingsModalHeader">
              <div>
                <span>FLIGHT CONTROL</span>
                <h2>Settings</h2>
              </div>
              <button type="button" onClick={() => setSettingsOpen(false)}><Icon name="close" /></button>
            </div>
            <div className="settingsList">
              <div><span>Connection</span><strong>{online ? "Online" : "Check backend"}</strong></div>
              <div><span>Router</span><strong>{modelLabel}</strong></div>
              <div><span>Session</span><strong>{sessionId ? sessionId.slice(0, 12) + "…" : "loading"}</strong></div>
              <div><span>Theme</span><strong>{theme.name}</strong></div>
              <div>
                <span>Adaptive learning</span>
                <strong>
                  {learningStatus
                    ? learningStatus.examples + " examples"
                    : health?.adaptive_learning?.enabled
                      ? "ready"
                      : "pending"}
                </strong>
              </div>
              <div>
                <span>Long-term knowledge</span>
                <strong>
                  {knowledgeStatus
                    ? knowledgeStatus.items +
                      " items · " +
                      knowledgeStatus.verified +
                      " verified"
                    : health?.knowledge?.enabled
                      ? "ready"
                      : "pending"}
                </strong>
              </div>
              <div>
                <span>Web research</span>
                <strong>
                  {health?.research?.configured
                    ? (health.research.provider ?? "ready")
                    : "API key needed"}
                </strong>
              </div>
            </div>
            <button className="settingsDanger" type="button" onClick={resetMemory} disabled={sending}>Clear current memory</button>
            <button className="settingsDanger" type="button" onClick={() => void resetRouteLearning()}>
              Clear learned routing
            </button>
            <button className="settingsDanger" type="button" onClick={() => void resetKnowledge()}>
              Clear long-term knowledge
            </button>
          </section>
        </div>
      )}
    </main>
  );
}
