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
import ConnectomePreview from "./ConnectomePreview";
import FullPopulationMap from "./FullPopulationMap";
import FlyThinkingIndicator from "./FlyThinkingIndicator";
import { Icon } from "./FlightIdentity";
import styles from "./FlyGPTApp.module.css";
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
  getGeneratorReadiness,
  getKnowledgeStatus,
  getRouteLearningStatus,
  sendChatStream,
  type HealthResponse,
  type KnowledgeStatus,
  type RouteLearningStatus,
  type RouterResult,
  type TraceFrame,
  type ChatStreamEvent,
} from "@/lib/api";
import {
  createMemorySessionId,
  getMemorySessionId,
  setMemorySessionId,
} from "@/lib/session";

const STARTERS = [
  { icon: "brain", title: "개념 이해하기", prompt: "신경망은 어떻게 학습하는지 쉽게 설명해줘." },
  { icon: "code", title: "코드 짜기", prompt: "자바스크립트로 만들 만한 작은 프로젝트를 추천해줘." },
  { icon: "spark", title: "새로운 이야기", prompt: "흥미로운 역사 이야기를 하나 들려줘." },
] as const;

const GENERATOR_LABELS = {
  checking: "모델 상태 확인 중",
  native: "파피티 자체 응답기",
  ready: "AI 모델 준비 완료",
  loading: "AI 모델 준비 중",
  unavailable: "AI 모델 연결 확인 필요",
  not_local: "외부 AI 모델",
} as const;

export default function FlyGPTApp({
  user,
  onSignOut,
}: {
  user: AuthUser;
  onSignOut: () => void | Promise<void>;
}) {
  const [messages, setMessages] = useState<Message[]>([INITIAL_MESSAGE]);
  const [chatHistory, setChatHistory] = useState<ChatRecord[]>([]);
  const [historyQuery, setHistoryQuery] = useState("");
  const [activeChatId, setActiveChatId] = useState("");
  const [historyReady, setHistoryReady] = useState(false);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [brainOpen, setBrainOpen] = useState(false);
  const [traceExpanded, setTraceExpanded] = useState(false);
  const [telemetryPreference, setTelemetryPreference] = useState<boolean | null>(null);
  const [wideViewport, setWideViewport] = useState(false);
  const [compactViewport, setCompactViewport] = useState(false);
  const [railOpen, setRailOpen] = useState(false);
  const [router, setRouter] = useState<RouterResult | null>(null);
  const [liveFrames, setLiveFrames] = useState<TraceFrame[]>([]);
  const [liveRoute, setLiveRoute] = useState<RouterResult | null>(null);
  const [monitorPhase, setMonitorPhase] = useState<"idle" | "processing" | "router_loading" | "routing" | "generation" | "tool" | "done" | "error">("idle");
  const [traceRequestId, setTraceRequestId] = useState(0);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [generatorReadiness, setGeneratorReadiness] = useState<"checking" | "native" | "ready" | "loading" | "unavailable" | "not_local">("checking");
  const [learningStatus, setLearningStatus] = useState<RouteLearningStatus | null>(null);
  const [knowledgeStatus, setKnowledgeStatus] = useState<KnowledgeStatus | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [conversationTitle, setConversationTitle] = useState("Untitled flight");
  const [themeId, setThemeId] = useState<(typeof THEMES)[number]["id"]>("bio");
  const messageViewportRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const settingsRef = useRef<HTMLDialogElement>(null);
  const railRef = useRef<HTMLElement>(null);
  const telemetryRef = useRef<HTMLElement>(null);
  const traceDetailsRef = useRef<HTMLDetailsElement>(null);
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const telemetryButtonRef = useRef<HTMLButtonElement>(null);
  const telemetryOpen = telemetryPreference ?? wideViewport;
  const activeDrawer = compactViewport && railOpen ? "history"
    : !wideViewport && telemetryOpen ? "telemetry" : null;

  useEffect(() => {
    const wide = window.matchMedia("(min-width: 1180px)");
    const compact = window.matchMedia("(max-width: 767px)");
    const update = () => {
      setWideViewport(wide.matches);
      setCompactViewport(compact.matches);
      if (!compact.matches) setRailOpen(false);
    };
    update();
    wide.addEventListener("change", update);
    compact.addEventListener("change", update);
    return () => {
      wide.removeEventListener("change", update);
      compact.removeEventListener("change", update);
    };
  }, []);

  useEffect(() => {
    if (!activeDrawer) return;
    const drawer = activeDrawer === "history" ? railRef.current : telemetryRef.current;
    const returnTo = activeDrawer === "history" ? menuButtonRef.current : telemetryButtonRef.current;
    drawer?.focus();
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key !== "Escape" || event.defaultPrevented || settingsRef.current?.open) return;
      if (activeDrawer === "history") setRailOpen(false);
      else setTelemetryPreference(false);
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("keydown", closeOnEscape);
      window.requestAnimationFrame(() => {
        if (document.activeElement === document.body || drawer?.contains(document.activeElement)) {
          returnTo?.focus();
        }
      });
    };
  }, [activeDrawer]);

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
    let active = true;
    const check = () => {
      getGeneratorReadiness()
        .then((result) => {
          if (active) setGeneratorReadiness(result.state);
        })
        .catch(() => {
          if (active) setGeneratorReadiness("unavailable");
        });
    };
    check();
    const timer = window.setInterval(check, 3_000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem("flygpt-original-theme", themeId);
    } catch {
      // Ignore storage failures.
    }
  }, [themeId]);

  useEffect(() => {
    const viewport = messageViewportRef.current;
    if (viewport) viewport.scrollTop = viewport.scrollHeight;
  }, [messages, sending]);

  useEffect(() => {
    const composer = composerRef.current;
    if (!composer) return;
    composer.style.height = "auto";
    composer.style.height = composer.scrollHeight + "px";
  }, [input]);

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

  const filteredChatHistory = useMemo(() => {
    const terms = historyQuery.normalize("NFKC").toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
    if (!terms.length) return chatHistory;
    return chatHistory.filter((chat) => {
      const searchable = [chat.title, ...chat.messages.map((message) => message.content)]
        .join(" ").normalize("NFKC").toLocaleLowerCase();
      return terms.every((term) => searchable.includes(term));
    });
  }, [chatHistory, historyQuery]);

  const online = !healthError && health?.status === "ok";

  const modelLabel = useMemo(
    () => formatRouterModel(health?.papiti_v05?.enabled
      ? "papiti-v0.5-category-free-166700" : health?.model_path),
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
    setRouter(null);
    setLiveFrames([]);
    setLiveRoute(null);
    setMonitorPhase("processing");
    setTraceRequestId((current) => current + 1);
    // Technical observatory is opt-in so chat keeps the user focused.
    setMessages((current) => [
      ...current,
      { id: makeId("user"), role: "user", content: text },
    ]);

    try {
      const response = await sendChatStream(text, sessionId, (event: ChatStreamEvent) => {
        if (event.type === "status") {
          setMonitorPhase(event.phase);
        } else if (event.type === "trace") {
          setLiveFrames((current) => [...current, event.frame]);
          setMonitorPhase("routing");
        } else if (event.type === "route") {
          setLiveRoute({
            route: event.route,
            confidence: event.confidence,
            accepted: event.accepted,
          });
        }
      });
      setRouter(response.router ?? null);
      setMonitorPhase(response.router?.trace?.length ? "done" : response.type === "math_fast_path" ? "tool" : "done");

      const meta = buildMessageMeta(response);
      setMessages((current) => [
        ...current,
        {
          id: makeId("assistant"),
          role: "assistant",
          content: response.answer,
          meta,
          learning: response.router && !response.router.category_free
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
      setMonitorPhase("error");
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
    setLiveFrames([]);
    setLiveRoute(null);
    setMonitorPhase("idle");
    setTraceRequestId((current) => current + 1);
    setBrainOpen(false);
    setTraceExpanded(false);
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
    setLiveFrames([]);
    setLiveRoute(null);
    setMonitorPhase("idle");
    setTraceRequestId((current) => current + 1);
    setBrainOpen(false);
    setTraceExpanded(false);
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

  function exportCurrentFlight() {
    if (sending) return;
    const exportMessages = messages.filter((message) => message.id !== INITIAL_MESSAGE.id);
    if (!exportMessages.some((message) => message.role === "user")) return;

    const exportedAt = new Date();
    const markdown = [
      "# " + (conversationTitle.replace(/\s+/g, " ").trim() || "FlyGPT 대화"),
      "",
      "_FlyGPT 대화 기록 · " + exportedAt.toLocaleString("ko-KR") + "_",
      "",
      ...exportMessages.flatMap((message) => [
        "## " + (message.role === "user" ? "사용자" : "FlyGPT"),
        "",
        message.content,
        "",
      ]),
    ].join("\n");

    // Export stays entirely in the browser; no extra API request is made.
    const url = URL.createObjectURL(new Blob([markdown], { type: "text/markdown;charset=utf-8" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "FlyGPT-" + exportedAt.toISOString().slice(0, 10) + ".md";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const topRoutes = router?.top_routes?.slice(0, 4) ?? [];
  const traceRouter = useMemo<RouterResult | null>(() => {
    if (!sending) return router;
    if (!liveFrames.length) return liveRoute;
    return {
      ...(liveRoute ?? { route: "pending", confidence: 0 }),
      trace: liveFrames,
      n_nodes: liveFrames[0]?.activations.length,
    };
  }, [sending, router, liveFrames, liveRoute]);
  // Keep the sentinel out of the transcript while preserving existing records.
  const conversationMessages = messages.filter((message) => message.id !== INITIAL_MESSAGE.id);
  const hasConversation = conversationMessages.length > 0;

  const connectionLabel = online ? "연결됨" : healthError ? "연결 끊김" : "연결 확인 중";
  const canExport = messages.some((message) => message.role === "user");
  const datasetLabel = health?.connectome?.dataset ?? "확인 중";
  const generatorLabel = GENERATOR_LABELS[generatorReadiness];

  function openBrain() {
    setRailOpen(false);
    if (!wideViewport) setTelemetryPreference(null);
    setBrainOpen(true);
  }

  function showLatestTrace() {
    setTraceExpanded(true);
    setTelemetryPreference(true);
    window.requestAnimationFrame(() => traceDetailsRef.current?.focus());
  }

  return (
    <main className={styles.app} data-inspector={telemetryOpen} style={themeVars}>
      <aside
        id="chat-history"
        ref={railRef}
        className={styles.sidebar}
        data-open={railOpen}
        aria-label="대화 메뉴"
        tabIndex={-1}
        inert={activeDrawer === "telemetry"}
      >
        <div className={styles.sidebarTop}>
          <div className={styles.brand}>
            <img src="/papiti-mark.svg" width={39} height={39} alt="" />
            <strong>FlyGPT<span>.</span></strong>
          </div>
          <button className={styles.closeMenu} type="button" aria-label="대화 메뉴 닫기" onClick={() => setRailOpen(false)}>
            <Icon name="close" size={19} />
          </button>
        </div>

        <button className={styles.newChat} type="button" onClick={() => {
          newFlight();
          if (!compactViewport) composerRef.current?.focus();
        }} disabled={sending}>
          <Icon name="plus" size={19} />
          새 대화
        </button>

        <section className={styles.history} aria-labelledby="chat-history-title">
          <div className={styles.historyHeading}>
            <h2 id="chat-history-title">최근 대화</h2>
            <span>{historyQuery.trim() ? filteredChatHistory.length + "/" + chatHistory.length : chatHistory.length}</span>
          </div>
          <label className={styles.search}>
            <Icon name="search" size={16} />
            <input
              type="search"
              value={historyQuery}
              onChange={(event) => setHistoryQuery(event.target.value)}
              placeholder="제목·내용 검색"
              aria-label="대화 기록 검색"
            />
          </label>
          <div className={styles.historyList} tabIndex={0} role="region" aria-label="저장된 대화 목록">
            {!historyReady ? (
              <p>대화 기록을 불러오는 중이에요.</p>
            ) : chatHistory.length === 0 ? (
              <p>새 대화를 시작하면<br />여기에 기록돼요.</p>
            ) : filteredChatHistory.length === 0 ? (
              <p>검색 결과가 없어요.</p>
            ) : (
              filteredChatHistory.map((chat) => (
                <button
                  key={chat.id}
                  className={styles.historyItem}
                  type="button"
                  aria-current={chat.id === activeChatId ? "true" : undefined}
                  onClick={() => openFlight(chat)}
                  disabled={sending}
                  title={chat.title}
                >
                  <Icon name="message" size={16} />
                  <span><strong>{chat.title}</strong><small>{formatChatTime(chat.updatedAt)}</small></span>
                </button>
              ))
            )}
          </div>
        </section>

        <div className={styles.sidebarBottom}>
          <nav className={styles.sidebarTools} aria-label="채팅 도구">
            <button type="button" onClick={openBrain}><Icon name="brain" size={18} /> 신경 지도</button>
            <button type="button" onClick={() => {
              setRailOpen(false);
              settingsRef.current?.showModal();
            }}><Icon name="settings" size={17} /> 설정</button>
          </nav>
          <div className={styles.account}>
            <span className={styles.accountAvatar} aria-hidden="true">{user.display_name.slice(0, 1).toUpperCase()}</span>
            <span className={styles.accountText}><strong>{user.display_name}</strong><small title={user.email}>{user.email}</small></span>
            <button className={styles.iconButton} type="button" aria-label="로그아웃" title="로그아웃" onClick={() => void onSignOut()}>
              <Icon name="logout" size={18} />
            </button>
          </div>
        </div>
      </aside>

      <section className={styles.workspace} aria-label="채팅" inert={Boolean(activeDrawer)}>
        <header className={styles.header}>
          <div className={styles.headerTitle}>
            <button
              ref={menuButtonRef}
              className={styles.menuButton}
              type="button"
              aria-label="대화 메뉴 열기"
              aria-controls="chat-history"
              aria-expanded={railOpen}
              onClick={() => setRailOpen(true)}
            ><Icon name="menu" /></button>
            <div>
              <h1 title={hasConversation ? conversationTitle : undefined}>{hasConversation ? conversationTitle : "새 대화"}</h1>
            </div>
          </div>
          <div className={styles.headerActions}>
            {canExport && (
              <button className={styles.iconButton} type="button" aria-label="현재 대화 Markdown으로 저장" title="Markdown으로 저장" disabled={sending} onClick={exportCurrentFlight}>
                <Icon name="download" size={19} />
              </button>
            )}
            <button
              ref={telemetryButtonRef}
              className={styles.inspectorToggle}
              type="button"
              aria-controls="chat-inspector"
              aria-expanded={telemetryOpen}
              aria-label={telemetryOpen ? "관제실 닫기" : "관제실 열기"}
              title={telemetryOpen ? "관제실 닫기" : "관제실 열기"}
              onClick={() => setTelemetryPreference(!telemetryOpen)}
            >
              <Icon name="panel" size={19} />
              <span>관제실</span>
            </button>
          </div>
        </header>

        <div className={styles.viewport} ref={messageViewportRef}>
          {hasConversation ? (
            <div className={styles.transcript} role="log" aria-label="대화 내용" aria-relevant="additions">
              {conversationMessages.map((message) => (
                <article className={message.role === "user" ? styles.userMessage : styles.assistantMessage} key={message.id}>
                  <div className={styles.messageAuthor}>
                    {message.role === "assistant" && <img src="/papiti-mark.svg" width={28} height={28} alt="" />}
                    <strong>{message.role === "user" ? "나" : "파피티"}</strong>
                  </div>
                  <div className={styles.messageText}>{message.content}</div>
                  {message.meta && message.meta.length > 0 && (
                    <details className={styles.responseDetails}>
                      <summary>응답 정보</summary>
                      <ul>
                        {message.meta.map((item, index) => <li key={index} data-tone={item.tone}>{item.text}</li>)}
                      </ul>
                    </details>
                  )}
                  {message.role === "assistant" && (
                    <div className={styles.messageActions}>
                      <button className={styles.iconButton} type="button" aria-label="답변 복사" title="답변 복사" onClick={() => void navigator.clipboard?.writeText(message.content).catch(() => undefined)}>
                        <Icon name="copy" size={16} />
                      </button>
                      {router && message.id === conversationMessages[conversationMessages.length - 1]?.id && (
                        <button className={styles.iconButton} type="button" aria-label="최근 라우터 기록 보기" title="최근 라우터 기록 보기" onClick={showLatestTrace}>
                          <Icon name="activity" size={16} />
                        </button>
                      )}
                      {message.learning && (
                        <button
                          className={styles.iconButton}
                          type="button"
                          aria-label={message.learning.confirmed ? "학습 완료" : "이 경로가 맞다고 학습시키기"}
                          title={message.learning.confirmed ? "학습 완료" : "이 경로가 맞다고 학습시키기"}
                          onClick={() => void confirmLearning(message.id, message.learning)}
                          disabled={message.learning.confirmed}
                        ><Icon name="thumb" size={16} /></button>
                      )}
                    </div>
                  )}
                </article>
              ))}
              {sending && (
                <div className={styles.thinking}>
                  <FlyThinkingIndicator phase={monitorPhase} frame={liveFrames[liveFrames.length - 1] ?? null} />
                </div>
              )}
            </div>
          ) : (
            <section className={styles.welcome} aria-labelledby="welcome-title">
              <div className={styles.welcomeHero}>
                <div className={styles.welcomeLogo} aria-hidden="true">
                  <img src="/papiti-mark.svg" width={53} height={53} alt="" />
                </div>
                <h2 id="welcome-title">오늘은 뭐가<br /><em>궁금해?</em></h2>
                <p>파피티에게 질문하거나 아래 예시로 대화를 시작해 봐.</p>
              </div>
              <div className={styles.starters} role="group" aria-label="대화 시작 예시">
                {STARTERS.map((item) => (
                  <button className={styles.starter} type="button" key={item.title} onClick={() => {
                    setInput(item.prompt);
                    composerRef.current?.focus();
                  }}>
                    <span className={styles.starterIcon}><Icon name={item.icon} size={18} /></span>
                    <span><strong>{item.title}</strong><small>{item.prompt}</small></span>
                    <Icon name="diagonal" size={16} />
                  </button>
                ))}
              </div>
            </section>
          )}
        </div>

        <footer className={styles.footer}>
          <form className={styles.composer} onSubmit={handleSubmit}>
            <textarea
              ref={composerRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey &&
                    !event.nativeEvent.isComposing && event.keyCode !== 229) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="파피티에게 메시지 보내기"
              rows={1}
              disabled={sending}
              aria-label="파피티에게 메시지 보내기"
            />
            <div className={styles.composerActions}>
              <span className={styles.modelStatus} role="status" data-state={generatorReadiness}><i />{generatorLabel}</span>
              <button className={styles.send} type="submit" disabled={sending || !input.trim() || !sessionId} aria-label="메시지 보내기">
                <Icon name="send" size={21} />
              </button>
            </div>
          </form>
          <div className={styles.footerMeta}>
            <span className={styles.keyboardHint}><kbd>Shift</kbd> + <kbd>Enter</kbd> 줄바꿈</span>
            <span className={styles.connection} role="status" data-state={online ? "online" : healthError ? "offline" : "checking"}><i />{connectionLabel}</span>
          </div>
        </footer>
      </section>

      {telemetryOpen && (
        <aside
          id="chat-inspector"
          ref={telemetryRef}
          className={styles.inspector}
          aria-label="신경 관제실"
          tabIndex={-1}
          inert={activeDrawer === "history"}
        >
          <header className={styles.inspectorHead}>
            <div><Icon name="activity" size={19} /><h2>신경 관제실</h2></div>
            <button className={styles.iconButton} type="button" aria-label="신경 관제실 닫기" onClick={() => {
              setTelemetryPreference(false);
              if (wideViewport) telemetryButtonRef.current?.focus();
            }}><Icon name="close" size={19} /></button>
          </header>
          <div className={styles.inspectorBody}>
            <FullPopulationMap
              phase={monitorPhase}
              pending={sending}
              routerNodes={traceRouter?.n_nodes ?? router?.n_nodes ?? null}
              fullPopulationActive={Boolean(health?.papiti_v05?.enabled || health?.full_population_router?.active)}
            />
            <button className={styles.graphButton} type="button" onClick={openBrain}>
              <Icon name="brain" size={18} /> 신경 지도 열기 <Icon name="diagonal" size={16} />
            </button>
            <details
              className={styles.inspectorDetails}
              ref={traceDetailsRef}
              tabIndex={-1}
              open={traceExpanded}
              onToggle={(event) => setTraceExpanded(event.currentTarget.open)}
            >
              <summary>최근 라우터 기록</summary>
              {traceExpanded && <ConnectomePreview router={traceRouter} pending={sending} live={sending} phase={monitorPhase} onExpand={openBrain} />}
              {traceExpanded && topRoutes.length > 0 && (
                <dl className={styles.routeMix}>
                  {topRoutes.map((item) => (
                    <div key={item.route}>
                      <dt>{item.route}</dt>
                      <dd>{(item.confidence * 100).toFixed(1)}%</dd>
                      <span aria-hidden="true"><i style={{ width: item.confidence * 100 + "%" }} /></span>
                    </div>
                  ))}
                </dl>
              )}
            </details>
            <details className={styles.inspectorDetails}>
              <summary>시스템 정보</summary>
              <dl className={styles.systemRows}>
                <div><dt>데이터셋</dt><dd>{datasetLabel}</dd></div>
                <div><dt>라우터</dt><dd>{modelLabel}</dd></div>
                <div><dt>생성기</dt><dd>{health?.generator?.model ?? generatorLabel}</dd></div>
                <div><dt>백엔드</dt><dd>{health?.app_version ? "v" + health.app_version : connectionLabel}</dd></div>
                <div><dt>메모리</dt><dd>{health?.memory?.enabled ? "활성" : "대기"}</dd></div>
              </dl>
            </details>
          </div>
        </aside>
      )}

      {activeDrawer && (
        <button
          className={styles.scrim}
          type="button"
          tabIndex={-1}
          aria-label={activeDrawer === "history" ? "대화 메뉴 닫기" : "관제실 닫기"}
          onClick={() => activeDrawer === "history" ? setRailOpen(false) : setTelemetryPreference(false)}
        />
      )}

      <BrainPanel open={brainOpen} onClose={() => setBrainOpen(false)} router={traceRouter} phase={monitorPhase} requestId={traceRequestId} />

      <dialog ref={settingsRef} className={styles.settings} aria-labelledby="chat-settings-title" onClick={(event) => {
        if (event.target === event.currentTarget) event.currentTarget.close();
      }}>
        <div className={styles.settingsInner}>
          <header>
            <h2 id="chat-settings-title">설정</h2>
            <button className={styles.iconButton} type="button" aria-label="설정 닫기" onClick={() => settingsRef.current?.close()}><Icon name="close" /></button>
          </header>
          <h3>강조 색상</h3>
          <div className={styles.swatches} role="group" aria-label="채팅 강조 색상">
            {THEMES.map((item) => (
              <button
                key={item.id}
                type="button"
                aria-pressed={themeId === item.id}
                onClick={() => setThemeId(item.id)}
              ><i style={{ backgroundColor: item.accent2 }} />{item.name}</button>
            ))}
          </div>
          <h3>세션 및 학습</h3>
          <dl className={styles.systemRows}>
            <div><dt>연결</dt><dd>{connectionLabel}</dd></div>
            <div><dt>라우터</dt><dd>{modelLabel}</dd></div>
            <div><dt>세션</dt><dd>{sessionId ? sessionId.slice(0, 12) + "…" : "준비 중"}</dd></div>
            <div><dt>경로 학습</dt><dd>{learningStatus ? learningStatus.examples + "개 예시" : health?.adaptive_learning?.enabled ? "활성" : "대기"}</dd></div>
            <div><dt>장기 지식</dt><dd>{knowledgeStatus ? knowledgeStatus.items + "개 · " + knowledgeStatus.verified + "개 검증" : health?.knowledge?.enabled ? "활성" : "대기"}</dd></div>
            <div><dt>웹 검색</dt><dd>{health?.research?.configured ? health.research.provider ?? "활성" : "API 키 필요"}</dd></div>
          </dl>
          <div className={styles.resetActions}>
            <button type="button" onClick={resetMemory} disabled={sending}>현재 대화 기억 지우기</button>
            <button type="button" onClick={() => void resetRouteLearning()} disabled={sending}>경로 학습 지우기</button>
            <button type="button" onClick={() => void resetKnowledge()} disabled={sending}>장기 지식 지우기</button>
          </div>
        </div>
      </dialog>
    </main>
  );
}
