export type MessageMeta = {
  text: string;
  tone?: "good" | "info" | "warn" | "muted";
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  meta?: MessageMeta[];
  retryPrompt?: string;
  learning?: {
    prompt: string;
    route: string;
    confirmed?: boolean;
  };
};

export type ChatRecord = {
  id: string;
  title: string;
  sessionId: string;
  messages: Message[];
  updatedAt: number;
};

export const INITIAL_MESSAGE: Message = {
  id: "welcome",
  role: "assistant",
  content:
    "안녕하세요. FlyGPT MaleCNS 인터페이스 준비 완료 🪰\n질문을 보내면 실제 Drosophila MaleCNS 배선에서 만든 라우터 scaffold를 따라 경로를 선택해요.",
};

export function makeId(prefix: string) {
  return prefix + "-" + Date.now() + "-" + Math.random().toString(16).slice(2);
}

export function titleFrom(text: string) {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return "Untitled flight";
  return clean.length > 28 ? clean.slice(0, 28) + "…" : clean;
}

export const MAX_STORED_CHATS = 40;
const CHAT_HISTORY_KEY_PREFIX = "flygpt-chat-history-v0.7";
const ACTIVE_CHAT_KEY_PREFIX = "flygpt-active-chat-v0.7";

export function activeChatKey(userId: string): string {
  return ACTIVE_CHAT_KEY_PREFIX + ":" + userId;
}
function chatHistoryKey(userId: string): string {
  return CHAT_HISTORY_KEY_PREFIX + ":" + userId;
}

function isMessage(input: unknown): input is Message {
  if (!input || typeof input !== "object") return false;
  const item = input as Partial<Message>;
  if (typeof item.id !== "string" || (item.role !== "user" && item.role !== "assistant") || typeof item.content !== "string") return false;
  if (item.retryPrompt !== undefined && typeof item.retryPrompt !== "string") return false;
  if (item.meta !== undefined && (!Array.isArray(item.meta) || !item.meta.every(meta => meta && typeof meta.text === "string"))) return false;
  if (item.learning !== undefined && (!item.learning || typeof item.learning.prompt !== "string" || typeof item.learning.route !== "string")) return false;
  return true;
}

export function loadChatHistory(userId: string): ChatRecord[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(chatHistoryKey(userId));
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item): item is ChatRecord => {
      if (!item || typeof item !== "object") return false;
      const chat = item as Partial<ChatRecord>;
      return typeof chat.id === "string" && typeof chat.title === "string" &&
        typeof chat.sessionId === "string" && typeof chat.updatedAt === "number" &&
        Number.isFinite(chat.updatedAt) && Array.isArray(chat.messages) &&
        chat.messages.every(isMessage);
    }).slice(0, MAX_STORED_CHATS);
  } catch {
    // Bad/disabled local storage should not prevent the workspace from loading.
    return [];
  }
}

export function writeChatHistory(userId: string, history: ChatRecord[]): void {
  try {
    window.localStorage.setItem(chatHistoryKey(userId), JSON.stringify(history.slice(0, MAX_STORED_CHATS)));
  } catch {
    // Persistence is best-effort (private mode/quota limits).
  }
}

export function formatChatTime(updatedAt: number): string {
  try {
    return new Intl.DateTimeFormat("ko-KR", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(updatedAt));
  } catch {
    return "";
  }
}
