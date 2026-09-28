const SESSION_KEY = "flygpt-memory-session-v0.5";

function makeSessionId(): string {
  return typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : "session-" + Date.now() + "-" + Math.random().toString(16).slice(2);
}

export function getMemorySessionId(): string {
  if (typeof window === "undefined") {
    return "";
  }

  try {
    const existing = window.localStorage.getItem(SESSION_KEY);
    if (existing) {
      return existing;
    }

    const value = makeSessionId();
    window.localStorage.setItem(SESSION_KEY, value);
    return value;
  } catch {
    return "ephemeral-" + Date.now() + "-" + Math.random().toString(16).slice(2);
  }
}

export function createMemorySessionId(): string {
  if (typeof window === "undefined") {
    return "";
  }

  try {
    const value = makeSessionId();
    window.localStorage.setItem(SESSION_KEY, value);
    return value;
  } catch {
    return "ephemeral-" + Date.now() + "-" + Math.random().toString(16).slice(2);
  }
}


export function setMemorySessionId(sessionId: string): void {
  if (typeof window === "undefined" || !sessionId) {
    return;
  }

  try {
    window.localStorage.setItem(SESSION_KEY, sessionId);
  } catch {
    // Session persistence is best-effort.
  }
}
