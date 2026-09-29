const SESSION_KEY_PREFIX = "flygpt-memory-session-v0.7";

function sessionKey(scope: string): string {
  return SESSION_KEY_PREFIX + ":" + scope;
}

function makeSessionId(): string {
  return typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : "session-" + Date.now() + "-" + Math.random().toString(16).slice(2);
}

export function getMemorySessionId(scope: string): string {
  if (typeof window === "undefined") {
    return "";
  }

  try {
    const key = sessionKey(scope);
    const existing = window.localStorage.getItem(key);
    if (existing) {
      return existing;
    }

    const value = makeSessionId();
    window.localStorage.setItem(key, value);
    return value;
  } catch {
    return "ephemeral-" + Date.now() + "-" + Math.random().toString(16).slice(2);
  }
}

export function createMemorySessionId(scope: string): string {
  if (typeof window === "undefined") {
    return "";
  }

  try {
    const value = makeSessionId();
    window.localStorage.setItem(sessionKey(scope), value);
    return value;
  } catch {
    return "ephemeral-" + Date.now() + "-" + Math.random().toString(16).slice(2);
  }
}

export function setMemorySessionId(scope: string, sessionId: string): void {
  if (typeof window === "undefined" || !sessionId) {
    return;
  }

  try {
    window.localStorage.setItem(sessionKey(scope), sessionId);
  } catch {
    // Session persistence is best-effort.
  }
}
