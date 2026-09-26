const SESSION_KEY = "flygpt-memory-session-v0.5";

export function getMemorySessionId(): string {
  if (typeof window === "undefined") {
    return "";
  }

  try {
    const existing = window.localStorage.getItem(SESSION_KEY);
    if (existing) {
      return existing;
    }

    const value =
      typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;

    window.localStorage.setItem(SESSION_KEY, value);
    return value;
  } catch {
    return `ephemeral-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }
}
