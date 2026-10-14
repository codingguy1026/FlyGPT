/** Shared JSON transport for chat, health and auth endpoints. */
export async function requestJson<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const response = await fetch(input, { credentials: "same-origin", ...init });
  let payload: unknown;

  try {
    payload = await response.json();
  } catch {
    throw new Error(`HTTP ${response.status}: 서버 응답을 읽을 수 없습니다.`);
  }

  if (!response.ok) {
    const detail = payload && typeof payload === "object" && "detail" in payload
      ? (payload as { detail: unknown }).detail
      : null;
    const message = typeof detail === "string" ? detail
      : detail == null ? `HTTP ${response.status}` : JSON.stringify(detail);
    throw new Error(message);
  }

  return payload as T;
}
