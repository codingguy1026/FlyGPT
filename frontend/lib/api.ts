export type RouteScore = {
  route: string;
  confidence: number;
};

export type TraceFrame = {
  stage: string;
  activations: number[];
};

export type RouterResult = {
  model?: string;
  route: string;
  confidence: number;
  top_routes?: RouteScore[];
  trace?: TraceFrame[];
};

export type ChatResponse = {
  answer: string;
  type: string;
  data: unknown;
  router?: RouterResult;
};

export type GraphNode = {
  root_id?: string;
};

export type GraphEdge = {
  src: number;
  dst: number;
  weight?: number;
};

export type GraphResponse = {
  n_nodes: number;
  n_edges: number;
  steps: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type HealthResponse = {
  status: string;
  app_version?: string;
  model_path?: string;
  model_available?: boolean;
  router_initialized?: boolean;
  generator?: {
    configured?: boolean;
    provider?: string;
    model?: string | null;
  };
  memory?: {
    enabled?: boolean;
    max_messages_per_session?: number;
    capsule_batch_size?: number;
    long_term_capsules?: boolean;
  };
  streaming_enabled?: boolean;
};

export type MemoryStatus = {
  messages: number;
  capsules: number;
  last_message_at?: number | null;
  last_capsule_at?: number | null;
  capsule_batch_size?: number;
};

export type ChatStreamEvent =
  | {
      type: "route";
      router: RouterResult;
    }
  | {
      type: "brain_step";
      index: number;
      total: number;
      frame: TraceFrame;
    }
  | {
      type: "answer";
      payload: ChatResponse;
    }
  | {
      type: "done";
    }
  | {
      type: "error";
      detail: string;
    };

async function jsonRequest<T>(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(input, init);
  let payload: unknown;

  try {
    payload = await response.json();
  } catch {
    throw new Error(`HTTP ${response.status}`);
  }

  if (!response.ok) {
    const detail =
      typeof payload === "object" &&
      payload !== null &&
      "detail" in payload
        ? String((payload as { detail: unknown }).detail)
        : `HTTP ${response.status}`;
    throw new Error(detail);
  }

  return payload as T;
}

export function sendChat(
  message: string,
  sessionId: string,
): Promise<ChatResponse> {
  return jsonRequest<ChatResponse>("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      session_id: sessionId,
    }),
  });
}

export async function sendChatStream(
  message: string,
  sessionId: string,
  onEvent: (event: ChatStreamEvent) => void,
): Promise<void> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      session_id: sessionId,
    }),
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.detail) detail = String(payload.detail);
    } catch {
      // Keep HTTP status fallback.
    }
    throw new Error(detail);
  }

  if (!response.body) {
    throw new Error("stream body is unavailable");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), {
      stream: !done,
    });

    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      onEvent(JSON.parse(trimmed) as ChatStreamEvent);
    }

    if (done) break;
  }

  const tail = buffer.trim();
  if (tail) {
    onEvent(JSON.parse(tail) as ChatStreamEvent);
  }
}

export function getHealth(): Promise<HealthResponse> {
  return jsonRequest<HealthResponse>("/api/health", {
    cache: "no-store",
  });
}

export function getRouterGraph(): Promise<GraphResponse> {
  return jsonRequest<GraphResponse>("/api/router/graph", {
    cache: "no-store",
  });
}

export function clearMemory(sessionId: string): Promise<{
  cleared: number;
  session_id_present: boolean;
}> {
  return jsonRequest("/api/memory/clear", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
    }),
  });
}

export function getMemoryStatus(sessionId: string): Promise<MemoryStatus> {
  const query = new URLSearchParams({ session_id: sessionId });
  return jsonRequest<MemoryStatus>(`/api/memory/status?${query.toString()}`, {
    cache: "no-store",
  });
}
