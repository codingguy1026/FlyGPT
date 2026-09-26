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
  };
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
