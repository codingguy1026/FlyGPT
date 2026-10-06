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
  n_nodes?: number;
  steps?: number;
  scaffold_kind?: string;
  vectorizer_version?: string;
  margin?: number;
  accepted?: boolean;
  learning_observed?: boolean;
  personalization?: {
    applied?: boolean;
    examples_considered?: number;
    neighbors_used?: number;
    blend?: number;
    nearest_similarity?: number;
  };
};

export type GenerationResult = {
  used?: boolean;
  provider?: string;
  model?: string | null;
  answer?: string | null;
  error?: string | null;
  finish_reason?: string | null;
  latency_ms?: number | null;
};

export type ChatTimings = {
  query_ms?: number | null;
  router_ms?: number | null;
  dispatch_ms?: number | null;
  generation_ms?: number | null;
  total_ms?: number | null;
};

export type ChatData = {
  generation?: GenerationResult | null;
  timings?: ChatTimings | null;
  dispatch?: unknown;
  memory_hits?: unknown;
  knowledge_hits?: unknown[];
  knowledge_learned?: unknown[];
  ui_meta?: {
    mode?: string;
    router_model?: string;
    app_version?: string;
  };
};

export type ChatResponse = {
  answer: string;
  type: string;
  data?: ChatData | null;
  timings?: ChatTimings | null;
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
  connectome?: {
    provider?: string;
    dataset?: string;
    configured?: boolean;
    initialized?: boolean;
  };
  generator?: {
    configured?: boolean;
    provider?: string;
    model?: string | null;
  };
  memory?: {
    enabled?: boolean;
    max_messages_per_session?: number;
  };
  adaptive_learning?: {
    enabled?: boolean;
    account_scoped?: boolean;
    stores_raw_prompts?: boolean;
  };
  knowledge?: {
    enabled?: boolean;
    account_scoped?: boolean;
    provenance_aware?: boolean;
  };
};

export type KnowledgeStatus = {
  enabled: boolean;
  items: number;
  verified: number;
  asserted: number;
  unverified: number;
  superseded: number;
  last_updated_at?: number | null;
};

export type RouteLearningStatus = {
  enabled: boolean;
  examples: number;
  auto_examples: number;
  feedback_examples: number;
  confirmations: number;
  last_learned_at?: number | null;
  routes: Record<string, number>;
  stores_raw_prompts: boolean;
  max_examples: number;
};

async function jsonRequest<T>(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(input, {
    credentials: "same-origin",
    ...init,
  });
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

export function getRouteLearningStatus(): Promise<RouteLearningStatus> {
  return jsonRequest<RouteLearningStatus>("/api/router/learning/status", {
    cache: "no-store",
  });
}

export function confirmRouteLearning(
  message: string,
  route: string,
): Promise<{
  learned: boolean;
  route: string;
  status: RouteLearningStatus;
}> {
  return jsonRequest("/api/router/learning/feedback", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message, route }),
  });
}

export function clearRouteLearning(): Promise<{
  cleared: number;
  status: RouteLearningStatus;
}> {
  return jsonRequest("/api/router/learning/clear", {
    method: "POST",
  });
}

export function getKnowledgeStatus(): Promise<KnowledgeStatus> {
  return jsonRequest<KnowledgeStatus>("/api/knowledge/status", {
    cache: "no-store",
  });
}

export function clearKnowledge(): Promise<{
  cleared: number;
  status: KnowledgeStatus;
}> {
  return jsonRequest("/api/knowledge/clear", {
    method: "POST",
  });
}
