import { requestJson } from "./http";

export type RouteScore = {
  route: string;
  confidence: number;
};

export type TraceFrame = {
  stage: string;
  activations: number[];
};

export type ChatStreamEvent =
  | { type: "status"; phase: "processing" | "router_loading" | "generation" }
  | { type: "trace"; frame: TraceFrame }
  | { type: "route"; route: string; confidence: number; accepted: boolean; semantic_override: boolean }
  | { type: "result"; response: ChatResponse }
  | { type: "error"; message: string; status?: number };

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
  semantic_override?: {
    source: string;
    speech_act: string;
    model_route: string;
    model_confidence: number;
    model_margin: number;
    model_accepted: boolean;
  };
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
  error_code?: string | null;
  attempts?: number;
  finish_reason?: string | null;
  latency_ms?: number | null;
};

export type ChatTimings = {
  query_ms?: number | null;
  router_ms?: number | null;
  dispatch_ms?: number | null;
  search_ms?: number | null;
  generation_ms?: number | null;
  verification_ms?: number | null;
  total_ms?: number | null;
};

export type ResearchSource = {
  url: string;
  title: string;
  hostname: string;
  root_domain?: string;
  authority?: number;
  age?: string[];
};

export type ResearchResult = {
  query: string;
  provider: string;
  sources: ResearchSource[];
  retrieved_at: number;
  error?: string | null;
  http_status?: number | null;
};

export type ChatData = {
  generation?: GenerationResult | null;
  timings?: ChatTimings | null;
  dispatch?: unknown;
  memory_hits?: unknown;
  knowledge_hits?: unknown[];
  knowledge_learned?: unknown[];
  research?: ResearchResult | null;
  web_verified_facts?: unknown[];
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

export type NeuronKind = "sensory" | "motor" | "interneuron" | "projection" | "other" | "unclassified";

export type NeuronAnnotation = {
  body_id: string;
  kind: NeuronKind;
  superclass: string | null;
  type: string | null;
  instance: string | null;
  consensus_nt: string | null;
};

export type NeuronAnnotationsResponse = {
  dataset: string;
  source: string;
  annotation_kind: string;
  requested: number;
  annotated: number;
  by_body_id: Record<string, NeuronAnnotation>;
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
  full_population_summary?: boolean;
  visualized_neurons?: number;
  visualization_status?: string;
  representation?: string;
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
  research?: {
    configured?: boolean;
    provider?: string;
    timeout_seconds?: number;
    count?: number;
    max_urls?: number;
    search_lang?: string;
    safesearch?: string;
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

/** Stream genuine backend inference events; never run a second router pass. */
export async function sendChatStream(
  message: string,
  sessionId: string,
  onEvent: (event: ChatStreamEvent) => void,
): Promise<ChatResponse> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ message, session_id: sessionId }),
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error("HTTP " + response.status + ": 채팅 스트림을 열지 못했어.");
  }
  if (!response.body) throw new Error("스트리밍을 지원하지 않는 응답이야.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult: ChatResponse | null = null;

  function parseBlock(block: string) {
    const json = block.split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (!json) return;
    const event = JSON.parse(json) as ChatStreamEvent;
    onEvent(event);
    if (event.type === "result") finalResult = event.response;
    if (event.type === "error") throw new Error(event.message);
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let index = buffer.indexOf("\n\n");
      while (index >= 0) {
        parseBlock(buffer.slice(0, index));
        buffer = buffer.slice(index + 2);
        index = buffer.indexOf("\n\n");
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) parseBlock(buffer);
  } finally {
    reader.releaseLock();
  }
  if (!finalResult) throw new Error("채팅 스트림이 최종 응답 없이 종료됐어.");
  return finalResult;
}

export function sendChat(
  message: string,
  sessionId: string,
): Promise<ChatResponse> {
  return requestJson<ChatResponse>("/api/chat", {
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
  return requestJson<HealthResponse>("/api/health", {
    cache: "no-store",
  });
}

export function getRouterAnnotations(): Promise<NeuronAnnotationsResponse> {
  return requestJson<NeuronAnnotationsResponse>("/api/router/annotations", {
    cache: "no-store",
  });
}

export function getRouterGraph(): Promise<GraphResponse> {
  return requestJson<GraphResponse>("/api/router/graph", {
    cache: "no-store",
  });
}

export function clearMemory(sessionId: string): Promise<{
  cleared: number;
  session_id_present: boolean;
}> {
  return requestJson("/api/memory/clear", {
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
  return requestJson<RouteLearningStatus>("/api/router/learning/status", {
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
  return requestJson("/api/router/learning/feedback", {
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
  return requestJson("/api/router/learning/clear", {
    method: "POST",
  });
}

export function getKnowledgeStatus(): Promise<KnowledgeStatus> {
  return requestJson<KnowledgeStatus>("/api/knowledge/status", {
    cache: "no-store",
  });
}

export function clearKnowledge(): Promise<{
  cleared: number;
  status: KnowledgeStatus;
}> {
  return requestJson("/api/knowledge/clear", {
    method: "POST",
  });
}
