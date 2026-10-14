import type { ChatResponse } from "./api";
import type { MessageMeta } from "./chat-history";

export function formatRouterModel(raw?: string | null) {
  if (!raw) return "Router pending";

  const filename = raw.split("/").pop() || raw;
  let version = filename
    .replace(/^fly_router_/, "")
    .replace(/\.pt$/, "");

  if (version.startsWith("v")) version = version.slice(1);
  version = version.replaceAll("_", ".");

  return version ? "Router v" + version : "Router";
}

export function buildMessageMeta(response: ChatResponse): MessageMeta[] {
      const meta: MessageMeta[] = [];
      if (response.type === "logic_verified") {
        meta.push({ text: "VERIFIED · ALL ARRANGEMENTS", tone: "good" });
      }
      if (response.router) {
        if (response.router.semantic_override) {
          meta.push({
            text: response.router.route + " · semantic rule",
            tone: "good",
          });
          meta.push({
            text:
              "MODEL " +
              response.router.semantic_override.model_route +
              " " +
              (response.router.semantic_override.model_confidence * 100).toFixed(1) +
              "%",
            tone: "muted",
          });
        } else {
          meta.push({
            text:
              response.router.route +
              " · " +
              (response.router.confidence * 100).toFixed(1) +
              "%",
            tone: response.router.confidence >= 0.55 ? "good" : "warn",
          });
        }

        if (response.router.model) {
          meta.push({
            text: formatRouterModel(response.router.model),
            tone: "muted",
          });
        }
      }

      const generation = response.data?.generation;
      if (generation?.used) {
        meta.push({
          text: "GEN · " + (generation.model ?? generation.provider ?? "ready"),
          tone: "info",
        });
        if ((generation.attempts ?? 0) > 1) {
          meta.push({
            text: "RECOVERED · " + generation.attempts + " attempts",
            tone: "good",
          });
        }

        if (generation.latency_ms != null) {
          meta.push({
            text: "GEN " + generation.latency_ms + " ms",
            tone: "muted",
          });
        }
      } else if (generation && !generation.used) {
        meta.push({
          text: "GEN · " + (generation.error_code ?? "unavailable"),
          tone: "warn",
        });
      }

      const timings = response.timings ?? response.data?.timings;
      if (timings?.query_ms != null) {
        meta.push({
          text: "QUERY " + timings.query_ms + " ms",
          tone: "info",
        });
      }
      if (timings?.router_ms != null) {
        meta.push({
          text: "ROUTE " + timings.router_ms + " ms",
          tone: "muted",
        });
      }
      if (timings?.dispatch_ms != null) {
        meta.push({
          text: "DISPATCH " + timings.dispatch_ms + " ms",
          tone: "muted",
        });
      }
      if (timings?.total_ms != null) {
        meta.push({
          text: "TOTAL " + timings.total_ms + " ms",
          tone: timings.total_ms >= 10000 ? "warn" : "muted",
        });
      }

      if (response.router?.learning_observed) {
        meta.push({ text: "LEARNED", tone: "info" });
      } else if (response.router?.personalization?.applied) {
        meta.push({
          text:
            "PERSONAL " +
            Math.round((response.router.personalization.blend ?? 0) * 100) +
            "%",
          tone: "info",
        });
      }

      if ((response.data?.knowledge_learned?.length ?? 0) > 0) {
        meta.push({
          text: "KNOWLEDGE +" + (response.data?.knowledge_learned?.length ?? 0),
          tone: "info",
        });
      }

      const research = response.data?.research;
      if ((research?.sources?.length ?? 0) > 0) {
        meta.push({
          text: "WEB · " + research!.sources.length + " sources",
          tone: "info",
        });
      } else if (response.router?.route === "research" && research?.error) {
        meta.push({ text: "WEB unavailable", tone: "warn" });
      }

      const verifiedCount = response.data?.web_verified_facts?.length ?? 0;
      if (verifiedCount > 0) {
        meta.push({ text: "VERIFIED +" + verifiedCount, tone: "good" });
      }

  return meta;
}
