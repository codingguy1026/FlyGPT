# Hybrid evidence-first conversation (experimental)

This changes how Hybrid decides whether to search, while preserving
MaleCNS activation, the Python reasoning tools and v0.6.1 neural speech.

## A request is not a router category

In Hybrid, the old MaleCNS general/research/memory/code prediction remains
diagnostic only. Its confidence threshold does not prevent the user from
talking. For compatibility, the existing downstream dispatcher still uses
an internal GENERAL execution identifier. It is NOT a new classifier.
Native and legacy HTTP modes retain their existing routing behavior.

A deterministic, inspectable evidence-need policy checks:

1. Did the brain already prepare a completed semantic packet?
2. Is relevant, verified account knowledge already available?
3. Is the user making a personal statement (not asking a question), asking
   about a past private conversation, or explicitly disallowing web search?
4. Otherwise, does the unresolved request ask about an external factual
   question or ask to look up fresh facts?

For an unanswered public question, Hybrid automatically calls its existing
Brave LLM Context search service if configured. It will not send a
personally disclosed achievement, private identifier, account secret, past
conversation, or simple greeting to web search. A negative MaleCNS confidence
score no longer blocks this decision.

Examples:

| Input | Decision |
|---|---|
| 오늘 야구에서 역전승했어! | No web request; conversational input |
| 흥미로운 역사 이야기를 하나 들려줘 | Search for attributable historical material, even though not a question |
| 왜 하늘이 파래? | Search when the brain lacks grounded evidence |
| 최근 대구 뉴스 알려줘 | Search for current public information |
| 내가 전에 정한 이름 기억해? | Look in private conversation memory only |
| 검색하지 마. | No web request |

## Search isn't training

The model cannot inspect its own weights to prove it hasn't learned a
particular fact. This policy searches for missing ground truth rather than
claiming to inspect neural knowledge.

Retrieved snippets are not automatically verified facts. The Python brain
builds a report_retrieved_web_evidence plan containing short, quoted
source excerpts with HTTPS URLs. The existing v0.6.1 language model may only
speak if its output faithfully preserves the exact evidence and attribution;
otherwise a labeled Native fallback shows bounded, sourced excerpts.
It must never guess an unseen answer. These excerpts are not automatically
folded into v0.6.1 training weights.

The existing independently supported verification pipeline can still store
properly validated facts; raw search hits do not meet that standard.

## Configuration / rollback

Start the usual Hybrid app:

    make hybrid

Automatic retrieval needs a server-side Brave Search API key in the
ignored .env file (do not commit the key):

    BRAVE_SEARCH_API_KEY=your-own-key

The search client uses strict safe search and the configured timeout. Without
a key, Hybrid reports unavailable live sources rather than fabricating
knowledge or silently calling a hosted LLM. This does not authorize sharing
private conversations with a search service.

Disable the new policy temporarily:

    FLYGPT_HYBRID_EVIDENCE_FIRST=0 make hybrid

Tests:

    python -m unittest tests.test_hybrid_evidence_first -v

## Remaining limitations

This release removes the confidence/category gate for Hybrid, but it
does not make v0.6.1 capable of unrestricted free-form reasoning or fluent
unsupervised conversation. The language model only speaks from ready
semantic packets. A personal win that was previously rejected by the router
will now flow through the conversation pipeline, but without a learned
social-reaction plan the speaker may still report that it cannot handle the
open-ended content. Supporting natural spontaneous congratulations requires
additional upstream meaning plans and subsequent neural speech training.
No canned congratulations or fabricated learned outputs are added here.

## Grounded storytelling requests

A user may request an external fact through an imperative rather than a
question, e.g. "흥미로운 역사 이야기를 하나 들려줘". The evidence policy now
detects invitations to recount **real historical content** and retrieves
source material rather than treating them as nonquestions.

Purely fictional creative writing ("가상의 역사 이야기를 만들어 줘") is not
sent to a public search engine. Historical anecdotes about a user's friends
or family also remain private.

**Honest boundary:** the existing v0.6.1 speaker has not been trained on
arbitrary web excerpts or long-form narratives. This version removes the
unhelpful "unsupported question" response when sourced material is available,
but its currently dependable output is **attributed excerpts**, not a newly
composed historical story. Without a configured Brave API key, it reports
missing external evidence. A later speaker-training iteration would be
needed for fluent sourced storytelling.
