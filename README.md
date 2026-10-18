# llm-translate

Zero-dependency Go library for bidirectional protocol translation between the Anthropic Messages API, OpenAI Chat Completions, and the OpenAI Responses API.

## Features

- **Anthropic ↔ OpenAI Request & Response:** Translation of messages, tool definitions, tool results, system prompts and vision content.
- **SSE Stream Translation:** Real-time translation between OpenAI chunk deltas and Anthropic SSE events (`message_start`, `content_block_start`, `content_block_delta`, `message_delta`, `message_stop`).
- **OpenAI Responses ↔ Chat:** Bidirectional translation for newer OpenAI Responses API backends to standard Chat completions.
- **Loss reports:** every `*WithReport` conversion returns a machine-readable list of what the target surface could not represent.
- **Typed wire codecs:** package `wire` decodes and re-encodes all three surfaces without losing unknown or vendor fields.
- **Pure Standard Library:** Zero CGO, zero external dependencies.

## Installation

```bash
go get github.com/kayiik/llm-translate
```

## Quick Start

### Convert Anthropic Messages Request to OpenAI

```go
import "github.com/kayiik/llm-translate"

openaiMessages := translate.AnthropicMessagesToOpenAI(req.Messages, req.System)
tools := translate.AnthropicToolsToOpenAI(req.Tools)
```

### Stream OpenAI Chunks as Anthropic SSE

```go
translate.OpenAIStreamToAnthropicSSE(chunksIter, "claude-3-5-sonnet", func(sseEvent string) {
    w.Write([]byte(sseEvent))
    w.(http.Flusher).Flush()
})
```

### Stream Anthropic SSE as OpenAI Chunks

`AnthropicSSEToOpenAIChunks` emits no usage and ends every stream with a
finish chunk. `AnthropicSSEToOpenAIChunksWithOptions` emits the same chunks
for a stream that completes, and returns a `*StreamReport` once `lines` is
exhausted:

```go
report := translate.AnthropicSSEToOpenAIChunksWithOptions(lines, model, emit,
    translate.StreamOptions{IncludeUsage: true})
var upstream *translate.StreamError
switch {
case report.Completed:
    // message_stop arrived and no error event did
case errors.As(report.Err, &upstream):
    // an error event: upstream.Type, upstream.Message
case errors.Is(report.Err, translate.ErrStreamTruncated):
    // the source ended before message_stop
}
```

- `report.Usage` holds the input, cache creation, cache read and output
  tokens the source reported, also for a failed stream. `PromptTokens()` sums
  the three input counts, as Chat's `prompt_tokens` does.
- `IncludeUsage` adds the chunk OpenAI sends for
  `stream_options.include_usage`: `"choices": []` with `prompt_tokens`,
  `completion_tokens`, `total_tokens` and
  `prompt_tokens_details.cached_tokens` (the cache reads). Every other chunk
  then carries `"usage": null`. The chunk is sent only for a completed stream
  whose source reported usage.
- A stream that is not `Completed` ends where its source did, without a
  finish or usage chunk, so a failure cannot pass for a complete answer.

## Loss reports

Each `*WithReport` function returns the converted value with a `Report`. A
`Loss` names the source value that changed with a dot-separated path into the
source payload (numeric segments index arrays), a class (`unsupported`,
`dropped`, `approximated`, `renamed`) and a severity:

- `material`: the conversion changed or omitted meaning.
  `RejectMaterialLoss` turns these into an error.
- `advisory`: the answer is intact. A product that depends on the value can
  still reject it by path, for example with a policy rule on `**.reasoning`.

```go
result := translate.ChatToResponsesWithReport(model, messages, options, false)
if err := result.RejectMaterialLoss(); err != nil {
    return err
}
```

### Vendor fields inside messages

Conversions carry a vendor field when the target can represent it, and report
it at its exact path when they cannot:

| Field | Carried | Otherwise reported as |
|---|---|---|
| Chat tool call `extra_content.google.thought_signature` (Gemini) | never; no other surface has a place for it | `material`, e.g. `messages.3.tool_calls.0.extra_content.google.thought_signature` |
| Chat tool call `function.thought_signature` or call-level `thought_signature` (other Gemini providers) | never | `material`, e.g. `messages.3.tool_calls.0.function.thought_signature`, `messages.3.tool_calls.0.thought_signature` |
| Chat message `reasoning_details` | never | `advisory`, e.g. `messages.2.reasoning_details` |
| Chat message `reasoning_content` | never | `advisory`, e.g. `messages.2.reasoning_content` |
| Chat content part `cache_control` | to Anthropic, on text parts | `advisory`, e.g. `messages.0.content.1.cache_control` |
| Anthropic block `cache_control` | never; Chat has no breakpoints | `material` (`unsupported`) by `AnthropicRequestToOpenAIWithReport`, e.g. `system.0.cache_control` |

A dropped thought signature is material because Gemini needs it back on the
next turn of a tool call. Each location that carries one is reported, and
empty values carry nothing, so they are never reported. Response and stream
conversions use the same field
paths under `choices.N.message` and `choices.N.delta`. In a stream the index
after `tool_calls` is the call's `index` field, which identifies it across
chunks.

A dropped reasoning item or thinking block keeps its own path with a
`.reasoning` suffix, such as `output.2.reasoning`, `content.0.reasoning` or
`messages.1.content.0.reasoning`, so one rule matches reasoning on every
surface. These losses are `advisory`, and they come in addition to whatever
the conversion already reported for the item.

### cache_control from Chat to Anthropic

`OpenAIMessagesToAnthropic` and its WithReport form turn user, assistant and
tool content that has a `cache_control` text part into Anthropic text blocks.
Text up to and including each marked part becomes one block carrying that
`cache_control`, joined exactly as before, so the cached prefix is the one
the Chat request marked. Content without `cache_control` converts as before.

The system prompt is returned as a string, so only
`OpenAIMessagesToAnthropicWithReport` can carry system breakpoints: it sets
`AnthropicMessages.SystemBlocks`, which should be sent as the request's
`system` value instead of `System`. `cache_control` on a part the conversion
drops, such as an image, is reported as `advisory`.

### Structured output between Chat and Responses

A Chat `response_format` and a Responses `text.format` ask for the same
structure, so `ChatToResponses` carries one as the other and
`ResponsesRequestToChat` carries it back. A `json_schema` format's `name`,
`description`, `schema` and `strict` sit in its `json_schema` object in Chat
and directly in the format in Responses. A format of type `text` asks for the
default and is not emitted. The rename is reported as `advisory`. A format
with a member neither API defines is reported as `material`, and so is a
Responses `text` member other than `format`, such as `verbosity`.

The adapting assessment checks the finite format/function envelopes: required
format name and object schema, typed description/strict controls, and precise
unknown member paths. Schema contents themselves remain verbatim JSON data,
including refs and custom vocabulary; this does not certify a target model's
schema support. Empty object schemas and explicit `strict:false` are retained.

Function strict defaults differ: Chat→Responses makes Chat's omitted/null
non-strict default explicit as `strict:false`. Missing Chat parameters use the
required nullable Responses field; explicit native Responses null parameters
omit optional Chat parameters on return. A supplied Chat parameter schema must
be an object. Responses' omitted/null function strict policy cannot be ported
to Chat's default without guessing; `AssessConversion` marks its exact path
unknown until true/false is supplied or a native Responses target is used.
Legacy value-producing APIs remain available; admission must inspect the
assessment's feasibility and unknown paths, not merely a nonempty result.

Chat/Responses conversion retains image `detail` and responses containing both
text and function calls, including when rendering buffered Chat chunks. Content
parts a conversion cannot represent are reported at their source paths. A
foreign user part passed through unchanged is still reported as unsupported;
passthrough alone does not establish that the target accepts its type.

Usage conversion between Chat and Responses retains supplied `cached_tokens`,
`cache_write_tokens` and `reasoning_tokens` detail counters. Responses→Chat omits
missing/null optional counters and retains explicitly reported zero.
Chat→Responses preserves the legacy required detail objects and cached/reasoning
fields, using synthetic zero placeholders when absent from the source. Those
placeholders are wire compatibility values, not provider-reported measurements;
source-presence observations use the source-side APIs below. Cache-write detail
is carried when supplied. Valid-source aggregate aliases and the legacy
absent/zero-total fallback are preserved.

Chat→Responses aggregate usage accepts exact `json.Number` integers without
floating-point conversion, including int64-sized values. Invalid supplied
aggregate counts or an overflowing derived total leave target usage unknown
(`null`); WithReport identifies the material source path instead of fabricating
measured zeros. Legacy valid-value aliases, fallback and required placeholders
remain separate from source observation provenance.

## Fidelity and source observations

`AssessConversion` is an additive assessment of the four named request edges.
Pass the full **source** payload, `PhaseRequest` or `PhaseResponse`, and explicit
`FidelityRequirements`. The conversion ID always names the originating request
direction: `ConversionChatToResponses` with `PhaseResponse` assesses a Responses
answer converted back to Chat. The result distinguishes evaluated features,
known losses, and `UnknownPaths`; a loss-free legacy helper alone is not proof
of complete fidelity. Model/endpoint capabilities and admission remain the
caller's responsibility. Native passthrough does not need a conversion and
keeps its unknown fields.

Opaque encrypted/signed continuation state infers replay requirements even with
an empty visible summary. Display-only reasoning retains advisory semantics.
Unknown nested controls cannot become supported merely because their enclosing
message/tool is supported. Schema envelopes that are carried unchanged do not
establish target-model schema-vocabulary support.

`ObserveUsage` extracts a native usage object **before** conversion placeholders.
`CountObservation` distinguishes absent, invalid, measured zero, reported and
derived values. `ObserveUsageWithSource(..., UsageSynthetic)` labels observations
when only a synthesized target object is available. OpenAI cache/read/write and
reasoning are subsets of inclusive totals; Anthropic input categories are
disjoint and TTL children partition writes. Incomplete source categories remain
unknown rather than being filled with measured zeros. `UsageAccumulator` merges
cumulative updates by replacement, retaining omitted counters independently of
bounded raw evidence (up to 32 members, 4 KiB per member, 16 KiB content total).

`ObserveUsageWithDetails` / `ObserveUsageWithSourceAndDetails` return a
`UsageObservationWithDetails` wrapper, preserving the existing observation and
integer-usage layouts. Its `Details` adds Chat prompt/completion audio and
accepted/rejected prediction tokens, plus Messages server web-search/web-fetch
request counts where reported. `UsageAccumulator.SnapshotWithDetails` retains
these counters independently of truncated raw evidence. Audio/prediction values
are subsets of inclusive totals, not additions; web requests are operation
counts, not tokens. Fetch is documented by the selected pinned beta schema,
not a guarantee for every stable deployment/model. No unsupported Responses
counter counterparts, wire detail fields, prices or cost estimates are invented.

`AnthropicSSEToOpenAIChunksObserved` and `OpenAIStreamToAnthropicSSEObserved` combine
event-time assessments, source usage and terminal quality without buffering a
whole stream. The observer runs before affected source events are converted and
at finalization. The Chat input iterator must include `[DONE]`; missing terminal
evidence is truncation, not success. Requested final Chat usage is emitted only
when inclusive source totals can be determined. The legacy stream functions keep
their existing behavior. Caller transport framing, cancellation, delivery state
and no-replay-after-output enforcement remain outside these pure converters.

`OpenAIStreamToResponsesSSEObserved` incrementally renders Chat JSON chunks plus
`[DONE]` as typed Responses SSE records, including text, refusal and interleaved
client function calls. Its request argument supplies only supported response
envelope fields; generated response/item IDs do not create stored lineage.
`ResponsesSSEToOpenAIChunksObserved` accepts native Responses JSON event data and
emits Chat JSON chunks; its caller adds `[DONE]` only after a successful terminal
result and delivery checks. Required state, hosted tools, citations and unassessed
semantics are reported before affected output and stop these bridges. Source
usage remains distinct from required wire placeholders, and terminal snapshots
do not duplicate already emitted deltas. Both emit during source consumption;
per-item text/arguments are assembled for done/terminal objects, without buffering
the source generation before delivery. Existing buffered helpers remain unchanged.

## Package wire

`github.com/kayiik/llm-translate/wire` holds typed request, response and
stream structs for the three surfaces:

| Surface | Request | Response | Stream |
|---|---|---|---|
| Chat Completions | `ChatRequest` | `ChatResponse` | `ChatChunk` |
| Anthropic Messages | `AnthropicRequest` | `AnthropicResponse` | `AnthropicEvent` |
| OpenAI Responses | `ResponsesRequest` | `ResponsesResponse` | `ResponsesEvent` |

Each has a `Decode…` function, such as `wire.DecodeChatRequest`, and encodes
with `json.Marshal`. Stream types hold the JSON data of one SSE event, without
the `data:` framing.

```go
req, err := wire.DecodeChatRequest(body)
if err != nil {
    return err
}
for _, call := range req.Messages[len(req.Messages)-1].ToolCalls {
    log.Println(call.ID, call.ThoughtSignature())
}
out, err := json.Marshal(req) // equal to body as JSON
```

### Round-trip guarantee

Decoding and then encoding reproduces the input as JSON: both parse to the
same value, with numbers compared by their literal text. Member order and
insignificant whitespace are not kept.

- Every modeled object keeps the members it does not model in `Extra` and
  writes them back.
- A member whose value does not have the modeled shape, such as `null`, an
  empty identifier, or a number where a string is expected, stays in `Extra`
  verbatim rather than failing the decode. When a typed field is set, it wins
  over an `Extra` entry of the same name.
- Polymorphic members keep their form: content as a string or parts,
  `tool_choice` as a string or an object, `stop` as a string or an array.
- Numbers are `json.Number`, so `1.0` stays `1.0`.
- Optional identifiers are strings omitted when empty. Text that may be empty
  on purpose, such as streamed `arguments`, is a `*string`, and booleans are
  `*bool`. A nil slice is omitted, while an empty non-nil slice encodes as
  `[]`.

### Vendor fields

The fields products depend on are typed rather than left in `Extra`:

- `ChatToolCall.ExtraContent.Google.ThoughtSignature` for Gemini's
  `extra_content.google.thought_signature`. The `ChatToolCall.ThoughtSignature()`
  helper returns the first non-empty signature from that field, then
  `function.thought_signature`, then a call-level `thought_signature`; the
  last two stay in `Extra`.
- `ChatMessage.ReasoningDetails`, the raw `reasoning_details` array.
- `CacheControl` on `ChatContentPart`, `AnthropicBlock` (content, system and
  nested tool_result blocks) and `AnthropicTool`.

## License

MIT
