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
