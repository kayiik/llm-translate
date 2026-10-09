package translate

import (
	"encoding/json"
	"errors"
	"strconv"
	"strings"
)

// sse formats one Anthropic SSE event (two lines + blank).
func sse(event string, data map[string]any) string {
	b, _ := json.Marshal(data)
	return "event: " + event + "\ndata: " + string(b) + "\n\n"
}

// streamState threads mutable state through the OpenAI->Anthropic translator.
type streamState struct {
	messageID      string
	model          string
	started        bool
	textIndex      int
	textIndexSet   bool
	textOpen       bool
	toolBlocks     map[int]map[string]any
	toolIndexByOAI map[int]int
	toolOrder      []int
	nextBlockIndex int
	stopReason     string
	inputTokens    int
	outputTokens   int
}

func newStreamState(model string) *streamState {
	return &streamState{
		messageID:      anthropicMessageID(),
		model:          model,
		stopReason:     "end_turn",
		toolBlocks:     map[int]map[string]any{},
		toolIndexByOAI: map[int]int{},
	}
}

func (s *streamState) reserveBlockIndex() int {
	i := s.nextBlockIndex
	s.nextBlockIndex++
	return i
}

func emitMessageStart(s *streamState) string {
	return sse("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": s.messageID, "type": "message", "role": "assistant",
			"content": []any{}, "model": s.model,
			"stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": s.inputTokens, "output_tokens": 0},
		},
	})
}

func openTextBlock(s *streamState) []string {
	if s.textOpen {
		return nil
	}
	if !s.textIndexSet {
		s.textIndex = s.reserveBlockIndex()
		s.textIndexSet = true
	}
	s.textOpen = true
	return []string{sse("content_block_start", map[string]any{
		"type": "content_block_start", "index": s.textIndex,
		"content_block": map[string]any{"type": "text", "text": ""},
	})}
}

func closeTextBlock(s *streamState) []string {
	if !s.textOpen || !s.textIndexSet {
		return nil
	}
	s.textOpen = false
	return []string{sse("content_block_stop", map[string]any{
		"type": "content_block_stop", "index": s.textIndex,
	})}
}

func openToolBlock(s *streamState, oaiIndex int, toolID, name string) []string {
	if _, ok := s.toolIndexByOAI[oaiIndex]; ok {
		return nil
	}
	blockIndex := s.reserveBlockIndex()
	s.toolIndexByOAI[oaiIndex] = blockIndex
	s.toolBlocks[blockIndex] = map[string]any{"id": toolID, "name": name, "args": ""}
	s.toolOrder = append(s.toolOrder, blockIndex)
	return []string{sse("content_block_start", map[string]any{
		"type": "content_block_start", "index": blockIndex,
		"content_block": map[string]any{"type": "tool_use", "id": toolID, "name": name, "input": map[string]any{}},
	})}
}

func emitToolPartial(s *streamState, oaiIndex int, partial string) []string {
	blockIndex, ok := s.toolIndexByOAI[oaiIndex]
	if !ok {
		return nil
	}
	s.toolBlocks[blockIndex]["args"] = s.toolBlocks[blockIndex]["args"].(string) + partial
	return []string{sse("content_block_delta", map[string]any{
		"type": "content_block_delta", "index": blockIndex,
		"delta": map[string]any{"type": "input_json_delta", "partial_json": partial},
	})}
}

func closeAllToolBlocks(s *streamState) []string {
	var out []string
	for _, blockIndex := range s.toolOrder {
		out = append(out, sse("content_block_stop", map[string]any{
			"type": "content_block_stop", "index": blockIndex,
		}))
	}
	return out
}

// OpenAIStreamToAnthropicSSE translates OpenAI SSE chunk payloads (JSON strings,
// data: prefix already stripped) into Anthropic SSE event blocks, invoking emit
// for each output event.
func OpenAIStreamToAnthropicSSE(chunks func() (string, bool), model string, emit func(string)) {
	s := newStreamState(model)
	sawAnyDelta := false

	for {
		raw, ok := chunks()
		if !ok {
			break
		}
		if raw == "" {
			continue
		}
		var chunk map[string]any
		if json.Unmarshal([]byte(raw), &chunk) != nil {
			continue
		}
		if um, ok := asMap(chunk["usage"]); ok {
			if v := toInt(um["prompt_tokens"]); v != 0 {
				s.inputTokens = v
			}
			if v := toInt(um["completion_tokens"]); v != 0 {
				s.outputTokens = v
			}
		}
		choices, _ := asList(chunk["choices"])
		for _, ch := range choices {
			choice, ok := asMap(ch)
			if !ok {
				continue
			}
			delta, _ := asMap(choice["delta"])
			if delta == nil {
				delta = map[string]any{}
			}
			finishReason := choice["finish_reason"]

			if !s.started {
				s.started = true
				emit(emitMessageStart(s))
				emit(sse("ping", map[string]any{"type": "ping"}))
			}

			if textPiece, ok := asStr(delta["content"]); ok && textPiece != "" {
				sawAnyDelta = true
				for _, e := range openTextBlock(s) {
					emit(e)
				}
				emit(sse("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": s.textIndex,
					"delta": map[string]any{"type": "text_delta", "text": textPiece},
				}))
				inc := len(textPiece) / 4
				if inc < 1 {
					inc = 1
				}
				s.outputTokens += inc
			}

			if toolCalls, ok := asList(delta["tool_calls"]); ok {
				for _, tc := range toolCalls {
					tcm, ok := asMap(tc)
					if !ok {
						continue
					}
					oaiIndex := toInt(tcm["index"])
					fn, _ := asMap(tcm["function"])
					if fn == nil {
						fn = map[string]any{}
					}
					if _, exists := s.toolIndexByOAI[oaiIndex]; !exists {
						toolID, _ := asStr(tcm["id"])
						if toolID == "" {
							toolID = anthropicToolUseID()
						}
						name, _ := asStr(fn["name"])
						for _, e := range openToolBlock(s, oaiIndex, toolID, name) {
							emit(e)
						}
					}
					if argsPiece, ok := asStr(fn["arguments"]); ok && argsPiece != "" {
						sawAnyDelta = true
						for _, e := range emitToolPartial(s, oaiIndex, argsPiece) {
							emit(e)
						}
					}
				}
			}

			if fr, ok := asStr(finishReason); ok && fr != "" {
				s.stopReason = OpenAIFinishReasonToAnthropic(finishReason)
			}
		}
	}

	if !s.started {
		emit(emitMessageStart(s))
	}
	if s.textOpen {
		for _, e := range closeTextBlock(s) {
			emit(e)
		}
	}
	for _, e := range closeAllToolBlocks(s) {
		emit(e)
	}
	if len(s.toolBlocks) > 0 && s.stopReason == "end_turn" && !sawAnyDelta {
		s.stopReason = "tool_use"
	}
	emit(sse("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": s.stopReason, "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": s.outputTokens},
	}))
	emit(sse("message_stop", map[string]any{"type": "message_stop"}))
}

// ---- REVERSE: OpenAI -> Anthropic (native anthropic backend) ------------ //

var stopReasonToFinish = map[string]string{
	"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls", "stop_sequence": "stop",
}

// AnthropicStopReasonToOpenAI maps an Anthropic stop_reason to an OpenAI finish_reason.
func AnthropicStopReasonToOpenAI(reason any) string {
	s, ok := asStr(reason)
	if !ok {
		return ""
	}
	if v, ok := stopReasonToFinish[s]; ok {
		return v
	}
	return "stop"
}

func openaiContentToText(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		var parts []string
		for _, block := range c {
			if bm, ok := asMap(block); ok && bm["type"] == "text" {
				if t, ok := asStr(bm["text"]); ok {
					parts = append(parts, t)
				}
			} else if s, ok := asStr(block); ok {
				parts = append(parts, s)
			}
		}
		return strings.Join(parts, "\n")
	case nil:
		return ""
	}
	return jsonString(content)
}

// OpenAIToolsToAnthropic converts OpenAI tools to Anthropic tool shape.
func OpenAIToolsToAnthropic(tools []any) []any {
	if len(tools) == 0 {
		return nil
	}
	var out []any
	for _, tool := range tools {
		tm, ok := asMap(tool)
		if !ok {
			continue
		}
		var fn map[string]any
		if tm["type"] == "function" {
			fn, _ = asMap(tm["function"])
		} else {
			fn = tm
		}
		if fn == nil {
			continue
		}
		name, _ := asStr(fn["name"])
		if name == "" {
			continue
		}
		desc, _ := fn["description"].(string)
		var schema any = fn["parameters"]
		if schema == nil {
			schema = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		out = append(out, map[string]any{"name": name, "description": desc, "input_schema": schema})
	}
	return out
}

// OpenAIMessagesToAnthropic splits an OpenAI message list into Anthropic (system, messages).
// cache_control on message content parts is carried to the Anthropic blocks;
// cache_control on system parts needs OpenAIMessagesToAnthropicWithReport,
// because the system returned here is a plain string.
func OpenAIMessagesToAnthropic(messages []map[string]any) (string, []map[string]any) {
	converted := openAIMessagesToAnthropic(messages)
	return converted.system, converted.messages
}

func openAIMessagesToAnthropic(messages []map[string]any) anthropicConversion {
	var systemParts []string
	var systemPieces []textPiece
	var systemCached, dropped []string
	toolResults := -1
	out := []map[string]any{}
	for i, msg := range messages {
		role, _ := asStr(msg["role"])
		content := msg["content"]
		pieces, cached, lost := contentPieces(content, "messages."+strconv.Itoa(i))
		dropped = append(dropped, lost...)
		switch role {
		case "system", "developer":
			if text := openaiContentToText(content); text != "" {
				systemParts = append(systemParts, text)
				if len(systemPieces) > 0 {
					pieces[0].sep = "\n\n"
				}
				systemPieces = append(systemPieces, pieces...)
				systemCached = append(systemCached, cached...)
			} else {
				dropped = append(dropped, cached...)
			}
		case "assistant":
			blocks := []any{}
			var text string
			if s, ok := content.(string); ok {
				text = s
			} else {
				text = openaiContentToText(content)
			}
			if len(cached) > 0 {
				blocks = append(blocks, cacheBlocks(pieces)...)
			} else if text != "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": text})
			}
			if calls, ok := asList(msg["tool_calls"]); ok {
				for _, call := range calls {
					cm, ok := asMap(call)
					if !ok {
						continue
					}
					fn, _ := asMap(cm["function"])
					if fn == nil {
						fn = map[string]any{}
					}
					var toolInput any
					args, _ := asStr(fn["arguments"])
					if args == "" {
						args = "{}"
					}
					if json.Unmarshal([]byte(args), &toolInput) != nil {
						toolInput = map[string]any{}
					}
					if _, ok := toolInput.(map[string]any); !ok {
						toolInput = map[string]any{}
					}
					id, _ := asStr(cm["id"])
					if id == "" {
						id = anthropicToolUseID()
					}
					name, _ := asStr(fn["name"])
					blocks = append(blocks, map[string]any{
						"type": "tool_use", "id": id, "name": name, "input": toolInput,
					})
				}
			}
			if len(blocks) == 0 {
				blocks = []any{map[string]any{"type": "text", "text": ""}}
			}
			out = append(out, map[string]any{"role": "assistant", "content": blocks})
		case "tool":
			tid, _ := asStr(msg["tool_call_id"])
			var resultContent any = openaiContentToText(content)
			if len(cached) > 0 {
				resultContent = cacheBlocks(pieces)
			}
			block := map[string]any{
				"type": "tool_result", "tool_use_id": tid, "content": resultContent,
			}
			// Consecutive tool results share one user message. Only a message
			// built from tool results takes more; user content that became
			// blocks to carry cache_control does not.
			if len(out) > 0 && out[len(out)-1]["role"] == "user" && len(out)-1 == toolResults {
				if lst, ok := out[len(out)-1]["content"].([]any); ok {
					out[len(out)-1]["content"] = append(lst, block)
					continue
				}
			}
			out = append(out, map[string]any{"role": "user", "content": []any{block}})
			toolResults = len(out) - 1
		default:
			var userContent any = openaiContentToText(content)
			if len(cached) > 0 {
				userContent = cacheBlocks(pieces)
			}
			out = append(out, map[string]any{"role": "user", "content": userContent})
		}
	}
	converted := anthropicConversion{system: strings.Join(systemParts, "\n\n"), messages: out, droppedCacheControl: dropped}
	if len(systemCached) > 0 {
		converted.systemBlocks = cacheBlocks(systemPieces)
	}
	return converted
}

// AnthropicResponseToOpenAI converts a non-streaming Anthropic Message to an OpenAI completion.
func AnthropicResponseToOpenAI(response map[string]any, model string) map[string]any {
	var textParts []string
	var toolCalls []any
	if blocks, ok := asList(response["content"]); ok {
		for _, block := range blocks {
			bm, ok := asMap(block)
			if !ok {
				continue
			}
			switch bm["type"] {
			case "text":
				if t, ok := asStr(bm["text"]); ok {
					textParts = append(textParts, t)
				}
			case "tool_use":
				id, _ := asStr(bm["id"])
				name, _ := asStr(bm["name"])
				var input any = bm["input"]
				if input == nil {
					input = map[string]any{}
				}
				toolCalls = append(toolCalls, map[string]any{
					"id": id, "type": "function",
					"function": map[string]any{"name": name, "arguments": jsonString(input)},
				})
			}
		}
	}
	message := map[string]any{"role": "assistant"}
	if joined := strings.Join(textParts, ""); joined != "" {
		message["content"] = joined
	} else {
		message["content"] = nil
	}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
	}
	usage, _ := asMap(response["usage"])
	prompt := toInt(usage["input_tokens"])
	completion := toInt(usage["output_tokens"])
	id, _ := asStr(response["id"])
	if id == "" {
		id = chatCmplID()
	}
	finish := AnthropicStopReasonToOpenAI(response["stop_reason"])
	if finish == "" {
		finish = "stop"
	}
	return map[string]any{
		"id": id, "object": "chat.completion", "model": model,
		"choices": []any{map[string]any{
			"index": 0, "message": message, "finish_reason": finish,
		}},
		"usage": map[string]any{
			"prompt_tokens": prompt, "completion_tokens": completion,
			"total_tokens": prompt + completion,
		},
	}
}

// AnthropicSSEToOpenAIChunks translates Anthropic SSE data lines into OpenAI
// chat.completion.chunk JSON strings, invoking emit for each. It emits no
// usage and ends every stream with a finish chunk, even one that failed or
// was cut short; AnthropicSSEToOpenAIChunksWithOptions reports both.
func AnthropicSSEToOpenAIChunks(lines func() (string, bool), model string, emit func(string)) {
	translateAnthropicSSE(lines, model, emit, StreamOptions{}, true)
}

// StreamOptions selects optional output of
// AnthropicSSEToOpenAIChunksWithOptions.
type StreamOptions struct {
	// IncludeUsage emits what OpenAI sends for stream_options.include_usage:
	// after the finish chunk, a chunk with empty choices and the usage of the
	// whole request, and "usage": null on every other chunk. The usage chunk
	// is sent only for a completed stream whose source reported usage, so it
	// never carries a partial or invented count.
	IncludeUsage bool
}

// Usage holds the token counts an Anthropic stream reported. Anthropic counts
// cache writes and cache reads apart from InputTokens, so the Chat
// prompt_tokens is the sum of all three.
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	OutputTokens             int `json:"output_tokens"`
}

// PromptTokens returns the Chat prompt_tokens, which also counts the input
// read from or written to the cache.
func (u Usage) PromptTokens() int {
	return u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens
}

// TotalTokens returns the Chat total_tokens.
func (u Usage) TotalTokens() int {
	return u.PromptTokens() + u.OutputTokens
}

// StreamReport tells how a translated stream ended.
type StreamReport struct {
	// Usage holds the latest count of each kind the source reported, also
	// for a stream that failed, and is zero when the source reported none.
	Usage Usage
	// Completed is true when message_stop arrived and no error event did.
	Completed bool
	// Err is nil when Completed. Otherwise it is the *StreamError of the
	// first error event, or ErrStreamTruncated.
	Err error
}

// StreamError is an error event received from an Anthropic stream, such as
// an overloaded_error after the response began.
type StreamError struct {
	Type    string // Anthropic's error type, such as overloaded_error
	Message string
}

func (e *StreamError) Error() string {
	if e.Type == "" {
		return "Anthropic stream error: " + e.Message
	}
	return "Anthropic stream " + e.Type + ": " + e.Message
}

// ErrStreamTruncated is the error of a source that ended without
// message_stop and without an error event, such as a dropped connection.
var ErrStreamTruncated = errors.New("Anthropic stream ended without message_stop")

// AnthropicSSEToOpenAIChunksWithOptions translates like
// AnthropicSSEToOpenAIChunks and, once lines is exhausted, returns a report
// that is never nil.
//
// Without options, a completed stream gets exactly the chunks
// AnthropicSSEToOpenAIChunks emits; StreamOptions describes what
// IncludeUsage adds. Any other stream ends where its source did, without a
// finish chunk, so a failed or truncated answer cannot pass for a complete
// one. Its chunks all come from source events: when an error arrives first,
// nothing has been emitted and the caller can still respond with an error
// status.
func AnthropicSSEToOpenAIChunksWithOptions(lines func() (string, bool), model string, emit func(string), opts StreamOptions) *StreamReport {
	return translateAnthropicSSE(lines, model, emit, opts, false)
}

// translateAnthropicSSE is the loop behind both forms. alwaysFinish ends every
// stream with a finish chunk, the contract of AnthropicSSEToOpenAIChunks;
// without it only a completed stream gets one.
func translateAnthropicSSE(lines func() (string, bool), model string, emit func(string), opts StreamOptions, alwaysFinish bool) *StreamReport {
	chatID := chatCmplID()
	toolOrdinalByBlock := map[int]int{}
	nextToolOrdinal := 0
	var stopReason any
	started := false
	report := &StreamReport{}
	sawUsage, sawStop := false, false

	chunk := func(delta map[string]any, finishReason any) string {
		value := map[string]any{
			"id": chatID, "object": "chat.completion.chunk", "model": model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finishReason}},
		}
		if opts.IncludeUsage {
			value["usage"] = nil
		}
		b, _ := json.Marshal(value)
		return string(b)
	}

	for {
		raw, ok := lines()
		if !ok {
			break
		}
		line := strings.TrimSpace(raw)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(line[len("data:"):])
		if payload == "" {
			continue
		}
		var obj map[string]any
		if json.Unmarshal([]byte(payload), &obj) != nil {
			continue
		}
		switch obj["type"] {
		case "message_start":
			if message, ok := asMap(obj["message"]); ok && mergeUsage(&report.Usage, message["usage"]) {
				sawUsage = true
			}
			if !started {
				started = true
				emit(chunk(map[string]any{"role": "assistant", "content": ""}, nil))
			}
		case "content_block_start":
			block, _ := asMap(obj["content_block"])
			if block != nil && block["type"] == "tool_use" {
				ordinal := nextToolOrdinal
				nextToolOrdinal++
				toolOrdinalByBlock[toInt(obj["index"])] = ordinal
				id, _ := asStr(block["id"])
				name, _ := asStr(block["name"])
				emit(chunk(map[string]any{"tool_calls": []any{map[string]any{
					"index": ordinal, "id": id, "type": "function",
					"function": map[string]any{"name": name, "arguments": ""},
				}}}, nil))
			}
		case "content_block_delta":
			delta, _ := asMap(obj["delta"])
			switch delta["type"] {
			case "text_delta":
				if text, ok := asStr(delta["text"]); ok && text != "" {
					if !started {
						started = true
						emit(chunk(map[string]any{"role": "assistant", "content": ""}, nil))
					}
					emit(chunk(map[string]any{"content": text}, nil))
				}
			case "input_json_delta":
				if partial, ok := asStr(delta["partial_json"]); ok && partial != "" {
					ordinal := toolOrdinalByBlock[toInt(obj["index"])]
					emit(chunk(map[string]any{"tool_calls": []any{map[string]any{
						"index": ordinal, "function": map[string]any{"arguments": partial},
					}}}, nil))
				}
			}
		case "message_delta":
			if inner, ok := asMap(obj["delta"]); ok {
				if sr := inner["stop_reason"]; sr != nil {
					stopReason = sr
				}
			}
			if mergeUsage(&report.Usage, obj["usage"]) {
				sawUsage = true
			}
		case "message_stop":
			sawStop = true
		case "error":
			if report.Err == nil {
				detail, _ := asMap(obj["error"])
				kind, _ := asStr(detail["type"])
				message, _ := asStr(detail["message"])
				report.Err = &StreamError{Type: kind, Message: message}
			}
		}
	}

	if report.Err == nil && !sawStop {
		report.Err = ErrStreamTruncated
	}
	report.Completed = report.Err == nil
	if !report.Completed && !alwaysFinish {
		return report
	}

	if !started {
		emit(chunk(map[string]any{"role": "assistant", "content": ""}, nil))
	}
	finish := AnthropicStopReasonToOpenAI(stopReason)
	if finish == "" {
		finish = "stop"
	}
	emit(chunk(map[string]any{}, finish))
	if opts.IncludeUsage && sawUsage && report.Completed {
		usage := report.Usage
		b, _ := json.Marshal(map[string]any{
			"id": chatID, "object": "chat.completion.chunk", "model": model, "choices": []any{},
			"usage": map[string]any{
				"prompt_tokens": usage.PromptTokens(), "completion_tokens": usage.OutputTokens,
				"total_tokens":          usage.TotalTokens(),
				"prompt_tokens_details": map[string]any{"cached_tokens": usage.CacheReadInputTokens},
			},
		})
		emit(string(b))
	}
	return report
}

// mergeUsage copies the counts present in an Anthropic usage object into u
// and reports whether there were any. Both message_start and message_delta
// carry running totals, so a later count of a kind replaces an earlier one
// and a kind left out keeps its value.
func mergeUsage(u *Usage, raw any) bool {
	usage, ok := asMap(raw)
	if !ok {
		return false
	}
	found := false
	for key, count := range map[string]*int{
		"input_tokens":                &u.InputTokens,
		"cache_creation_input_tokens": &u.CacheCreationInputTokens,
		"cache_read_input_tokens":     &u.CacheReadInputTokens,
		"output_tokens":               &u.OutputTokens,
	} {
		if value := usage[key]; value != nil {
			*count = toInt(value)
			found = true
		}
	}
	return found
}
