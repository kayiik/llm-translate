package translate

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// ResponsesSSEToOpenAIChunksObserved consumes complete JSON event data and
// emits Chat JSON chunks. The caller owns SSE framing and the final [DONE].
func ResponsesSSEToOpenAIChunksObserved(events func() (string, bool), model string, emit func(string), opts StreamObservationOptions) StreamObservation {
	b := &responsesChatBridge{s: newObservedStream(ConversionChatToResponses, UsageDialectResponses, opts), model: model, emit: emit, includeUsage: opts.IncludeUsage, items: map[int]*nativeBridgeItem{}, createdAt: time.Now().Unix()}
	bridgeFeature(b.s, FidelityContent)
	terminal := false
	for {
		raw, ok := events()
		if !ok {
			b.s.result.Err = ErrResponsesStreamTruncated
			break
		}
		event, ok := decodeObservedObject(raw)
		if !ok {
			b.s.result.Err = protocolErrorf("Responses", "stream contains invalid JSON event data")
			break
		}
		kind, _ := event["type"].(string)
		if rawSequence, exists := event["sequence_number"]; exists {
			sequence, valid := usageInteger(rawSequence)
			if !valid || (b.sequenceSeen && sequence <= b.lastSequence) {
				b.s.add(material("sequence_number", LossUnsupported, "source event sequence did not advance"))
			}
			b.lastSequence = sequence
			b.sequenceSeen = true
		}
		known := "type sequence_number response response_id output_index item item_id content_index part delta text refusal arguments annotation logprobs error code message param summary_index"
		// Responses stream padding is enabled by default upstream. It is
		// transport metadata, not caller content or a control to disable.
		if strings.HasSuffix(kind, ".delta") {
			known += " obfuscation"
			if padding := event["obfuscation"]; padding != nil {
				if _, ok := padding.(string); !ok {
					b.s.unknown("$event.obfuscation")
				}
			}
		}
		bridgeFields(b.s, event, known, "$event")
		switch kind {
		case "response.created", "response.in_progress":
			response, valid := asMap(event["response"])
			if !valid {
				b.s.add(material("response", LossUnsupported, "created/in-progress event requires a response object"))
				break
			}
			b.envelope(response)
			if bridgeGate(b.s, CompletionPartial) {
				b.start()
			}
		case "response.output_item.added", "response.output_item.done":
			index, ok := bridgeIndex(event, "output_index")
			if !ok {
				b.s.add(material("output_index", LossUnsupported, "item event requires a source output index"))
				break
			}
			item, _ := asMap(event["item"])
			b.item(index, item)
		case "response.content_part.added", "response.content_part.done":
			index, ok := bridgeIndex(event, "output_index")
			content, valid := bridgeIndex(event, "content_index")
			if !ok || !valid {
				b.s.add(material("content_index", LossUnsupported, "part event requires output/content indexes"))
				break
			}
			part, _ := asMap(event["part"])
			b.partFull(index, content, event["item_id"], part)
		case "response.output_text.delta", "response.refusal.delta":
			index, ok := bridgeIndex(event, "output_index")
			content, valid := bridgeIndex(event, "content_index")
			if !ok || !valid {
				b.s.add(material("content_index", LossUnsupported, "text event requires output/content indexes"))
				break
			}
			partKind := "output_text"
			if kind == "response.refusal.delta" {
				partKind = "refusal"
			}
			if raw := event["logprobs"]; raw != nil {
				list, ok := asList(raw)
				if !ok || len(list) > 0 {
					b.s.unknown(fmt.Sprintf("output.%d.content.%d.logprobs", index, content))
				}
			}
			value, ok := event["delta"].(string)
			if !ok {
				b.s.add(material("delta", LossUnsupported, "text/refusal delta must be a string"))
				break
			}
			b.partDelta(index, content, event["item_id"], partKind, value, false)
		case "response.output_text.done", "response.refusal.done":
			index, ok := bridgeIndex(event, "output_index")
			content, valid := bridgeIndex(event, "content_index")
			if !ok || !valid {
				b.s.add(material("content_index", LossUnsupported, "done event requires output/content indexes"))
				break
			}
			partKind, key := "output_text", "text"
			if kind == "response.refusal.done" {
				partKind, key = "refusal", "refusal"
			}
			value, ok := event[key].(string)
			if !ok {
				b.s.add(material(key, LossUnsupported, "done content must be a string"))
				break
			}
			b.partDelta(index, content, event["item_id"], partKind, value, true)
		case "response.function_call_arguments.delta", "response.function_call_arguments.done":
			index, ok := bridgeIndex(event, "output_index")
			if !ok {
				b.s.add(material("output_index", LossUnsupported, "argument event requires an output index"))
				break
			}
			key := "delta"
			if kind == "response.function_call_arguments.done" {
				key = "arguments"
			}
			value, valid := event[key].(string)
			if !valid {
				b.s.add(material(key, LossUnsupported, "argument fragment must be a string"))
				break
			}
			b.arguments(index, event["item_id"], value, key == "arguments")
		case "response.output_text.annotation.added":
			index, _ := bridgeIndex(event, "output_index")
			content, _ := bridgeIndex(event, "content_index")
			b.s.add(material(fmt.Sprintf("output.%d.content.%d.annotations", index, content), LossDropped, "native annotations are not emitted by Chat"))
		case "response.completed", "response.incomplete":
			response, ok := asMap(event["response"])
			if !ok {
				b.s.add(material("response", LossUnsupported, "terminal event requires a response object"))
				break
			}
			b.envelope(response)
			if kind == "response.completed" {
				for index, native := range b.items {
					if native.item.kind == "function_call" {
						if _, ok := decodeObservedObject(native.item.arguments.String()); !ok {
							b.s.add(material(fmt.Sprintf("output.%d.arguments", index), LossUnsupported, "completed function arguments are not a JSON object"))
						}
					}
				}
			}
			finish := "stop"
			if b.toolCount > 0 {
				finish = "tool_calls"
			}
			if kind == "response.incomplete" {
				detail, _ := asMap(response["incomplete_details"])
				switch detail["reason"] {
				case "max_output_tokens":
					finish = "length"
				case "content_filter":
					finish = "content_filter"
				default:
					b.s.add(material("incomplete_details.reason", LossUnsupported, "native incomplete reason is not representable by Chat"))
				}
			}
			if status, _ := response["status"].(string); status != "" && status != strings.TrimPrefix(kind, "response.") {
				b.s.add(material("response.status", LossUnsupported, "terminal event and response status disagree"))
			}
			if detail, ok := asMap(response["error"]); ok && len(detail) > 0 {
				b.s.result.Err = bridgeSourceError(detail)
			}
			bridgeUsageValidity(b.s)
			if bridgeGate(b.s, CompletionComplete) {
				b.s.result.Completed = true
				b.start()
				b.chunk(map[string]any{}, finish, nil)
				if b.includeUsage {
					if usage := bridgeChatUsage(b.s.usage.Snapshot(CompletionComplete)); usage != nil {
						b.chunk(nil, nil, usage)
					} else {
						b.s.add(advisory("usage", LossDropped, "complete source usage is unavailable; no final usage chunk is fabricated"))
					}
				}
				terminal = true
			}
		case "response.failed", "response.cancelled", "error":
			response, _ := asMap(event["response"])
			b.s.usage.Merge(response["usage"])
			detail, _ := asMap(event["error"])
			if detail == nil {
				detail, _ = asMap(response["error"])
			}
			if detail == nil {
				detail = event
			}
			b.s.result.Err = bridgeSourceError(detail)
		default:
			if strings.HasPrefix(kind, "response.reasoning_summary_") {
				bridgeReplay(b.s, event, "$event")
				b.s.add(advisory("$event.summary", LossDropped, "display reasoning summary is not emitted by Chat"))
				if b.s.assessment.Requirements.Replay {
					b.s.add(material("$event.item_id", LossDropped, "native reasoning identity cannot be replayed through Chat"))
				}
			} else {
				b.s.unknown("$event.type")
			}
		}
		if !bridgeGate(b.s, CompletionPartial) || terminal {
			break
		}
	}
	quality := CompletionComplete
	if !terminal || b.s.result.Err != nil {
		b.s.result.Completed = false
		quality = CompletionFailed
		if b.s.result.Err == ErrResponsesStreamTruncated {
			quality = CompletionTruncated
		}
	}
	return b.s.finish(quality)
}

type nativeBridgeItem struct {
	item      *bridgeItem
	toolIndex int
	parts     map[int]*bridgePart
}
type responsesChatBridge struct {
	s                     *observedStream
	model, id             string
	emit                  func(string)
	includeUsage, started bool
	items                 map[int]*nativeBridgeItem
	toolCount             int
	createdAt             int64
	lastSequence          int64
	sequenceSeen          bool
}

func (b *responsesChatBridge) chunk(delta map[string]any, finish any, usage map[string]any) {
	choices := []any{}
	if delta != nil {
		choices = append(choices, map[string]any{"index": 0, "delta": delta, "finish_reason": finish})
	}
	body := map[string]any{"id": b.id, "object": "chat.completion.chunk", "model": b.model, "choices": choices, "created": b.createdAt}
	if b.includeUsage {
		body["usage"] = usage
	}
	encoded, _ := json.Marshal(body)
	b.emit(string(encoded))
}

func (b *responsesChatBridge) start() {
	if b.started {
		return
	}
	if b.id == "" {
		b.id = chatCmplID()
	}
	b.started = true
	b.chunk(map[string]any{"role": "assistant"}, nil, nil)
}

func (b *responsesChatBridge) envelope(response map[string]any) {
	bridgeFields(b.s, response, "id object created_at model output output_text status error incomplete_details usage background instructions max_output_tokens max_tool_calls parallel_tool_calls previous_response_id reasoning service_tier store temperature text tool_choice tools top_logprobs top_p truncation user metadata", "response")
	if output := response["output"]; output != nil {
		if _, ok := asList(output); !ok {
			b.s.add(material("response.output", LossUnsupported, "supplied response output must be a list"))
		}
	}
	if id, _ := response["id"].(string); id != "" {
		if b.id == "" {
			b.id = id
		} else if b.id != id {
			loss := advisory("response.id", LossApproximated, "late source identity cannot replace the fixed output identity")
			if b.s.assessment.Requirements.Replay {
				loss.Severity = LossMaterial
			}
			b.s.add(loss)
		}
	}
	if !b.started {
		if model, _ := response["model"].(string); model != "" {
			b.model = model
		}
		if created, ok := usageInteger(response["created_at"]); ok {
			b.createdAt = created
		}
	}
	b.s.usage.Merge(response["usage"])
	bridgeReplay(b.s, response, "response")
	if response["background"] == true {
		b.s.add(material("response.background", LossUnsupported, "background lifecycle is not implemented by a Chat stream"))
	}
	if !bridgeGate(b.s, CompletionPartial) {
		return
	}
	for index, raw := range mustList(response["output"]) {
		item, _ := asMap(raw)
		if id, _ := item["id"].(string); id != "" {
			for existing, native := range b.items {
				if native.item.id == id {
					index = existing
					break
				}
			}
		}
		if !b.item(index, item) {
			return
		}
	}
}

func (b *responsesChatBridge) item(index int, raw map[string]any) bool {
	path := fmt.Sprintf("output.%d", index)
	bridgeFields(b.s, raw, "id type status role content call_id name arguments summary encrypted_content phase", path)
	bridgeReplay(b.s, raw, path)
	kind, _ := raw["type"].(string)
	if kind == "reasoning" {
		b.s.add(advisory(path+".reasoning", LossDropped, "display reasoning is not emitted by Chat"))
		return bridgeGate(b.s, CompletionPartial)
	}
	if kind != "message" && kind != "function_call" {
		b.s.add(material(path, LossUnsupported, "native/provider-executed item is not a client function or message"))
		return false
	}
	id, _ := raw["id"].(string)
	if id == "" {
		b.s.add(material(path+".id", LossUnsupported, "native item identity is required for incremental correlation"))
	}
	if !bridgeGate(b.s, CompletionPartial) {
		return false
	}
	current := b.items[index]
	for otherIndex, other := range b.items {
		if otherIndex != index && other.item.id == id {
			b.s.add(material(path+".id", LossUnsupported, "native item identity is shared by different output indexes"))
			return false
		}
	}
	if current == nil {
		current = &nativeBridgeItem{item: &bridgeItem{kind: kind, id: id, index: index}, parts: map[int]*bridgePart{}}
		b.items[index] = current
	} else if current.item.id != id || current.item.kind != kind {
		b.s.add(material(path+".id", LossUnsupported, "source output index was reused for a different item"))
		return false
	}
	if kind == "function_call" {
		bridgeFeature(b.s, FidelityTools)
		callID, _ := raw["call_id"].(string)
		name, _ := raw["name"].(string)
		if callID == "" || name == "" {
			b.s.add(material(path, LossUnsupported, "function item needs original call_id and name; neither is repaired"))
			return false
		}
		for otherIndex, other := range b.items {
			if otherIndex != index && other.item.kind == "function_call" && other.item.callID == callID {
				b.s.add(material(path+".call_id", LossUnsupported, "native function call identity is duplicated"))
				return false
			}
		}
		if current.item.published && (current.item.callID != callID || current.item.name != name) {
			b.s.add(material(path, LossUnsupported, "published function identity/name changed"))
			return false
		}
		if !current.item.published {
			current.item.callID = callID
			current.item.name = name
			current.item.published = true
			current.toolIndex = b.toolCount
			b.toolCount++
			if !bridgeGate(b.s, CompletionPartial) {
				return false
			}
			b.start()
			b.chunk(map[string]any{"tool_calls": []any{map[string]any{"index": current.toolIndex, "id": callID, "type": "function", "function": map[string]any{"name": name, "arguments": ""}}}}, nil, nil)
		}
		if args, ok := raw["arguments"].(string); ok {
			return b.arguments(index, id, args, true)
		}
		if raw["arguments"] != nil {
			b.s.add(material(path+".arguments", LossUnsupported, "function arguments must be a string"))
			return false
		}
		return true
	}
	if role, _ := raw["role"].(string); role != "" && role != "assistant" {
		b.s.add(material(path+".role", LossUnsupported, "message role is not assistant"))
		return false
	}
	if value := raw["content"]; value != nil {
		if _, ok := asList(value); !ok {
			b.s.add(material(path+".content", LossUnsupported, "message content must be a list of native parts"))
			return false
		}
	}
	for content, rawPart := range mustList(raw["content"]) {
		part, _ := asMap(rawPart)
		if !b.partFull(index, content, id, part) {
			return false
		}
	}
	return true
}

func (b *responsesChatBridge) message(index int, rawID any) *nativeBridgeItem {
	id, _ := rawID.(string)
	path := fmt.Sprintf("output.%d", index)
	if id == "" {
		b.s.add(material(path+".id", LossUnsupported, "content event needs a native item_id"))
		return nil
	}
	current := b.items[index]
	if current == nil {
		current = &nativeBridgeItem{item: &bridgeItem{kind: "message", id: id, index: index}, parts: map[int]*bridgePart{}}
		b.items[index] = current
	}
	if current.item.kind != "message" || current.item.id != id {
		b.s.add(material(path+".id", LossUnsupported, "content event does not match its message"))
		return nil
	}
	return current
}

func (b *responsesChatBridge) partFull(index, content int, id any, part map[string]any) bool {
	path := fmt.Sprintf("output.%d.content.%d", index, content)
	bridgeFields(b.s, part, "type text refusal annotations logprobs", path)
	if raw := part["annotations"]; raw != nil {
		list, ok := asList(raw)
		if !ok || len(list) > 0 {
			b.s.add(material(path+".annotations", LossDropped, "native annotations are not emitted by Chat"))
		}
	}
	if raw := part["logprobs"]; raw != nil {
		list, ok := asList(raw)
		if !ok || len(list) > 0 {
			b.s.unknown(path + ".logprobs")
		}
	}
	kind, _ := part["type"].(string)
	key := "text"
	if kind == "refusal" {
		key = "refusal"
	} else if kind != "output_text" {
		b.s.add(material(path, LossUnsupported, "native content part is not representable by Chat"))
		return false
	}
	value, ok := part[key].(string)
	if !ok {
		b.s.add(material(path+"."+key, LossUnsupported, "native content must be a string"))
		return false
	}
	return b.partDelta(index, content, id, kind, value, true)
}

func (b *responsesChatBridge) partDelta(index, content int, id any, kind, value string, full bool) bool {
	current := b.message(index, id)
	if current == nil {
		return false
	}
	part := current.parts[content]
	if part == nil {
		part = &bridgePart{kind: kind, index: content}
		current.parts[content] = part
	}
	path := fmt.Sprintf("output.%d.content.%d", index, content)
	if part.kind != kind {
		b.s.add(material(path+".type", LossUnsupported, "content index changed kind"))
		return false
	}
	if full {
		if !strings.HasPrefix(value, part.text.String()) {
			b.s.add(material(path, LossUnsupported, "done content contradicts already emitted text"))
			return false
		}
		value = strings.TrimPrefix(value, part.text.String())
	}
	if !bridgeGate(b.s, CompletionPartial) {
		return false
	}
	if value != "" {
		part.text.WriteString(value)
		b.start()
		key := "content"
		if kind == "refusal" {
			key = "refusal"
		}
		b.chunk(map[string]any{key: value}, nil, nil)
	}
	return true
}

func (b *responsesChatBridge) arguments(index int, id any, value string, full bool) bool {
	current := b.items[index]
	path := fmt.Sprintf("output.%d.arguments", index)
	if current == nil || current.item.kind != "function_call" || !current.item.published || id != current.item.id {
		b.s.add(material(path, LossUnsupported, "argument event has no matching published function item"))
		return false
	}
	if full {
		if !strings.HasPrefix(value, current.item.arguments.String()) {
			b.s.add(material(path, LossUnsupported, "done arguments contradict already emitted fragments"))
			return false
		}
		value = strings.TrimPrefix(value, current.item.arguments.String())
	}
	if !bridgeGate(b.s, CompletionPartial) {
		return false
	}
	if value != "" {
		current.item.arguments.WriteString(value)
		b.chunk(map[string]any{"tool_calls": []any{map[string]any{"index": current.toolIndex, "function": map[string]any{"arguments": value}}}}, nil, nil)
	}
	return true
}

func bridgeChatUsage(u UsageObservation) map[string]any {
	if !allPresent(u.InputTokens, u.OutputTokens, u.TotalTokens) {
		return nil
	}
	usage := map[string]any{"prompt_tokens": u.InputTokens.Value, "completion_tokens": u.OutputTokens.Value, "total_tokens": u.TotalTokens.Value}
	details := map[string]any{}
	if u.CacheReadTokens.Presence == CountPresent {
		details["cached_tokens"] = u.CacheReadTokens.Value
	}
	if u.CacheWriteTokens.Presence == CountPresent {
		details["cache_write_tokens"] = u.CacheWriteTokens.Value
	}
	if len(details) > 0 {
		usage["prompt_tokens_details"] = details
	}
	if u.ReasoningTokens.Presence == CountPresent {
		usage["completion_tokens_details"] = map[string]any{"reasoning_tokens": u.ReasoningTokens.Value}
	}
	return usage
}
