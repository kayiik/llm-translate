package translate

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

var ErrResponsesStreamTruncated = protocolErrorf("Responses", "stream ended without a terminal event")

// OpenAIStreamToResponsesSSEObserved emits typed Responses SSE incrementally.
// It retains emitted item fields for protocol-required terminal objects, not
// a queue of source events or a generation that must finish before delivery.
func OpenAIStreamToResponsesSSEObserved(chunks func() (string, bool), model string, request map[string]any, emit func(string), opts StreamObservationOptions) StreamObservation {
	b := &chatResponsesBridge{s: newObservedStream(ConversionResponsesToChat, UsageDialectChat, opts), model: model, emit: emit, tools: map[int]*bridgeItem{}, createdAt: time.Now().Unix()}
	bridgeFeature(b.s, FidelityContent)
	b.request = bridgeRequest(b.s, request)
	if !bridgeGate(b.s, CompletionPartial) {
		return b.s.finish(CompletionFailed)
	}
	done := false
	for {
		raw, ok := chunks()
		if !ok {
			b.s.result.Err = ErrChatStreamTruncated
			break
		}
		if strings.TrimSpace(raw) == "[DONE]" {
			done = true
			break
		}
		chunk, ok := decodeObservedObject(raw)
		if !ok {
			b.s.result.Err = protocolErrorf("Chat", "stream contains invalid JSON")
			break
		}
		b.identity(chunk)
		b.s.usage.Merge(chunk["usage"])
		bridgeFields(b.s, chunk, "id object created model choices usage system_fingerprint service_tier", "$")
		if detail, ok := asMap(chunk["error"]); ok {
			b.s.result.Err = bridgeSourceError(detail)
			break
		}
		choices := mustList(chunk["choices"])
		if rawChoices := chunk["choices"]; rawChoices != nil {
			if _, ok := asList(rawChoices); !ok {
				b.s.add(material("choices", LossUnsupported, "choices must be a list"))
			}
		}
		if len(choices) > 1 {
			b.s.add(material("choices.1", LossUnsupported, "multiple Chat choices cannot be represented by one Responses stream"))
		}
		for _, rawChoice := range choices {
			choice, ok := asMap(rawChoice)
			if !ok {
				b.s.add(material("choices", LossUnsupported, "choice must be an object"))
				break
			}
			index, valid := bridgeIndex(choice, "index")
			if !valid || index != 0 {
				b.s.add(material("choices", LossUnsupported, "only source choice index zero is representable"))
				break
			}
			bridgeFields(b.s, choice, "index delta finish_reason", "choices.0")
			delta, deltaOK := asMap(choice["delta"])
			if choice["delta"] != nil && !deltaOK {
				b.s.add(material("choices.0.delta", LossUnsupported, "delta must be an object"))
			}
			bridgeFields(b.s, delta, "role content refusal tool_calls reasoning_content reasoning_details", "choices.0.delta")
			for _, loss := range reasoningFieldLosses("choices.0.delta", delta, "OpenAI Responses") {
				b.s.add(loss)
			}
			bridgeReplay(b.s, delta, "choices.0.delta")
			if raw := delta["role"]; raw != nil {
				if _, ok := raw.(string); !ok {
					b.s.add(material("choices.0.delta.role", LossUnsupported, "role must be a string"))
				}
			}
			if raw := delta["reasoning_details"]; raw != nil {
				if _, ok := asList(raw); !ok {
					b.s.unknown("choices.0.delta.reasoning_details")
				}
			}
			if raw := delta["reasoning_content"]; raw != nil {
				if _, ok := raw.(string); !ok {
					b.s.unknown("choices.0.delta.reasoning_content")
				}
			}
			if value := delta["tool_calls"]; value != nil {
				if _, ok := asList(value); !ok {
					b.s.add(material("choices.0.delta.tool_calls", LossUnsupported, "tool_calls must be a list"))
				}
			}
			for _, key := range []string{"content", "refusal"} {
				if v := delta[key]; v != nil {
					if _, ok := v.(string); !ok {
						b.s.add(material("choices.0.delta."+key, LossUnsupported, "content/refusal delta must be a string"))
					}
				}
			}
			if role, _ := delta["role"].(string); role != "" && role != "assistant" {
				b.s.add(material("choices.0.delta.role", LossUnsupported, "response role must be assistant"))
			}
			if !bridgeGate(b.s, CompletionPartial) {
				break
			}
			b.start()
			if text, ok := delta["content"].(string); ok {
				b.partDelta("output_text", text)
			}
			if refusal, ok := delta["refusal"].(string); ok {
				b.partDelta("refusal", refusal)
			}
			for _, rawCall := range mustList(delta["tool_calls"]) {
				if !b.callDelta(rawCall) {
					break
				}
			}
			if reason, _ := choice["finish_reason"].(string); reason != "" {
				if b.finishReason != "" && b.finishReason != reason {
					b.s.add(material("choices.0.finish_reason", LossUnsupported, "source finish reason changed"))
				}
				b.finishReason = reason
				if reason != "stop" && reason != "tool_calls" && reason != "length" && reason != "content_filter" {
					b.s.add(material("choices.0.finish_reason", LossUnsupported, "source finish reason is not representable"))
				}
			}
		}
		if !bridgeGate(b.s, CompletionPartial) {
			break
		}
	}
	if done && b.finishReason == "" && b.s.result.Err == nil {
		b.s.result.Err = protocolErrorf("Chat", "stream ended without a finish reason")
	}
	for index, item := range b.tools {
		if !item.published {
			b.s.add(material(fmt.Sprintf("choices.0.delta.tool_calls.%d", index), LossUnsupported, "tool identity/name never arrived; no identity is synthesized"))
		}
		if done && b.finishReason != "length" && b.finishReason != "content_filter" {
			if _, ok := decodeObservedObject(item.arguments.String()); !ok {
				b.s.add(material(fmt.Sprintf("choices.0.delta.tool_calls.%d.function.arguments", index), LossUnsupported, "completed function arguments are not a JSON object"))
			}
		}
	}
	bridgeUsageValidity(b.s)
	bridgeUsagePlaceholders(b.s)
	if !bridgeGate(b.s, CompletionPartial) || !done || b.s.result.Err != nil {
		if b.s.result.Err == nil {
			b.s.result.Err = ErrChatStreamTruncated
		}
		if b.id != "" {
			b.event("response.failed", map[string]any{"response": b.envelope("failed", map[string]any{"code": "stream_error", "message": b.s.result.Err.Error()})})
		}
		quality := CompletionFailed
		if !done && b.s.result.Err == ErrChatStreamTruncated {
			quality = CompletionTruncated
		}
		return b.s.finish(quality)
	}
	b.start()
	status, event := "completed", "response.completed"
	if b.finishReason == "length" || b.finishReason == "content_filter" {
		status, event = "incomplete", "response.incomplete"
	}
	if status == "completed" {
		for _, item := range b.items {
			b.doneItem(item, status)
		}
	}
	b.s.result.Completed = true
	b.s.notify(PhaseStreamEvent, CompletionComplete)
	b.event(event, map[string]any{"response": b.envelope(status, nil)})
	return b.s.finish(CompletionComplete)
}

type bridgePart struct {
	kind  string
	index int
	text  strings.Builder
}
type bridgeItem struct {
	kind, id, callID, name string
	index                  int
	arguments              strings.Builder
	parts                  []*bridgePart
	published              bool
}

func (i *bridgeItem) object(status string) map[string]any {
	item := map[string]any{"id": i.id, "type": i.kind, "status": status}
	if i.kind == "function_call" {
		item["call_id"] = i.callID
		item["name"] = i.name
		item["arguments"] = i.arguments.String()
	} else {
		item["role"] = "assistant"
		parts := make([]any, 0, len(i.parts))
		for _, p := range i.parts {
			parts = append(parts, p.object())
		}
		item["content"] = parts
	}
	return item
}

func (p *bridgePart) object() map[string]any {
	if p.kind == "refusal" {
		return map[string]any{"type": "refusal", "refusal": p.text.String()}
	}
	return map[string]any{"type": "output_text", "text": p.text.String(), "annotations": []any{}}
}

type chatResponsesBridge struct {
	s            *observedStream
	model, id    string
	request      map[string]any
	emit         func(string)
	sequence     int
	createdAt    int64
	started      bool
	message      *bridgeItem
	items        []*bridgeItem
	tools        map[int]*bridgeItem
	finishReason string
}

func (b *chatResponsesBridge) identity(chunk map[string]any) {
	if b.id == "" {
		if id, _ := chunk["id"].(string); id != "" {
			b.id = id
			if strings.HasPrefix(id, "chatcmpl-") {
				b.id = "resp_" + strings.TrimPrefix(id, "chatcmpl-")
			}
		} else {
			b.id = "resp_" + randHex(24)
		}
	}
	if !b.started {
		if model, _ := chunk["model"].(string); model != "" {
			b.model = model
		}
		if created, ok := usageInteger(chunk["created"]); ok {
			b.createdAt = created
		}
	}
}

func (b *chatResponsesBridge) event(kind string, body map[string]any) {
	body["type"] = kind
	body["sequence_number"] = b.sequence
	b.sequence++
	b.emit(sse(kind, body))
}
func (b *chatResponsesBridge) start() {
	if b.started {
		return
	}
	if b.id == "" {
		b.id = "resp_" + randHex(24)
	}
	b.started = true
	b.event("response.created", map[string]any{"response": b.envelope("in_progress", nil)})
}

func (b *chatResponsesBridge) envelope(status string, failure map[string]any) map[string]any {
	output := make([]any, 0, len(b.items))
	for _, item := range b.items {
		if item.published {
			output = append(output, item.object(status))
		}
	}
	response := NewResponseEnvelope(b.model, b.id, status, output, b.request)
	response["created_at"] = b.createdAt
	// Synthetic IDs are local envelope identity, never stored lineage.
	response["store"] = false
	response["background"] = false
	response["previous_response_id"] = nil
	if status != "in_progress" {
		response["usage"] = bridgeResponsesUsage(b.s.usage.Snapshot(CompletionComplete))
	}
	if status == "incomplete" {
		reason := "max_output_tokens"
		if b.finishReason == "content_filter" {
			reason = "content_filter"
		}
		response["incomplete_details"] = map[string]any{"reason": reason}
	}
	if failure != nil {
		response["error"] = failure
	}
	return response
}

func (b *chatResponsesBridge) partDelta(kind, text string) {
	if b.message == nil {
		item := &bridgeItem{kind: "message", id: "msg_" + randHex(24), index: len(b.items), published: true}
		b.message = item
		b.items = append(b.items, item)
		b.event("response.output_item.added", map[string]any{"response_id": b.id, "output_index": item.index, "item": item.object("in_progress")})
	}
	item := b.message
	var part *bridgePart
	for _, p := range item.parts {
		if p.kind == kind {
			part = p
		}
	}
	if part == nil {
		part = &bridgePart{kind: kind, index: len(item.parts)}
		item.parts = append(item.parts, part)
		b.event("response.content_part.added", map[string]any{"response_id": b.id, "item_id": item.id, "output_index": item.index, "content_index": part.index, "part": part.object()})
	}
	if text == "" {
		return
	}
	part.text.WriteString(text)
	event := "response.output_text.delta"
	if kind == "refusal" {
		event = "response.refusal.delta"
	}
	b.event(event, map[string]any{"response_id": b.id, "item_id": item.id, "output_index": item.index, "content_index": part.index, "delta": text})
}

func (b *chatResponsesBridge) callDelta(raw any) bool {
	bridgeFeature(b.s, FidelityTools)
	call, ok := asMap(raw)
	index, valid := bridgeIndex(call, "index")
	path := fmt.Sprintf("choices.0.delta.tool_calls.%d", index)
	if !ok || !valid {
		b.s.add(material("choices.0.delta.tool_calls", LossUnsupported, "tool delta requires a nonnegative source index"))
		return false
	}
	bridgeFields(b.s, call, "index id type function extra_content thought_signature", path)
	fn, _ := asMap(call["function"])
	bridgeFields(b.s, fn, "name arguments thought_signature", path+".function")
	if extra, ok := asMap(call["extra_content"]); ok {
		bridgeFields(b.s, extra, "google", path+".extra_content")
		if google, ok := asMap(extra["google"]); ok {
			bridgeFields(b.s, google, "thought_signature", path+".extra_content.google")
		} else if extra["google"] != nil {
			b.s.unknown(path + ".extra_content.google")
		}
	} else if call["extra_content"] != nil {
		b.s.unknown(path + ".extra_content")
	}
	for _, l := range thoughtSignatureLosses(path, call, "OpenAI Responses") {
		b.s.add(l)
	}
	if kind, _ := call["type"].(string); kind != "" && kind != "function" {
		b.s.add(material(path+".type", LossUnsupported, "only client function calls can be converted"))
	}
	if !bridgeGate(b.s, CompletionPartial) {
		return false
	}
	item := b.tools[index]
	if item == nil {
		item = &bridgeItem{kind: "function_call", id: "fc_" + randHex(24), index: len(b.items)}
		b.tools[index] = item
		b.items = append(b.items, item)
	}
	if id, _ := call["id"].(string); id != "" {
		for otherIndex, other := range b.tools {
			if otherIndex != index && other.callID == id {
				b.s.add(material(path+".id", LossUnsupported, "tool identity is shared by different source indexes"))
			}
		}
		if item.callID != "" && item.callID != id {
			b.s.add(material(path+".id", LossUnsupported, "source tool identity changed"))
		}
		item.callID = id
	}
	if name, _ := fn["name"].(string); name != "" {
		if item.published {
			b.s.add(material(path+".function.name", LossUnsupported, "published function name cannot be changed by this wire format"))
		}
		item.name += name
	}
	if !bridgeGate(b.s, CompletionPartial) {
		return false
	}
	args, hasArgs := fn["arguments"].(string)
	if value := fn["arguments"]; value != nil && !hasArgs {
		b.s.add(material(path+".function.arguments", LossUnsupported, "argument fragments must be strings"))
		return false
	}
	if hasArgs {
		item.arguments.WriteString(args)
	}
	if !item.published && item.callID != "" && item.name != "" {
		item.published = true
		initial := item.object("in_progress")
		initial["arguments"] = ""
		b.event("response.output_item.added", map[string]any{"response_id": b.id, "output_index": item.index, "item": initial})
		args = item.arguments.String()
	}
	if item.published && args != "" {
		b.event("response.function_call_arguments.delta", map[string]any{"response_id": b.id, "item_id": item.id, "output_index": item.index, "delta": args})
	}
	return true
}

func (b *chatResponsesBridge) doneItem(item *bridgeItem, status string) {
	if item.kind == "function_call" {
		b.event("response.function_call_arguments.done", map[string]any{"response_id": b.id, "item_id": item.id, "output_index": item.index, "arguments": item.arguments.String()})
	} else {
		for _, part := range item.parts {
			kind, field := "response.output_text.done", "text"
			if part.kind == "refusal" {
				kind, field = "response.refusal.done", "refusal"
			}
			b.event(kind, map[string]any{"response_id": b.id, "item_id": item.id, "output_index": item.index, "content_index": part.index, field: part.text.String()})
			b.event("response.content_part.done", map[string]any{"response_id": b.id, "item_id": item.id, "output_index": item.index, "content_index": part.index, "part": part.object()})
		}
	}
	b.event("response.output_item.done", map[string]any{"response_id": b.id, "output_index": item.index, "item": item.object(status)})
}

func bridgeGate(s *observedStream, quality CompletionQuality) bool {
	s.notify(PhaseStreamEvent, quality)
	if s.result.Err != nil {
		return false
	}
	if err := RejectMaterialLoss(s.assessment.Report); err != nil {
		s.result.Err = err
		return false
	}
	if s.assessment.Feasibility != FeasibilitySupported || len(s.assessment.UnknownPaths) > 0 {
		s.result.Err = fmt.Errorf("stream semantics are unassessed at %s", strings.Join(s.assessment.UnknownPaths, ", "))
		return false
	}
	return true
}

func bridgeFields(s *observedStream, object map[string]any, known, prefix string) {
	for key, value := range object {
		if value != nil && !assessmentKnown(known, key) {
			s.unknown(prefix + "." + key)
		}
	}
}

func bridgeReplay(s *observedStream, object map[string]any, prefix string) {
	inferred := InferFidelityRequirements(object)
	if inferred.Replay {
		s.assessment.Requirements.Replay = true
	}
	if !s.assessment.Requirements.Replay {
		return
	}
	bridgeFeature(s, FidelityReplay)
	walkFidelity(object, prefix, func(path, key string, value any, item map[string]any) {
		if present(value) && (replayField(key, item) || (key == "id" && (item["type"] == "reasoning" || item["type"] == "message" || item["type"] == "function_call"))) {
			s.add(material(path, LossDropped, "opaque native continuation state is not representable by this bridge"))
		}
	})
}

func bridgeFeature(s *observedStream, feature FidelityFeature) {
	switch feature {
	case FidelityContent:
		s.assessment.Requirements.Content = true
	case FidelityTools:
		s.assessment.Requirements.Tools = true
	case FidelityReplay:
		s.assessment.Requirements.Replay = true
	}
	for _, existing := range s.assessment.Evaluated {
		if feature == existing {
			return
		}
	}
	s.assessment.Evaluated = append(s.assessment.Evaluated, feature)
}

func bridgeUsagePlaceholders(s *observedStream) {
	u := s.usage.Snapshot(CompletionPartial)
	if len(u.Raw) == 0 && u.InputTokens.Presence == CountAbsent && u.OutputTokens.Presence == CountAbsent {
		return
	}
	for _, field := range []struct {
		count CountObservation
		path  string
	}{{u.InputTokens, "input_tokens"}, {u.OutputTokens, "output_tokens"}, {u.TotalTokens, "total_tokens"}, {u.CacheReadTokens, "input_tokens_details.cached_tokens"}, {u.ReasoningTokens, "output_tokens_details.reasoning_tokens"}} {
		if field.count.Presence == CountAbsent {
			s.add(advisory("usage."+field.path, LossApproximated, "required target-wire placeholder; source count was not reported"))
		}
	}
}

func bridgeIndex(object map[string]any, key string) (int, bool) {
	n, ok := usageInteger(object[key])
	if !ok || n > int64(^uint(0)>>1) {
		return 0, false
	}
	return int(n), true
}

func bridgeSourceError(detail map[string]any) error {
	kind, _ := detail["type"].(string)
	if kind == "" {
		kind, _ = detail["code"].(string)
	}
	message, _ := detail["message"].(string)
	return &StreamError{Type: kind, Message: message}
}

func bridgeRequest(s *observedStream, request map[string]any) map[string]any {
	// Reuse the same finite envelope audit as request admission. This is
	// response-echo fidelity, not another provider/router decision.
	losses, unknown := structuredOutputFindings(ConversionResponsesToChat, request)
	for _, loss := range losses {
		loss.Path = "request." + loss.Path
		s.add(loss)
	}
	for _, path := range unknown {
		s.unknown("request." + path)
	}
	echo := map[string]any{}
	for key, value := range request {
		switch key {
		case "model", "input", "stream":
		case "force_api_support":
			// Already resolved by the prepared caller; never a second routing
			// decision and never a field to echo into a native response.
		case "client_metadata", "prompt_cache_key":
			if present(value) {
				s.add(material("request."+key, LossDropped, "source control has no request-aware response mapping"))
			}
		case "instructions", "max_output_tokens", "tools", "tool_choice", "parallel_tool_calls", "temperature", "top_p", "metadata", "text", "reasoning":
			echo[key] = value
		case "store", "background":
			if value != nil && value != false {
				s.add(material("request."+key, LossUnsupported, "a generated envelope cannot promise native stored/background execution"))
			}
		case "previous_response_id", "conversation", "include":
			if present(value) {
				s.add(material("request."+key, LossUnsupported, "native state/lifecycle cannot be fabricated by this bridge"))
			}
		default:
			if value != nil {
				s.unknown("request." + key)
			}
		}
	}
	encoded, err := json.Marshal(echo)
	if err != nil {
		s.add(material("request", LossUnsupported, "request-aware fields must be JSON values"))
		return nil
	}
	cloned, _ := decodeObservedObject(string(encoded))
	return cloned
}

func bridgeUsageValidity(s *observedStream) {
	u := s.usage.Snapshot(CompletionPartial)
	for _, c := range []CountObservation{u.InputTokens, u.OutputTokens, u.TotalTokens, u.CacheReadTokens, u.CacheWriteTokens, u.ReasoningTokens} {
		if c.Presence == CountInvalid {
			s.add(material("usage."+c.Path, LossUnsupported, "invalid source usage is not a measured zero"))
		}
	}
}

func bridgeResponsesUsage(u UsageObservation) any {
	for _, count := range []CountObservation{u.InputTokens, u.OutputTokens, u.TotalTokens, u.CacheReadTokens, u.CacheWriteTokens, u.ReasoningTokens} {
		if count.Presence == CountInvalid {
			return nil
		}
	}
	if len(u.Raw) == 0 && u.InputTokens.Presence == CountAbsent && u.OutputTokens.Presence == CountAbsent {
		return nil
	}
	count := func(c CountObservation) int64 {
		if c.Presence == CountPresent {
			return c.Value
		}
		return 0
	}
	input := map[string]any{"cached_tokens": count(u.CacheReadTokens)}
	if u.CacheWriteTokens.Presence == CountPresent {
		input["cache_write_tokens"] = u.CacheWriteTokens.Value
	}
	return map[string]any{"input_tokens": count(u.InputTokens), "output_tokens": count(u.OutputTokens), "total_tokens": count(u.TotalTokens), "input_tokens_details": input, "output_tokens_details": map[string]any{"reasoning_tokens": count(u.ReasoningTokens)}}
}
