package translate

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
)

var ErrChatStreamTruncated = protocolErrorf("Chat", "stream ended without [DONE]")

type StreamObservationOptions struct {
	IncludeUsage bool
	Requirements FidelityRequirements
	Observe      func(StreamObservation)
}

type StreamObservation struct {
	Assessment Assessment
	Usage      UsageObservation
	Completed  bool
	Err        error
}

func AnthropicSSEToOpenAIChunksObserved(lines func() (string, bool), model string, emit func(string), opts StreamObservationOptions) StreamObservation {
	s := newObservedStream(ConversionChatToMessages, UsageDialectMessages, opts)
	read := func() (string, bool) {
		raw, ok := lines()
		if !ok {
			return raw, false
		}
		if data, ok := strings.CutPrefix(strings.TrimSpace(raw), "data:"); ok {
			if event, ok := decodeObservedObject(strings.TrimSpace(data)); ok {
				switch event["type"] {
				case "message_start":
					message, _ := asMap(event["message"])
					s.usage.Merge(message["usage"])
				case "message_delta":
					s.usage.Merge(event["usage"])
				case "content_block_start":
					block, _ := asMap(event["content_block"])
					a := AssessConversion(ConversionChatToMessages, PhaseResponse, map[string]any{"content": []any{block}}, s.assessment.Requirements)
					for i := range a.Report.Losses {
						a.Report.Losses[i].Path = strings.Replace(a.Report.Losses[i].Path, "content.0", fmt.Sprintf("content.%d", toInt(event["index"])), 1)
					}
					for i := range a.UnknownPaths {
						a.UnknownPaths[i] = strings.Replace(a.UnknownPaths[i], "content.0", fmt.Sprintf("content.%d", toInt(event["index"])), 1)
					}
					s.merge(a)
				case "content_block_delta":
					delta, _ := asMap(event["delta"])
					path := fmt.Sprintf("content.%d.delta", toInt(event["index"]))
					switch delta["type"] {
					case "text_delta":
						for key := range delta {
							if key != "type" && key != "text" {
								s.unknown(path + "." + key)
							}
						}
					case "input_json_delta":
						for key := range delta {
							if key != "type" && key != "partial_json" {
								s.unknown(path + "." + key)
							}
						}
					case "thinking_delta":
						s.add(advisory(fmt.Sprintf("content.%d.reasoning", toInt(event["index"])), LossDropped, "display thinking delta is not emitted by Chat"))
					case "signature_delta":
					case "citations_delta":
						s.add(material(path+".citation", LossDropped, "citation delta is not emitted by Chat"))
					default:
						s.unknown(path + ".type")
					}
					if delta["type"] == "signature_delta" && present(delta["signature"]) {
						s.assessment.Requirements.Replay = true
						s.add(material(fmt.Sprintf("content.%d.signature", toInt(event["index"])), LossDropped, "signed continuation state is not emitted by Chat"))
					}
				case "message_stop", "content_block_stop", "ping", "error":
				default:
					s.unknown("$event.type")
				}
				s.notify(PhaseStreamEvent, CompletionPartial)
			}
		}
		return raw, true
	}
	var id string
	forward := func(raw string) {
		if body, ok := decodeObservedObject(raw); ok {
			if value, _ := body["id"].(string); value != "" {
				id = value
			}
			if opts.IncludeUsage {
				body["usage"] = nil
				if encoded, err := json.Marshal(body); err == nil {
					raw = string(encoded)
				}
			}
		}
		emit(raw)
	}
	// Reuse lifecycle-safe text/tool rendering, but project terminal usage
	// from the same exact extraction used by the native source observation.
	terminal := AnthropicSSEToOpenAIChunksWithOptions(read, model, forward, StreamOptions{})
	s.result.Completed = terminal.Completed
	s.result.Err = terminal.Err
	quality := CompletionComplete
	if !terminal.Completed {
		quality = CompletionFailed
		if errors.Is(terminal.Err, ErrStreamTruncated) {
			quality = CompletionTruncated
		}
	}
	if terminal.Completed && opts.IncludeUsage {
		u := s.usage.Snapshot(quality)
		if allPresent(u.InputTokens, u.OutputTokens, u.TotalTokens) {
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
			body, _ := json.Marshal(map[string]any{"id": id, "object": "chat.completion.chunk", "model": model, "choices": []any{}, "usage": usage})
			s.notify(PhaseStreamEvent, quality)
			emit(string(body))
		} else {
			s.add(advisory("usage", LossDropped, "complete source usage is unavailable; no final usage chunk is fabricated"))
		}
	}
	return s.finish(quality)
}

func OpenAIStreamToAnthropicSSEObserved(chunks func() (string, bool), model string, emit func(string), opts StreamObservationOptions) StreamObservation {
	s := newObservedStream(ConversionMessagesToChat, UsageDialectChat, opts)
	sawText := false
	read := func() (string, bool) {
		raw, ok := chunks()
		if !ok {
			if s.result.Err == nil && !s.result.Completed {
				s.result.Err = ErrChatStreamTruncated
			}
			return raw, false
		}
		if strings.TrimSpace(raw) == "[DONE]" {
			s.result.Completed = s.result.Err == nil
			return "", false
		}
		chunk, valid := decodeObservedObject(raw)
		if !valid {
			s.result.Err = protocolErrorf("Chat", "stream contains invalid JSON")
			return "", false
		}
		if detail, ok := asMap(chunk["error"]); ok {
			kind, _ := asStr(detail["type"])
			message, _ := asStr(detail["message"])
			s.result.Err = &StreamError{Type: kind, Message: message}
			s.notify(PhaseStreamEvent, CompletionFailed)
			return "", false
		}
		s.usage.Merge(chunk["usage"])
		for _, rawChoice := range mustList(chunk["choices"]) {
			choice, _ := asMap(rawChoice)
			delta, _ := asMap(choice["delta"])
			if text, _ := asStr(delta["content"]); text != "" {
				sawText = true
			}
			prefix := fmt.Sprintf("choices.%d.delta", toInt(choice["index"]))
			for key, value := range delta {
				if value == nil {
					continue
				}
				switch key {
				case "role", "content", "tool_calls", "reasoning_content", "reasoning_details":
				case "refusal", "audio", "function_call":
					s.add(material(prefix+"."+key, LossDropped, "Chat delta field is not emitted by Messages"))
				default:
					s.unknown(prefix + "." + key)
				}
			}
			for _, loss := range deltaVendorLosses(choice, delta) {
				s.add(loss)
			}
			walkFidelity(delta, fmt.Sprintf("choices.%d.delta", toInt(choice["index"])), func(path, key string, value any, object map[string]any) {
				if replayField(key, object) && present(value) {
					s.assessment.Requirements.Replay = true
					s.add(material(path, LossDropped, "opaque continuation state is not emitted by Messages"))
				}
			})
		}
		s.notify(PhaseStreamEvent, CompletionPartial)
		return raw, true
	}
	forward := func(event string) {
		// The legacy converter emits closure on exhaustion; the combined API
		// only emits it after terminal evidence. Prior bytes are never replayed.
		if s.result.Err != nil {
			return
		}
		if strings.HasPrefix(event, "event: message_delta\n") {
			usage := s.usage.Snapshot(CompletionPartial)
			if usage.OutputTokens.Presence == CountPresent {
				if _, data, ok := strings.Cut(event, "data: "); ok {
					var body map[string]any
					if json.Unmarshal([]byte(strings.TrimSpace(data)), &body) == nil {
						body["usage"] = map[string]any{"output_tokens": usage.OutputTokens.Value}
						event = sse("message_delta", body)
					}
				}
			}
		}
		emit(event)
	}
	OpenAIStreamToAnthropicSSE(read, model, forward)
	if sawText && s.usage.Snapshot(CompletionPartial).OutputTokens.Presence != CountPresent {
		s.add(advisory("usage.output_tokens", LossApproximated, "wire output tokens are estimated; source usage was not reported"))
	}
	quality := CompletionComplete
	if !s.result.Completed {
		quality = CompletionFailed
		if errors.Is(s.result.Err, ErrChatStreamTruncated) {
			quality = CompletionTruncated
		}
	}
	return s.finish(quality)
}

func decodeObservedObject(raw string) (map[string]any, bool) {
	var object map[string]any
	decoder := json.NewDecoder(strings.NewReader(raw))
	decoder.UseNumber()
	if decoder.Decode(&object) != nil || object == nil {
		return nil, false
	}
	var tail any
	if decoder.Decode(&tail) != io.EOF {
		return nil, false
	}
	return object, true
}

type observedStream struct {
	assessment Assessment
	usage      *UsageAccumulator
	result     StreamObservation
	observer   func(StreamObservation)
}

func newObservedStream(id ConversionID, dialect UsageDialect, opts StreamObservationOptions) *observedStream {
	return &observedStream{assessment: Assessment{Conversion: id, Phase: PhaseStreamEvent, Feasibility: FeasibilitySupported, Requirements: opts.Requirements, Evaluated: []FidelityFeature{FidelityContent}}, usage: NewUsageAccumulator(dialect), observer: opts.Observe}
}

func (s *observedStream) add(loss Loss) {
	s.assessment.Report = NewReport(append(s.assessment.Report.Losses, loss)...)
	if loss.Severity == LossMaterial || loss.Class == LossUnsupported {
		s.assessment.Feasibility = FeasibilityUnsupported
	}
}

func (s *observedStream) unknown(path string) {
	s.assessment.UnknownPaths = append(s.assessment.UnknownPaths, path)
	sort.Strings(s.assessment.UnknownPaths)
	s.assessment.UnknownPaths = uniquePaths(s.assessment.UnknownPaths)
	if s.assessment.Feasibility != FeasibilityUnsupported {
		s.assessment.Feasibility = FeasibilityUnknown
	}
}

func (s *observedStream) merge(a Assessment) {
	for _, l := range a.Report.Losses {
		s.add(l)
	}
	r := s.assessment.Requirements
	s.assessment.Requirements = FidelityRequirements{Content: r.Content || a.Requirements.Content, Tools: r.Tools || a.Requirements.Tools, Schema: r.Schema || a.Requirements.Schema, Billing: r.Billing || a.Requirements.Billing, Replay: r.Replay || a.Requirements.Replay}
	for _, feature := range a.Evaluated {
		found := false
		for _, existing := range s.assessment.Evaluated {
			if feature == existing {
				found = true
			}
		}
		if !found {
			s.assessment.Evaluated = append(s.assessment.Evaluated, feature)
		}
	}
	if len(a.UnknownPaths) > 0 {
		for _, path := range a.UnknownPaths {
			s.unknown(path)
		}
	}
}

func (s *observedStream) snapshot(phase ObservationPhase, quality CompletionQuality) StreamObservation {
	a := s.assessment
	a.Phase = phase
	a.Report = NewReport(a.Report.Losses...)
	a.UnknownPaths = append([]string(nil), a.UnknownPaths...)
	a.Evaluated = append([]FidelityFeature(nil), a.Evaluated...)
	return StreamObservation{Assessment: a, Usage: s.usage.Snapshot(quality), Completed: s.result.Completed, Err: s.result.Err}
}

func (s *observedStream) notify(phase ObservationPhase, quality CompletionQuality) {
	if s.observer != nil {
		s.observer(s.snapshot(phase, quality))
	}
}

func (s *observedStream) finish(quality CompletionQuality) StreamObservation {
	s.notify(PhaseFinal, quality)
	return s.snapshot(PhaseFinal, quality)
}
