package translate

import (
	"errors"
	"strings"
	"testing"
)

func TestObservedMessagesStreamLossBeforeEmissionAndTerminalUsage(t *testing.T) {
	lines := []string{
		`data: {"type":"message_start","message":{"usage":{"input_tokens":7,"cache_creation_input_tokens":13,"cache_read_input_tokens":11,"output_tokens":0}}}`,
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"thinking","thinking":"","signature":"signed-test"}}`,
		`data: {"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"t1","name":"lookup","input":{}}}`,
		`data: {"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`,
		`data: {"type":"message_stop"}`,
	}
	seenLoss := false
	emittedTool := false
	finalObserved := false
	result := AnthropicSSEToOpenAIChunksObserved(pullItems(lines), "m", func(raw string) {
		if strings.Contains(raw, `"tool_calls"`) {
			if !seenLoss {
				t.Fatal("tool output delivered before prior signed-state loss observation")
			}
			emittedTool = true
		}
	}, StreamObservationOptions{IncludeUsage: true, Observe: func(o StreamObservation) {
		if loss, ok := lossAt(o.Assessment.Report, "content.2.signature"); ok && loss.Severity == LossMaterial {
			seenLoss = true
		}
		if o.Assessment.Phase == PhaseFinal {
			finalObserved = true
		}
	}})
	if !seenLoss || !emittedTool || !finalObserved || !result.Completed || result.Err != nil || result.Usage.TotalTokens.Value != 36 || result.Usage.Quality != CompletionComplete {
		t.Fatalf("combined result=%+v", result)
	}
}

func TestObservedMessagesStreamTruncatedAndError(t *testing.T) {
	for _, tail := range []string{"", `data: {"type":"error","error":{"type":"overloaded_error","message":"synthetic"}}`} {
		lines := []string{`data: {"type":"message_start","message":{"usage":{"input_tokens":7,"output_tokens":0}}}`, `data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"partial"}}`}
		if tail != "" {
			lines = append(lines, tail)
		}
		var output []string
		result := AnthropicSSEToOpenAIChunksObserved(pullItems(lines), "m", func(s string) { output = append(output, s) }, StreamObservationOptions{IncludeUsage: true})
		if result.Completed || result.Err == nil || result.Usage.InputTokens.Presence == CountPresent {
			t.Fatalf("invalid complete/source aggregate result=%+v", result)
		}
		if result.Usage.OutputTokens.Presence != CountPresent || result.Usage.OutputTokens.Value != 0 {
			t.Fatalf("zero lost on failure: %+v", result.Usage)
		}
		for _, raw := range output {
			if strings.Contains(raw, `"finish_reason":"`) {
				t.Fatalf("fabricated finish: %s", raw)
			}
		}
		if tail == "" && !errors.Is(result.Err, ErrStreamTruncated) {
			t.Fatalf("truncation error=%v", result.Err)
		}
	}
}

func TestObservedMessagesStreamFinalUsage(t *testing.T) {
	for _, tc := range []struct {
		name, startUsage, deltaUsage, wantPrompt, wantCompletion, wantTotal string
	}{
		{"known", `"input_tokens":2,"output_tokens":0`, `"output_tokens":1`, `"prompt_tokens":2`, `"completion_tokens":1`, `"total_tokens":3`},
		{"known zero", `"input_tokens":0,"output_tokens":0`, `"output_tokens":0`, `"prompt_tokens":0`, `"completion_tokens":0`, `"total_tokens":0`},
		{"absent input", `"output_tokens":0`, `"output_tokens":1`, "", "", ""},
		{"absent output", `"input_tokens":2,"output_tokens":0`, ``, "", "", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lines := []string{
				`data: {"type":"message_start","message":{"usage":{` + tc.startUsage + `}}}`,
			}
			if tc.deltaUsage != "" {
				lines = append(lines, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{`+tc.deltaUsage+`}}`)
			}
			lines = append(lines, `data: {"type":"message_stop"}`)
			var output []string
			result := AnthropicSSEToOpenAIChunksObserved(pullItems(lines), "m", func(s string) { output = append(output, s) }, StreamObservationOptions{IncludeUsage: true})
			if !result.Completed || result.Err != nil {
				t.Fatalf("stream did not complete: %+v", result)
			}
			joined := strings.Join(output, "")
			if tc.wantPrompt == "" {
				if strings.Contains(joined, `"usage":{"prompt_tokens"`) {
					t.Fatalf("final usage was fabricated from an unknown component: %s", joined)
				}
			} else if !strings.Contains(joined, tc.wantPrompt) || !strings.Contains(joined, tc.wantCompletion) || !strings.Contains(joined, tc.wantTotal) {
				t.Fatalf("final usage did not preserve known counts: %s", joined)
			}
		})
	}
}

func TestObservedChatStreamZeroUsageAndTermination(t *testing.T) {
	for _, complete := range []bool{false, true} {
		chunks := []string{`{"choices":[{"index":0,"delta":{"content":"hello"}}]}`, `{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":0,"prompt_tokens_details":{"cached_tokens":4,"cache_write_tokens":3},"completion_tokens_details":{"reasoning_tokens":0}}}`}
		if complete {
			chunks = append(chunks, "[DONE]")
		}
		var output []string
		result := OpenAIStreamToAnthropicSSEObserved(pullItems(chunks), "m", func(s string) { output = append(output, s) }, StreamObservationOptions{})
		if result.Completed != complete || result.Usage.OutputTokens.Presence != CountPresent || result.Usage.OutputTokens.Value != 0 || result.Usage.CacheWriteTokens.Value != 3 {
			t.Fatalf("zero/terminal=%+v", result)
		}
		joined := strings.Join(output, "")
		if complete {
			if !strings.Contains(joined, "event: message_stop") || !strings.Contains(joined, `"output_tokens":0`) {
				t.Fatalf("measured zero/finish=%s", joined)
			}
		} else if strings.Contains(joined, "event: message_stop") {
			t.Fatalf("truncation looked complete: %s", joined)
		}
	}
}

func TestObservedChatStreamLateInputUsage(t *testing.T) {
	for _, tc := range []struct {
		name, usage, wantInput string
	}{
		{"known", `"prompt_tokens":3,"completion_tokens":4,"total_tokens":7`, `"input_tokens":3`},
		{"known zero", `"prompt_tokens":0,"completion_tokens":4,"total_tokens":4`, `"input_tokens":0`},
		{"absent", `"completion_tokens":4`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			chunks := []string{
				`{"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`,
				`{"choices":[{"index":0,"delta":{"content":"OK"}}]}`,
				`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{` + tc.usage + `}}`,
				"[DONE]",
			}
			var output []string
			result := OpenAIStreamToAnthropicSSEObserved(pullItems(chunks), "m", func(s string) { output = append(output, s) }, StreamObservationOptions{})
			if !result.Completed || result.Err != nil {
				t.Fatalf("stream did not complete: %+v", result)
			}
			var terminal string
			for _, event := range output {
				if strings.HasPrefix(event, "event: message_delta\n") {
					terminal = event
				}
			}
			if !strings.Contains(terminal, `"output_tokens":4`) {
				t.Fatalf("terminal output usage changed: %s", terminal)
			}
			if tc.wantInput == "" {
				if strings.Contains(terminal, `"input_tokens"`) {
					t.Fatalf("missing source input was fabricated: %s", terminal)
				}
			} else if !strings.Contains(terminal, tc.wantInput) {
				t.Fatalf("late source input was not preserved: %s", terminal)
			}
		})
	}
}

func TestObservedStreamsPreserveLargeSourceIntegers(t *testing.T) {
	const want int64 = 9007199254740993
	chat := OpenAIStreamToAnthropicSSEObserved(pullItems([]string{`{"choices":[],"usage":{"prompt_tokens":9007199254740993,"completion_tokens":0}}`, "[DONE]"}), "m", func(string) {}, StreamObservationOptions{})
	if chat.Usage.InputTokens.Presence != CountPresent || chat.Usage.InputTokens.Value != want {
		t.Fatalf("Chat source float-rounded: %+v", chat.Usage)
	}
	var output []string
	messages := AnthropicSSEToOpenAIChunksObserved(pullItems([]string{`data: {"type":"message_start","message":{"usage":{"input_tokens":9007199254740993,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}}`, `data: {"type":"message_stop"}`}), "m", func(raw string) { output = append(output, raw) }, StreamObservationOptions{IncludeUsage: true})
	if messages.Usage.InputTokens.Presence != CountPresent || messages.Usage.InputTokens.Value != want {
		t.Fatalf("Messages source float-rounded: %+v", messages.Usage)
	}
	found := false
	for _, raw := range output {
		if strings.Contains(raw, `"prompt_tokens":9007199254740993`) {
			found = true
		}
	}
	if !found {
		t.Fatalf("translated terminal usage did not use same exact extraction: %v", output)
	}
}

func TestObservedStreamDroppedRefusalAndCitationAreExplicit(t *testing.T) {
	t.Run("Chat refusal", func(t *testing.T) {
		observed := false
		result := OpenAIStreamToAnthropicSSEObserved(pullItems([]string{`{"choices":[{"index":0,"delta":{"refusal":"cannot comply"}}]}`, "[DONE]"}), "m", func(raw string) {
			if strings.Contains(raw, "event: message_stop") && !observed {
				t.Fatal("terminal emitted before refusal loss observation")
			}
		}, StreamObservationOptions{Observe: func(o StreamObservation) {
			if _, ok := lossAt(o.Assessment.Report, "choices.0.delta.refusal"); ok {
				observed = true
			}
		}})
		if !observed || result.Assessment.Feasibility != FeasibilityUnsupported {
			t.Fatalf("refusal silently disappeared: %+v", result)
		}
	})
	t.Run("Messages citation delta", func(t *testing.T) {
		observed := false
		result := AnthropicSSEToOpenAIChunksObserved(pullItems([]string{`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`, `data: {"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"char_location","start_char_index":0,"end_char_index":1}}}`, `data: {"type":"message_stop"}`}), "m", func(raw string) {
			if strings.Contains(raw, `"finish_reason":"`) && !observed {
				t.Fatal("terminal emitted before citation loss observation")
			}
		}, StreamObservationOptions{Observe: func(o StreamObservation) {
			if _, ok := lossAt(o.Assessment.Report, "content.0.delta.citation"); ok {
				observed = true
			}
		}})
		if !observed || result.Assessment.Feasibility != FeasibilityUnsupported {
			t.Fatalf("citation silently disappeared: %+v", result)
		}
	})
}

func TestObservedStreamUnknownPathsUseNativeBlockIndex(t *testing.T) {
	seen := false
	result := AnthropicSSEToOpenAIChunksObserved(pullItems([]string{
		`data: {"type":"content_block_start","index":3,"content_block":{"type":"text","text":"answer","citations":[{"type":"char_location"}]}}`,
		`data: {"type":"message_stop"}`,
	}), "m", func(string) {}, StreamObservationOptions{Observe: func(o StreamObservation) {
		for _, path := range o.Assessment.UnknownPaths {
			if path == "content.0.citations" {
				t.Fatal("unknown path retained synthetic block zero")
			}
			if path == "content.3.citations" {
				seen = true
			}
		}
	}})
	if !seen || result.Assessment.Feasibility != FeasibilityUnknown {
		t.Fatalf("native index unknown observation=%+v", result.Assessment)
	}
}
