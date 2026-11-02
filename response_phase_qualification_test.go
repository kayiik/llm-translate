package translate

import (
	"encoding/json"
	"strings"
	"testing"
)

func finalAnswerFixture(phase, status string) map[string]any {
	return map[string]any{
		"id": "fixture-response", "object": "response", "status": status,
		"output": []any{map[string]any{
			"id": "fixture-message", "type": "message", "role": "assistant", "status": "completed", "phase": phase,
			"content": []any{map[string]any{"type": "output_text", "text": "CAPTURE_OK", "annotations": []any{}}},
		}},
		"usage": map[string]any{"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
	}
}

func TestFinalAnswerDisplayRejectsAdditionalPhasedMessagesAndNonemptyLogprobs(t *testing.T) {
	response := finalAnswerFixture("final_answer", "completed")
	item := response["output"].([]any)[0].(map[string]any)
	part := item["content"].([]any)[0].(map[string]any)
	part["logprobs"] = []any{}
	if got := AssessConversion(ConversionChatToResponses, PhaseResponse, response, FidelityRequirements{}); got.Feasibility != FeasibilitySupported {
		t.Fatalf("empty optional log probabilities rejected: %+v", got)
	}
	part["logprobs"] = []any{map[string]any{"token": "CAPTURE_OK", "logprob": -0.1}}
	if got := AssessConversion(ConversionChatToResponses, PhaseResponse, response, FidelityRequirements{}); got.Feasibility == FeasibilitySupported {
		t.Fatalf("nonempty log probabilities disappeared: %+v", got)
	}
	delete(part, "logprobs")
	response["output"] = append(response["output"].([]any), map[string]any{
		"type": "message", "role": "assistant", "status": "completed", "phase": "commentary",
		"content": []any{map[string]any{"type": "output_text", "text": "unrelated"}},
	})
	if got := AssessConversion(ConversionChatToResponses, PhaseResponse, response, FidelityRequirements{}); got.Feasibility == FeasibilitySupported {
		t.Fatalf("multiple phased messages were collapsed into a final answer: %+v", got)
	}
}

func TestResponsesStreamModerationMetadataIsOnlyDisplayAdvisoryWhenUnblocked(t *testing.T) {
	metadata := map[string]any{"openai_chatgpt_moderation_metadata": map[string]any{
		"generation": map[string]any{}, "tool_call": map[string]any{}, "tool_response": map[string]any{},
		"prompt": map[string]any{"omnimod": map[string]any{
			"protection_type": "fixture", "outputs": []any{map[string]any{"is_blocked": false, "results": []any{}}},
		}},
	}}
	for _, tc := range []struct {
		name    string
		change  func()
		allowed bool
	}{
		{"unblocked", func() {}, true},
		{"blocked", func() {
			metadata["openai_chatgpt_moderation_metadata"].(map[string]any)["prompt"].(map[string]any)["omnimod"].(map[string]any)["outputs"].([]any)[0].(map[string]any)["is_blocked"] = true
		}, false},
		{"unknown extra field", func() { metadata["extra"] = "future" }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			metadata["openai_chatgpt_moderation_metadata"].(map[string]any)["prompt"].(map[string]any)["omnimod"].(map[string]any)["outputs"].([]any)[0].(map[string]any)["is_blocked"] = false
			delete(metadata, "extra")
			tc.change()
			raw, err := json.Marshal(map[string]any{"type": "response.metadata", "metadata": metadata})
			if err != nil {
				t.Fatal(err)
			}
			events := []string{
				`{"type":"response.created","response":{"id":"fixture","object":"response","status":"in_progress","output":[]}}`,
				string(raw),
				`{"type":"response.completed","response":{"id":"fixture","object":"response","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":0,"total_tokens":1}}}`,
			}
			result := ResponsesSSEToOpenAIChunksObserved(pullItems(events), "fixture", func(string) {}, StreamObservationOptions{})
			if result.Completed != tc.allowed || (result.Err == nil) != tc.allowed {
				t.Fatalf("moderation %s: completed=%t err=%v, want supported=%t", tc.name, result.Completed, result.Err, tc.allowed)
			}
		})
	}
}

func TestCompletedFinalAnswerPhaseIsDisplayOnlyButNotReplayable(t *testing.T) {
	response := finalAnswerFixture("final_answer", "completed")
	assessment := AssessConversion(ConversionChatToResponses, PhaseResponse, response, FidelityRequirements{})
	if assessment.Feasibility != FeasibilitySupported || assessment.Report.HasMaterialLoss() {
		t.Fatalf("final Chat display should be supported: %+v", assessment)
	}
	if got := ResponsesToChatWithReport("fixture", response).Value["choices"]; got == nil {
		t.Fatal("final Chat display has no choice")
	}
	for _, tc := range []struct {
		name         string
		phase, state string
		replay       bool
	}{
		{"commentary", "commentary", "completed", false},
		{"future phase", "future_phase", "completed", false},
		{"incomplete final", "final_answer", "incomplete", false},
		{"native replay", "final_answer", "completed", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := AssessConversion(ConversionChatToResponses, PhaseResponse, finalAnswerFixture(tc.phase, tc.state), FidelityRequirements{Replay: tc.replay})
			if got.Feasibility == FeasibilitySupported {
				t.Fatalf("lost phase was accepted: %+v", got)
			}
		})
	}
}

func TestCompletedProviderResponseMetadataDoesNotHideMaterialOutput(t *testing.T) {
	response := finalAnswerFixture("final_answer", "completed")
	response["completed_at"] = 12
	response["frequency_penalty"] = 0
	response["presence_penalty"] = 0
	response["prompt_cache_key"] = "synthetic-cache"
	response["prompt_cache_retention"] = "24h"
	response["safety_identifier"] = "synthetic-id"
	response["tool_usage"] = map[string]any{"web_search": map[string]any{"num_requests": 0}}
	response["copilot_usage"] = map[string]any{"token_details": []any{map[string]any{"token_count": 3}}}
	if a := AssessConversion(ConversionChatToResponses, PhaseResponse, response, FidelityRequirements{}); a.Feasibility != FeasibilitySupported {
		t.Fatalf("provider's display metadata rejected: %+v", a)
	}
	response["tool_usage"] = map[string]any{"web_search": map[string]any{"num_requests": 1}}
	if a := AssessConversion(ConversionChatToResponses, PhaseResponse, response, FidelityRequirements{}); a.Feasibility == FeasibilitySupported {
		t.Fatalf("nonempty tool usage disappeared: %+v", a)
	}
}

func TestFinalAnswerPhaseOnResponsesStreamDoesNotDropText(t *testing.T) {
	input := []string{
		`{"type":"response.created","response":{"id":"fixture-response","object":"response","status":"in_progress","output":[]}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"fixture-message","type":"message","role":"assistant","status":"in_progress","phase":"final_answer","content":[]}}`,
		`{"type":"response.content_part.added","output_index":0,"content_index":0,"item_id":"fixture-message","part":{"type":"output_text","text":"","annotations":[]}}`,
		`{"type":"response.output_text.delta","output_index":0,"content_index":0,"item_id":"fixture-message","delta":"CAPTURE_OK"}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"fixture-message","type":"message","role":"assistant","status":"completed","phase":"final_answer","content":[{"type":"output_text","text":"CAPTURE_OK","annotations":[]}]}}`,
		`{"type":"response.completed","response":{"id":"fixture-response","object":"response","status":"completed","output":[{"id":"fixture-message","type":"message","role":"assistant","status":"completed","phase":"final_answer","content":[{"type":"output_text","text":"CAPTURE_OK","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}`,
	}
	var output []string
	result := ResponsesSSEToOpenAIChunksObserved(pullItems(input), "fixture", func(value string) { output = append(output, value) }, StreamObservationOptions{IncludeUsage: true})
	if result.Err != nil || !result.Completed || !strings.Contains(strings.Join(output, ""), "CAPTURE_OK") {
		t.Fatalf("final-answer stream failed: err=%v completed=%t chunks=%d", result.Err, result.Completed, len(output))
	}
}

func TestChatToResponsesStreamUsageIsSatisfiedAtTheTerminal(t *testing.T) {
	for _, test := range []struct {
		name   string
		option any
		want   bool
	}{
		{"include final usage", map[string]any{"include_usage": true}, true},
		{"no optional usage", map[string]any{"include_usage": false}, true},
		{"unknown option", map[string]any{"include_usage": true, "future": true}, false},
		{"invalid option", map[string]any{"include_usage": "true"}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			body := map[string]any{"model": "fixture", "stream": true, "stream_options": test.option, "messages": []any{map[string]any{"role": "user", "content": "hi"}}}
			a := AssessConversion(ConversionChatToResponses, PhaseRequest, body, FidelityRequirements{})
			if supported := a.Feasibility == FeasibilitySupported; supported != test.want {
				t.Fatalf("stream usage option assessment=%+v, want support=%t", a, test.want)
			}
		})
	}
}
