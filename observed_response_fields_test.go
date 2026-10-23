package translate

import (
	"encoding/json"
	"testing"
)

func TestObservedChatResponseAnnotations(t *testing.T) {
	body := decodeJSON[map[string]any](t, `{"id":"chat_fixture","object":"chat.completion","model":"fixture","provider":"fixture-router","choices":[{"index":0,"message":{"role":"assistant","content":"Hello","reasoning":"Consider the greeting."},"finish_reason":"stop","native_finish_reason":"COMPLETE"}]}`)
	before, _ := json.Marshal(body)
	a := AssessConversion(ConversionMessagesToChat, PhaseResponse, body, FidelityRequirements{})
	if a.Feasibility != FeasibilitySupported || len(a.UnknownPaths) != 0 || a.Report.HasMaterialLoss() {
		t.Fatalf("ordinary annotated Chat response refused: %+v", a)
	}
	result := OpenAIResponseToAnthropicWithReport(body, "fixture")
	requireLoss(t, result.Report, "choices.0.message.reasoning", LossAdvisory)
	if err := result.RejectMaterialLoss(); err != nil {
		t.Fatalf("optional reasoning annotation rejected: %v", err)
	}
	if result.Value["stop_reason"] != "end_turn" {
		t.Fatalf("standard finish semantics lost: %+v", result.Value)
	}
	if content, _ := json.Marshal(result.Value["content"]); string(content) != `[{"text":"Hello","type":"text"}]` {
		t.Fatalf("translated content=%s", content)
	}
	for _, field := range []string{"provider", "native_finish_reason", "reasoning"} {
		if _, ok := result.Value[field]; ok {
			t.Fatalf("source annotation emitted as destination root %s", field)
		}
	}
	after, _ := json.Marshal(body)
	if string(before) != string(after) {
		t.Fatal("source/native payload mutated")
	}
}

func TestObservedChatResponseAnnotationsRemainBounded(t *testing.T) {
	for _, tc := range []struct {
		name, body, path string
	}{
		{"provider object", `{"provider":{"route":"x"},"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`, "provider"},
		{"reasoning object", `{"choices":[{"message":{"content":"ok","reasoning":{"text":"opaque","signature":"state"}},"finish_reason":"stop"}]}`, "choices.0.message.reasoning"},
		{"reasoning number", `{"choices":[{"message":{"content":"ok","reasoning":42},"finish_reason":"stop"}]}`, "choices.0.message.reasoning"},
		{"native object", `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop","native_finish_reason":{"code":"x"}}]}`, "choices.0.native_finish_reason"},
		{"missing standard", `{"choices":[{"message":{"content":"ok"},"native_finish_reason":"COMPLETE"}]}`, "choices.0.native_finish_reason"},
		{"unknown standard", `{"choices":[{"message":{"content":"ok"},"finish_reason":"future","native_finish_reason":"COMPLETE"}]}`, "choices.0.native_finish_reason"},
		{"unknown extension", `{"choices":[{"message":{"content":"ok","future_reasoning":"x"},"finish_reason":"stop"}]}`, "choices.0.message.future_reasoning"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := AssessConversion(ConversionMessagesToChat, PhaseResponse, decodeJSON[map[string]any](t, tc.body), FidelityRequirements{})
			if a.Feasibility == FeasibilitySupported {
				t.Fatalf("unassessed shape accepted: %+v", a)
			}
			found := false
			for _, path := range a.UnknownPaths {
				found = found || path == tc.path
			}
			if !found {
				t.Fatalf("missing exact path %s: %+v", tc.path, a)
			}
		})
	}
	for _, raw := range []string{
		`{"choices":[{"message":{"content":[{"type":"text","text":"not mapped"}],"reasoning":"text"},"finish_reason":"stop"}],"provider":"x"}`,
		`{"choices":[{"message":{"content":"ok","reasoning":"text","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}","thought_signature":"opaque"}}]},"finish_reason":"tool_calls","native_finish_reason":"TOOL"}],"provider":"x"}`,
	} {
		a := AssessConversion(ConversionMessagesToChat, PhaseResponse, decodeJSON[map[string]any](t, raw), FidelityRequirements{})
		if a.Feasibility != FeasibilityUnsupported || !a.Report.HasMaterialLoss() {
			t.Fatalf("material content/replay loss waived: %+v", a)
		}
	}
	request := decodeJSON[map[string]any](t, `{"model":"fixture","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}`)
	if a := AssessConversion(ConversionChatToMessages, PhaseRequest, request, FidelityRequirements{}); a.Feasibility != FeasibilityUnsupported {
		t.Fatalf("requested reasoning control waived: %+v", a)
	}
}
