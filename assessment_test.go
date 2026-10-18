package translate

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestAssessmentOrdinaryEdgesAndUnknownScope(t *testing.T) {
	for _, tc := range []struct {
		edge ConversionID
		body string
	}{
		{ConversionChatToResponses, `{"messages":[{"role":"user","content":"hi"}]}`},
		{ConversionResponsesToChat, `{"input":"hi"}`},
		{ConversionChatToMessages, `{"messages":[{"role":"user","content":"hi"}]}`},
		{ConversionMessagesToChat, `{"messages":[{"role":"user","content":"hi"}]}`},
	} {
		t.Run(string(tc.edge), func(t *testing.T) {
			body := decodeJSON[map[string]any](t, tc.body)
			before, _ := json.Marshal(body)
			a := AssessConversion(tc.edge, PhaseRequest, body, FidelityRequirements{})
			if a.Feasibility != FeasibilitySupported || len(a.Evaluated) == 0 {
				t.Fatalf("ordinary assessment=%+v", a)
			}
			body["unreviewed_extension"] = map[string]any{"enabled": true}
			a = AssessConversion(tc.edge, PhaseRequest, body, FidelityRequirements{})
			want := FeasibilityUnsupported
			if tc.edge == ConversionChatToMessages {
				want = FeasibilityUnknown
			}
			if a.Feasibility != want || !reflect.DeepEqual(a.UnknownPaths, []string{"unreviewed_extension"}) {
				t.Fatalf("unknown scope=%+v", a)
			}
			delete(body, "unreviewed_extension")
			after, _ := json.Marshal(body)
			if string(before) != string(after) {
				t.Fatal("assessment mutated source")
			}
		})
	}
	a := AssessConversion("unimplemented", PhaseRequest, map[string]any{}, FidelityRequirements{})
	if a.Feasibility != FeasibilityUnknown {
		t.Fatalf("unknown edge=%+v", a)
	}
}

func TestAssessmentImageSourceScope(t *testing.T) {
	for _, raw := range []string{
		`{"messages":[{"role":"user","content":[{"type":"input_image","file_id":"file_fixture"}]}]}`,
		`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.invalid/image","unreviewed_control":true}}]}]}`,
	} {
		a := AssessConversion(ConversionChatToResponses, PhaseRequest, decodeJSON[map[string]any](t, raw), FidelityRequirements{})
		if a.Feasibility == FeasibilitySupported {
			t.Fatalf("unreviewed source became supported: %+v", a)
		}
	}
}

func TestAssessmentUnmappedNestedSemanticsAreNotSupported(t *testing.T) {
	for _, tc := range []struct {
		edge  ConversionID
		phase ObservationPhase
		body  string
		path  string
	}{
		{ConversionChatToResponses, PhaseRequest, `{"messages":[{"role":"user","name":"speaker","content":"hi"}]}`, "messages.0.name"},
		{ConversionChatToResponses, PhaseRequest, `{"messages":[{"role":"user","content":[{"type":"text","text":"hi","annotations":[{"type":"x"}]}]}]}`, "messages.0.content.0.annotations"},
		{ConversionChatToResponses, PhaseRequest, `{"messages":[{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}","unreviewed":true}}]},{"role":"tool","tool_call_id":"c1","content":"ok"}]}`, "messages.0.tool_calls.0.function.unreviewed"},
		{ConversionResponsesToChat, PhaseResponse, `{"choices":[{"message":{"content":[{"type":"text","text":"lost"}]}}]}`, "choices.0.message.content"},
		{ConversionChatToMessages, PhaseResponse, `{"content":[{"type":"text","text":"hi","citations":[{"type":"x"}]}]}`, "content.0.citations"},
	} {
		t.Run(tc.path, func(t *testing.T) {
			a := AssessConversion(tc.edge, tc.phase, decodeJSON[map[string]any](t, tc.body), FidelityRequirements{})
			if a.Feasibility == FeasibilitySupported {
				t.Fatalf("unmapped semantic supported: %+v", a)
			}
			found := false
			for _, path := range a.UnknownPaths {
				if path == tc.path {
					found = true
				}
			}
			if _, ok := lossAt(a.Report, tc.path); ok {
				found = true
			}
			if !found {
				t.Fatalf("missing exact evidence %s: %+v", tc.path, a)
			}
		})
	}
}

func TestAssessmentSchemaNamesAreNotReplayControls(t *testing.T) {
	body := decodeJSON[map[string]any](t, `{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object","properties":{"signature":{"type":"string"},"store":{"type":"boolean"}}}}}}`)
	a := AssessConversion(ConversionChatToResponses, PhaseRequest, body, FidelityRequirements{})
	if a.Requirements.Replay || a.Requirements.Billing || a.Feasibility != FeasibilitySupported {
		t.Fatalf("schema names treated as wire state: %+v", a)
	}
}

func TestAssessmentRequiredFutureGuaranteeRemainsUnknown(t *testing.T) {
	a := AssessConversion(ConversionChatToResponses, PhaseRequest, decodeJSON[map[string]any](t, `{"messages":[{"role":"user","content":"hi"}]}`), FidelityRequirements{Replay: true})
	if a.Feasibility != FeasibilityUnknown || !reflect.DeepEqual(a.UnknownPaths, []string{"$requirements.replay"}) {
		t.Fatalf("future native state preservation falsely guaranteed: %+v", a)
	}
}

func TestAssessmentDoesNotAuthorizeToolHistoryRepair(t *testing.T) {
	for _, body := range []string{
		`{"messages":[{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"f","arguments":"{}"}}]}]}`,
		`{"messages":[{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]}]}`,
	} {
		a := AssessConversion(ConversionChatToResponses, PhaseRequest, decodeJSON[map[string]any](t, body), FidelityRequirements{})
		if a.Feasibility != FeasibilityUnsupported {
			t.Fatalf("repair silently authorized: %+v", a)
		}
	}
}

func TestAssessmentReplayOpaqueEmptySummaryToolTurn(t *testing.T) {
	body := decodeJSON[map[string]any](t, `{"output":[{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque-test"},{"type":"message","role":"assistant","phase":"commentary","content":[]},{"type":"function_call","call_id":"c1","name":"lookup","arguments":"{}"}]}`)
	a := AssessConversion(ConversionChatToResponses, PhaseResponse, body, FidelityRequirements{})
	if !a.Requirements.Replay || a.Feasibility != FeasibilityUnsupported {
		t.Fatalf("opaque replay=%+v", a)
	}
	for _, path := range []string{"output.0.encrypted_content", "output.0.id", "output.1.phase"} {
		requireRecoveryLoss(t, a.Report, path, LossDropped)
	}
	// The stateless converter still refuses a following tool turn that needs
	// native reasoning history; no synthesized transcript or state ID bridge.
	next := decodeJSON[map[string]any](t, `{"input":[{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque-test"},{"type":"function_call_output","call_id":"c1","output":"ok"}]}`)
	if got := AssessConversion(ConversionResponsesToChat, PhaseRequest, next, FidelityRequirements{}); got.Feasibility != FeasibilityUnsupported {
		t.Fatalf("tool replay=%+v", got)
	}
	textOnly := decodeJSON[map[string]any](t, `{"output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"display"}]},{"type":"message","content":[{"type":"output_text","text":"answer"}]}]}`)
	if got := AssessConversion(ConversionChatToResponses, PhaseResponse, textOnly, FidelityRequirements{}); got.Feasibility != FeasibilitySupported || got.Report.HasMaterialLoss() {
		t.Fatalf("display summary=%+v", got)
	}
}

func TestAssessmentSignedThinkingAndRequirementsCannotWaiveLoss(t *testing.T) {
	body := decodeJSON[map[string]any](t, `{"content":[{"type":"thinking","thinking":"","signature":"signed-test"},{"type":"redacted_thinking","data":"redacted-test"},{"type":"tool_use","id":"t1","name":"lookup","input":{}}]}`)
	a := AssessConversion(ConversionChatToMessages, PhaseResponse, body, FidelityRequirements{})
	for _, path := range []string{"content.0.signature", "content.1.data"} {
		requireRecoveryLoss(t, a.Report, path, LossDropped)
	}
	if !a.Requirements.Replay || a.Feasibility != FeasibilityUnsupported {
		t.Fatalf("signed=%+v", a)
	}
	for _, input := range []string{
		`{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"x","strict":true,"schema":{"type":"object"}}}}`,
		`{"messages":[{"role":"user","content":"hi"}],"service_tier":"priority"}`,
	} {
		a = AssessConversion(ConversionChatToMessages, PhaseRequest, decodeJSON[map[string]any](t, input), FidelityRequirements{})
		if a.Feasibility != FeasibilityUnsupported {
			t.Fatalf("required semantics=%+v", a)
		}
	}
}
