package translate

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestFileInputsRoundTripChatResponses(t *testing.T) {
	for _, file := range []map[string]any{
		{"filename": "fixture.pdf", "file_data": "data:application/pdf;base64,AQID"},
		{"filename": "fixture.pdf", "file_data": "AQID"},
		{"file_id": "file_fixture"},
	} {
		t.Run(jsonString(file), func(t *testing.T) {
			messages := []map[string]any{{"role": "user", "content": []any{
				map[string]any{"type": "text", "text": "before"},
				map[string]any{"type": "file", "file": file},
				map[string]any{"type": "text", "text": "after"},
			}}}
			payload := map[string]any{"messages": []any{messages[0]}}
			if a := AssessConversion(ConversionChatToResponses, PhaseRequest, payload, FidelityRequirements{}); a.Feasibility != FeasibilitySupported {
				t.Fatalf("file request refused: %+v", a)
			}
			forward := ChatToResponsesWithReport("fixture", messages, nil, false)
			if forward.Report.HasMaterialLoss() {
				t.Fatalf("file lost: %+v", forward.Report)
			}
			content := forward.Value["input"].([]any)[0].(map[string]any)["content"].([]any)
			part := content[1].(map[string]any)
			if part["type"] != "input_file" || len(part) != len(file)+1 {
				t.Fatalf("wrong Responses file shape: %s", jsonString(content))
			}
			for key, value := range file {
				if part[key] != value {
					t.Fatalf("file field %s changed", key)
				}
			}
			if a := AssessConversion(ConversionResponsesToChat, PhaseRequest, forward.Value, FidelityRequirements{}); a.Feasibility != FeasibilitySupported {
				t.Fatalf("reverse file request refused: %+v", a)
			}
			back, err := ResponsesRequestToChatWithReport(forward.Value)
			if err != nil || back.Report.HasMaterialLoss() || !reflect.DeepEqual(back.Value.Messages, messages) {
				t.Fatalf("round trip=%s, report=%+v, error=%v", jsonString(back.Value.Messages), back.Report, err)
			}
		})
	}
}

func TestFileInputAssessmentRetainsUnknownAndUnsupportedPaths(t *testing.T) {
	for _, tc := range []struct {
		name, body, path string
		id               ConversionID
	}{
		{"missing file", `{"messages":[{"role":"user","content":[{"type":"file"}]}]}`, "messages.0.content.0.file", ConversionChatToResponses},
		{"two file sources", `{"messages":[{"role":"user","content":[{"type":"file","file":{"file_id":"file_fixture","file_data":"AQID"}}]}]}`, "messages.0.content.0.file", ConversionChatToResponses},
		{"nested file control", `{"messages":[{"role":"user","content":[{"type":"file","file":{"file_data":"AQID","unmapped_control":true}}]}]}`, "messages.0.content.0.file.unmapped_control", ConversionChatToResponses},
		{"remote Responses file", `{"input":[{"role":"user","content":[{"type":"input_file","file_url":"https://example.invalid/fixture.pdf"}]}]}`, "input.0.content.0.file_url", ConversionResponsesToChat},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var body map[string]any
			if err := json.Unmarshal([]byte(tc.body), &body); err != nil {
				t.Fatal(err)
			}
			a := AssessConversion(tc.id, PhaseRequest, body, FidelityRequirements{})
			if a.Feasibility == FeasibilitySupported || !strings.Contains(jsonString(a), tc.path) {
				t.Fatalf("missing refusal path %s: %+v", tc.path, a)
			}
		})
	}
}

func TestMessagesPDFAndURLImagesToChat(t *testing.T) {
	var payload map[string]any
	_ = json.Unmarshal([]byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"image","source":{"type":"url","url":"https://example.invalid/fixture.png"}},{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"AQID"}},{"type":"text","text":"after"}]}]}`), &payload)
	a := AssessConversion(ConversionMessagesToChat, PhaseRequest, payload, FidelityRequirements{})
	if a.Feasibility != FeasibilitySupported {
		t.Fatalf("document/image source refused: %+v", a)
	}
	got := AnthropicRequestToOpenAIWithReport(payload)
	want := `[{"role":"user","content":[{"type":"text","text":"before"},{"type":"image_url","image_url":{"url":"https://example.invalid/fixture.png"}},{"type":"file","file":{"file_data":"data:application/pdf;base64,AQID"}},{"type":"text","text":"after"}]}]`
	var expected []map[string]any
	_ = json.Unmarshal([]byte(want), &expected)
	if got.Report.HasMaterialLoss() || !reflect.DeepEqual(got.Value.Messages, expected) {
		t.Fatalf("mapped=%s report=%+v", jsonString(got.Value.Messages), got.Report)
	}
	for _, control := range []string{"citations", "title", "context", "cache_control"} {
		document := payload["messages"].([]any)[0].(map[string]any)["content"].([]any)[2].(map[string]any)
		document[control] = map[string]any{"fixture": true}
		if a := AssessConversion(ConversionMessagesToChat, PhaseRequest, payload, FidelityRequirements{}); a.Feasibility == FeasibilitySupported {
			t.Fatalf("silently lost document %s", control)
		}
		delete(document, control)
	}
}
