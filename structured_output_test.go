package translate

import (
	"encoding/json"
	"reflect"
	"testing"
)

func structuredSchemaFixture() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"strict": map[string]any{"type": "boolean"},
			"schema": map[string]any{"$ref": "#/$defs/entry"},
			"tools":  map[string]any{"type": "array", "items": map[string]any{"type": "string", "enum": []any{"a", "b"}}},
			"phase":  map[string]any{"type": "string"},
			"字段":     map[string]any{"type": "string", "enum": []any{"é", "值"}},
		},
		"$defs":    map[string]any{"entry": map[string]any{"type": "object", "properties": map[string]any{"signature": map[string]any{"type": "integer", "minimum": json.Number("9007199254740993")}}, "required": []any{"signature"}, "additionalProperties": false}},
		"required": []any{"strict", "schema", "tools", "phase"}, "additionalProperties": false, "x-forwarded-vocabulary": map[string]any{"enum": []any{false, nil}},
	}
}

func TestStructuredOutputSupportedEnvelopeRoundTrip(t *testing.T) {
	for _, strict := range []any{nil, false, true} {
		format := map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": "Result_1", "description": "", "schema": structuredSchemaFixture()}}
		if strict != nil {
			format["json_schema"].(map[string]any)["strict"] = strict
		}
		original, err := json.Marshal(format)
		if err != nil {
			t.Fatal(err)
		}
		body := map[string]any{"messages": []any{map[string]any{"role": "user", "content": "hi"}}, "response_format": format}
		a := AssessConversion(ConversionChatToResponses, PhaseRequest, body, FidelityRequirements{})
		if a.Feasibility != FeasibilitySupported || a.Requirements.Replay || a.Requirements.Billing || a.Requirements.Tools {
			t.Fatalf("schema property names misclassified: %+v", a)
		}
		forward := ChatToResponsesWithReport("m", []map[string]any{{"role": "user", "content": "hi"}}, map[string]any{"response_format": format}, false)
		back, err := ResponsesRequestToChatWithReport(forward.Value)
		if err != nil || back.Report.HasMaterialLoss() || !reflect.DeepEqual(back.Value.Keywords["response_format"], format) {
			t.Fatalf("schema envelope changed: err=%v back=%#v report=%+v", err, back.Value.Keywords, back.Report)
		}
		after, _ := json.Marshal(format)
		roundtrip, _ := json.Marshal(back.Value.Keywords["response_format"])
		if string(original) != string(after) || string(original) != string(roundtrip) {
			t.Fatalf("schema controls/data mutated: before=%s source=%s roundtrip=%s", original, after, roundtrip)
		}
	}
	for _, text := range []any{nil, map[string]any{}, map[string]any{"format": nil}, map[string]any{"format": map[string]any{"type": "text"}}} {
		body := map[string]any{"input": "hi", "text": text}
		a := AssessConversion(ConversionResponsesToChat, PhaseRequest, body, FidelityRequirements{})
		if a.Feasibility != FeasibilitySupported {
			t.Fatalf("absent/empty text config rejected: %#v %+v", text, a)
		}
	}
	for _, format := range []any{nil, map[string]any{"type": "text"}, map[string]any{"type": "json_object"}} {
		forward := ChatToResponsesWithReport("m", []map[string]any{{"role": "user", "content": "hi"}}, map[string]any{"response_format": format}, false)
		if forward.Report.HasMaterialLoss() {
			t.Fatalf("supported mode rejected: %#v %+v", format, forward.Report)
		}
		if format == nil || format.(map[string]any)["type"] == "text" {
			if forward.Value["text"] != nil {
				t.Fatalf("default text asks for nothing: %#v", forward.Value)
			}
		} else {
			back, err := ResponsesRequestToChatWithReport(forward.Value)
			if err != nil || !reflect.DeepEqual(back.Value.Keywords["response_format"], format) {
				t.Fatalf("json_object mode changed: %+v %v", back, err)
			}
		}
	}
}

func TestStructuredOutputMalformedEnvelopeExactPaths(t *testing.T) {
	for _, tc := range []struct {
		edge       ConversionID
		body, path string
	}{
		{ConversionChatToResponses, `{"response_format":{"type":"json_schema","json_schema":{"schema":{}}}}`, "response_format.json_schema.name"},
		{ConversionChatToResponses, `{"response_format":{"type":"json_schema","json_schema":{"name":"x"}}}`, "response_format.json_schema.schema"},
		{ConversionChatToResponses, `{"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":true}}}`, "response_format.json_schema.schema"},
		{ConversionChatToResponses, `{"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{},"strict":"false"}}}`, "response_format.json_schema.strict"},
		{ConversionChatToResponses, `{"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{},"description":false}}}`, "response_format.json_schema.description"},
		{ConversionChatToResponses, `{"response_format":{}}`, "response_format.type"},
		{ConversionResponsesToChat, `{"text":{"format":{"type":"json_schema","name":"","schema":{}}}}`, "text.format.name"},
		{ConversionResponsesToChat, `{"text":{"format":{"type":"json_schema","name":"bad.name","schema":{}}}}`, "text.format.name"},
		{ConversionResponsesToChat, `{"text":{"format":{"type":"json_schema","name":"x","schema":null}}}`, "text.format.schema"},
		{ConversionResponsesToChat, `{"text":{"format":{"type":"json_schema","name":"x","schema":{},"strict":0}}}`, "text.format.strict"},
		{ConversionResponsesToChat, `{"text":false}`, "text"},
	} {
		t.Run(tc.path, func(t *testing.T) {
			body := decodeJSON[map[string]any](t, tc.body)
			if tc.edge == ConversionChatToResponses {
				body["messages"] = []any{map[string]any{"role": "user", "content": "hi"}}
			} else {
				body["input"] = "hi"
			}
			a := AssessConversion(tc.edge, PhaseRequest, body, FidelityRequirements{})
			if a.Feasibility != FeasibilityUnsupported {
				t.Fatalf("malformed envelope admitted: %+v", a)
			}
			requireRecoveryLoss(t, a.Report, tc.path, LossUnsupported)
		})
	}
}

func TestStructuredOutputUnknownNestedEnvelopePaths(t *testing.T) {
	for _, tc := range []struct {
		edge       ConversionID
		body, path string
	}{
		{ConversionChatToResponses, `{"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{},"vendor_control":null}}}`, "response_format.json_schema.vendor_control"},
		{ConversionResponsesToChat, `{"text":{"format":{"type":"json_schema","name":"x","schema":{},"vendor_control":false}}}`, "text.format.vendor_control"},
		{ConversionResponsesToChat, `{"text":{"format":{"type":"json_object"},"verbosity":"high"}}`, "text.verbosity"},
	} {
		body := decodeJSON[map[string]any](t, tc.body)
		body["input"] = "hi"
		if tc.edge == ConversionChatToResponses {
			delete(body, "input")
			body["messages"] = []any{map[string]any{"role": "user", "content": "hi"}}
		}
		a := AssessConversion(tc.edge, PhaseRequest, body, FidelityRequirements{})
		found := false
		for _, path := range a.UnknownPaths {
			if path == tc.path {
				found = true
			}
		}
		if a.Feasibility == FeasibilitySupported || !found {
			t.Fatalf("unknown envelope control not precise: %+v want=%s", a, tc.path)
		}
	}
}

func TestFunctionSchemaExplicitControlsAndDefaultBoundary(t *testing.T) {
	for _, strict := range []any{false, true} {
		fn := map[string]any{"name": "call_1", "description": "", "parameters": structuredSchemaFixture(), "strict": strict}
		original, _ := json.Marshal(fn)
		forward := ChatToResponsesWithReport("m", []map[string]any{{"role": "user", "content": "hi"}}, map[string]any{"tools": []any{map[string]any{"type": "function", "function": fn}}}, false)
		back, err := ResponsesRequestToChatWithReport(forward.Value)
		if err != nil || !reflect.DeepEqual(back.Value.Keywords["tools"], []any{map[string]any{"type": "function", "function": fn}}) {
			t.Fatalf("explicit tool schema/strict changed: %+v %v", back, err)
		}
		after, _ := json.Marshal(fn)
		if string(original) != string(after) {
			t.Fatalf("function schema source mutated: before=%s after=%s", original, after)
		}
	}
	for _, strict := range []any{nil, "absent"} {
		fn := map[string]any{"name": "call_1", "parameters": map[string]any{}}
		if strict == nil {
			fn["strict"] = nil
		}
		original, _ := json.Marshal(fn)
		body := map[string]any{"messages": []any{map[string]any{"role": "user", "content": "hi"}}, "tools": []any{map[string]any{"type": "function", "function": fn}}}
		forward := ChatToResponsesWithReport("m", []map[string]any{{"role": "user", "content": "hi"}}, map[string]any{"tools": body["tools"]}, false)
		tool := forward.Value["tools"].([]any)[0].(map[string]any)
		if value, exists := tool["strict"]; !exists || value != false {
			t.Fatalf("Chat non-strict default became Responses default: %v", tool)
		}
		after, _ := json.Marshal(fn)
		if string(original) != string(after) {
			t.Fatalf("implicit strict was falsely made a source field: before=%s after=%s", original, after)
		}
		a := AssessConversion(ConversionChatToResponses, PhaseRequest, body, FidelityRequirements{Schema: true})
		if a.Feasibility != FeasibilitySupported || !a.Requirements.Schema {
			t.Fatalf("empty schema/default not evaluated: %+v", a)
		}
		native := map[string]any{"input": "hi", "tools": []any{map[string]any{"type": "function", "name": "call_1", "parameters": map[string]any{}}}}
		if strict == nil {
			native["tools"].([]any)[0].(map[string]any)["strict"] = nil
		}
		a = AssessConversion(ConversionResponsesToChat, PhaseRequest, native, FidelityRequirements{})
		found := false
		for _, path := range a.UnknownPaths {
			if path == "tools.0.strict" {
				found = true
			}
		}
		if a.Feasibility != FeasibilityUnknown || !found {
			t.Fatalf("Responses implicit strict policy weakened: %+v", a)
		}
	}
}

func TestFunctionSchemaMalformedAndNullableParameters(t *testing.T) {
	for _, edge := range []ConversionID{ConversionChatToResponses, ConversionResponsesToChat} {
		for key, value := range map[string]any{"name": 0, "strict": "yes", "parameters": false, "description": 42} {
			fn := map[string]any{"name": "call_1", "parameters": map[string]any{}, "strict": false}
			fn[key] = value
			body := map[string]any{"input": "hi", "tools": []any{map[string]any{"type": "function", "name": fn["name"], "parameters": fn["parameters"], "strict": fn["strict"], "description": fn["description"]}}}
			path := "tools.0." + key
			if edge == ConversionChatToResponses {
				delete(body, "input")
				body["messages"] = []any{map[string]any{"role": "user", "content": "hi"}}
				body["tools"] = []any{map[string]any{"type": "function", "function": fn}}
				path = "tools.0.function." + key
			}
			a := AssessConversion(edge, PhaseRequest, body, FidelityRequirements{})
			if a.Feasibility != FeasibilityUnsupported {
				t.Fatalf("invalid tool accepted: %+v", a)
			}
			requireRecoveryLoss(t, a.Report, path, LossUnsupported)
		}
	}
	forward := ChatToResponsesWithReport("m", []map[string]any{{"role": "user", "content": "hi"}}, map[string]any{"tools": []any{map[string]any{"type": "function", "function": map[string]any{"name": "no_args"}}}}, false)
	tool := forward.Value["tools"].([]any)[0].(map[string]any)
	if value, exists := tool["parameters"]; !exists || value != nil {
		t.Fatalf("required nullable parameters missing: %v", tool)
	}
	back, err := ResponsesRequestToChatWithReport(forward.Value)
	if err != nil {
		t.Fatal(err)
	}
	fn := back.Value.Keywords["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if _, exists := fn["parameters"]; exists {
		t.Fatalf("optional Chat parameters became explicit null: %v", fn)
	}
	a := AssessConversion(ConversionResponsesToChat, PhaseRequest, forward.Value, FidelityRequirements{})
	if a.Feasibility != FeasibilitySupported {
		t.Fatalf("explicit nullable parameters and strict:false should map: %+v", a)
	}
	missing := decodeJSON[map[string]any](t, `{"input":"hi","tools":[{"type":"function","name":"f","strict":false}]}`)
	a = AssessConversion(ConversionResponsesToChat, PhaseRequest, missing, FidelityRequirements{})
	if a.Feasibility != FeasibilityUnsupported {
		t.Fatalf("required Responses parameters missing: %+v", a)
	}
	requireRecoveryLoss(t, a.Report, "tools.0.parameters", LossUnsupported)
	for _, edge := range []ConversionID{ConversionChatToResponses, ConversionResponsesToChat} {
		body := map[string]any{"input": "hi", "tools": []any{map[string]any{"type": "custom"}}}
		if edge == ConversionChatToResponses {
			delete(body, "input")
			body["messages"] = []any{map[string]any{"role": "user", "content": "hi"}}
		}
		a = AssessConversion(edge, PhaseRequest, body, FidelityRequirements{})
		requireRecoveryLoss(t, a.Report, "tools.0.type", LossUnsupported)
	}
}

func TestStructuredOutputRequestAwareStreamUsesFiniteFindings(t *testing.T) {
	for _, tc := range []struct {
		request map[string]any
		path    string
	}{
		{decodeJSON[map[string]any](t, `{"tools":[{"type":"function","name":"f","parameters":{}}]}`), "request.tools.0.strict"},
		{decodeJSON[map[string]any](t, `{"tools":[{"type":"function","name":"f","parameters":{},"strict":null}]}`), "request.tools.0.strict"},
		{decodeJSON[map[string]any](t, `{"text":{"format":{"type":"json_schema","name":"x","schema":{},"strict":"true"}}}`), "request.text.format.strict"},
	} {
		reads := 0
		source := pullItems([]string{`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`, "[DONE]"})
		result := OpenAIStreamToResponsesSSEObserved(func() (string, bool) { reads++; return source() }, "m", tc.request, func(string) { t.Error("invalid/unassessed request envelope emitted data") }, StreamObservationOptions{})
		found := false
		for _, path := range result.Assessment.UnknownPaths {
			if path == tc.path {
				found = true
			}
		}
		if _, ok := lossAt(result.Assessment.Report, tc.path); ok {
			found = true
		}
		if reads != 0 || result.Completed || result.Err == nil || !found {
			t.Fatalf("stream bypassed finite request audit: path=%s reads=%d result=%+v", tc.path, reads, result)
		}
	}
	valid := decodeJSON[map[string]any](t, `{"text":{"format":{"type":"json_schema","name":"x","schema":{},"strict":false}},"tools":[{"type":"function","name":"f","parameters":{},"strict":false}]}`)
	result := OpenAIStreamToResponsesSSEObserved(pullItems([]string{`{"choices":[{"index":0,"delta":{"content":"{}"},"finish_reason":"stop"}]}`, "[DONE]"}), "m", valid, func(string) {}, StreamObservationOptions{})
	if !result.Completed || result.Err != nil {
		t.Fatalf("explicit valid request-aware controls rejected: %+v", result)
	}
}

func TestStructuredOutputSharedAuditRejectsMalformedToolsContainer(t *testing.T) {
	for _, tools := range []any{"not-a-list", false, map[string]any{"type": "function"}} {
		t.Run("forward", func(t *testing.T) {
			forward := ChatToResponsesWithReport("m", []map[string]any{{"role": "user", "content": "hi"}}, map[string]any{"tools": tools}, false)
			requireRecoveryLoss(t, forward.Report, "tools", LossUnsupported)
		})
		t.Run("reverse", func(t *testing.T) {
			reverse, _ := ResponsesRequestToChatWithReport(map[string]any{"input": "hi", "tools": tools})
			requireRecoveryLoss(t, reverse.Report, "tools", LossUnsupported)
		})
		t.Run("stream", func(t *testing.T) {
			reads := 0
			source := pullItems([]string{`{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`, "[DONE]"})
			stream := OpenAIStreamToResponsesSSEObserved(func() (string, bool) { reads++; return source() }, "m", map[string]any{"tools": tools}, func(string) { t.Error("malformed tools emitted output") }, StreamObservationOptions{})
			if reads != 0 || stream.Completed || stream.Err == nil {
				t.Fatalf("malformed tools consumed source: reads=%d result=%+v", reads, stream)
			}
			requireRecoveryLoss(t, stream.Assessment.Report, "request.tools", LossUnsupported)
		})
	}
	for _, request := range []map[string]any{nil, {}, {"tools": nil}, {"tools": []any{}}} {
		stream := OpenAIStreamToResponsesSSEObserved(pullItems([]string{`{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`, "[DONE]"}), "m", request, func(string) {}, StreamObservationOptions{})
		if !stream.Completed || stream.Err != nil {
			t.Fatalf("absent/empty tools semantics changed: request=%#v result=%+v", request, stream)
		}
	}
}
