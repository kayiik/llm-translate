package translate

import (
	"encoding/json"
	"reflect"
	"testing"
)

func requireRecoveryLoss(t *testing.T, report Report, path string, class LossClass) {
	t.Helper()
	loss, ok := lossAt(report, path)
	if !ok || loss.Severity != LossMaterial || loss.Class != class {
		t.Fatalf("want material %s at %s, got %+v in %+v", class, path, loss, report.Losses)
	}
}

func TestRecoveryImageDetailSurvivesChatResponsesRoundTrip(t *testing.T) {
	for _, kind := range []string{"image_url", "input_image"} {
		t.Run(kind, func(t *testing.T) {
			part := map[string]any{"type": kind, "image_url": "data:image/png;base64,AA==", "detail": "low"}
			if kind == "image_url" {
				part = map[string]any{"type": kind, "image_url": map[string]any{"url": "data:image/png;base64,AA==", "detail": "low"}}
			}
			result := ChatToResponsesWithReport("m", []map[string]any{{"role": "user", "content": []any{part}}}, nil, false)
			input := result.Value["input"].([]any)[0].(map[string]any)
			image := input["content"].([]any)[0].(map[string]any)
			if image["detail"] != "low" || result.Report.HasMaterialLoss() {
				t.Fatalf("image=%#v report=%+v", image, result.Report)
			}
			back, err := ResponsesRequestToChatWithReport(result.Value)
			if err != nil || back.Report.HasMaterialLoss() {
				t.Fatalf("roundtrip err=%v report=%+v", err, back.Report)
			}
			got := back.Value.Messages[0]["content"].([]any)[0].(map[string]any)["image_url"].(map[string]any)
			if got["detail"] != "low" {
				t.Fatalf("roundtrip image=%#v", got)
			}
		})
	}
}

func TestRecoveryMixedTextToolResponseAndChunks(t *testing.T) {
	response := decodeJSON[map[string]any](t, `{"id":"resp_fixture","output":[{"type":"message","content":[{"type":"output_text","text":"checking"}]},{"type":"function_call","call_id":"call_a","name":"lookup","arguments":"{}"}],"usage":{"input_tokens":11,"output_tokens":5,"total_tokens":16,"input_tokens_details":{"cached_tokens":4,"cache_write_tokens":3},"output_tokens_details":{"reasoning_tokens":2}}}`)
	chat := ResponsesToChatWithReport("m", response)
	message := chat.Value["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if message["content"] != "checking" || len(message["tool_calls"].([]any)) != 1 || chat.Report.HasMaterialLoss() {
		t.Fatalf("message=%#v report=%+v", message, chat.Report)
	}
	var text string
	var calls int
	sawUsage := false
	for _, raw := range ResponsesToChatChunks("m", response) {
		var chunk map[string]any
		if err := json.Unmarshal([]byte(raw), &chunk); err != nil {
			t.Fatal(err)
		}
		delta := chunk["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
		if s, ok := delta["content"].(string); ok {
			text += s
		}
		if c, ok := delta["tool_calls"].([]any); ok {
			calls += len(c)
		}
		if usage, ok := chunk["usage"].(map[string]any); ok {
			sawUsage = true
			want := decodeJSON[map[string]any](t, `{"prompt_tokens":11,"completion_tokens":5,"total_tokens":16,"prompt_tokens_details":{"cached_tokens":4,"cache_write_tokens":3},"completion_tokens_details":{"reasoning_tokens":2}}`)
			if !reflect.DeepEqual(usage, want) {
				t.Fatalf("chunk usage=%#v want=%#v", usage, want)
			}
		}
	}
	if text != "checking" || calls != 1 || !sawUsage {
		t.Fatalf("chunks lost content: text=%q calls=%d usage=%v", text, calls, sawUsage)
	}
	back := ChatResponseToResponses("m", chat.Value)
	if back["output_text"] != "checking" || len(back["output"].([]any)) != 2 {
		t.Fatalf("roundtrip=%#v", back)
	}
}

func TestRecoveryNestedContentReportsExactPaths(t *testing.T) {
	for _, role := range []string{"system", "developer", "user", "assistant", "tool"} {
		t.Run(role, func(t *testing.T) {
			messages := []map[string]any{{"role": role, "content": []any{
				map[string]any{"type": "text", "text": "keep"},
				map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "AA==", "format": "wav"}},
			}}}
			for name, report := range map[string]Report{
				"Responses": ChatToResponsesWithReport("m", messages, nil, false).Report,
				"Messages":  OpenAIMessagesToAnthropicWithReport(messages).Report,
			} {
				t.Run(name, func(t *testing.T) {
					class := LossDropped
					if name == "Responses" && role == "user" {
						class = LossUnsupported
					}
					requireRecoveryLoss(t, report, "messages.0.content.1", class)
				})
			}
		})
	}
	for _, content := range []any{[]any{42}, []any{map[string]any{"type": "text", "text": 42}}, 42} {
		report := ChatToResponsesWithReport("m", []map[string]any{{"role": "assistant", "content": content}}, nil, false).Report
		if !report.HasMaterialLoss() {
			t.Fatalf("malformed content=%#v lacks loss: %+v", content, report)
		}
	}
}

func TestRecoveryAnthropicArgumentObjectsAreReported(t *testing.T) {
	for _, args := range []string{"{}", `{"x":1}`, "", "null", "[]", "true", "1", `"text"`, "{"} {
		t.Run(args, func(t *testing.T) {
			messages := []map[string]any{{"role": "assistant", "tool_calls": []any{map[string]any{
				"id": "call_a", "type": "function", "function": map[string]any{"name": "lookup", "arguments": args},
			}}}}
			result := OpenAIMessagesToAnthropicWithReport(messages)
			valid := args == "{}" || args == `{"x":1}` || args == ""
			if result.Report.HasMaterialLoss() == valid {
				t.Fatalf("args=%q report=%+v", args, result.Report)
			}
			if !valid {
				requireRecoveryLoss(t, result.Report, "messages.0.tool_calls.0.function.arguments", LossApproximated)
			}
		})
	}
	// A non-string value is not the legacy empty-string shorthand for {}.
	for _, args := range []any{nil, map[string]any{}, 0} {
		result := OpenAIMessagesToAnthropicWithReport([]map[string]any{{"role": "assistant", "tool_calls": []any{map[string]any{
			"function": map[string]any{"name": "lookup", "arguments": args},
		}}}})
		requireRecoveryLoss(t, result.Report, "messages.0.tool_calls.0.function.arguments", LossApproximated)
	}
}

func TestRecoveryAnthropicResponseArgumentObjectsAreReported(t *testing.T) {
	for _, args := range []any{"null", "[]", "1", "{", nil, map[string]any{}} {
		response := map[string]any{"choices": []any{map[string]any{"message": map[string]any{
			"tool_calls": []any{map[string]any{"function": map[string]any{"name": "lookup", "arguments": args}}},
		}}}}
		report := OpenAIResponseToAnthropicWithReport(response, "m").Report
		requireRecoveryLoss(t, report, "choices.0.message.tool_calls.0.function.arguments", LossUnsupported)
	}
}

func TestRecoveryResponsesNestedContentRefusalPath(t *testing.T) {
	for _, kind := range []string{"input_file", "input_audio", "unknown"} {
		result, err := ResponsesRequestToChatWithReport(map[string]any{"input": []any{map[string]any{
			"role": "user", "content": []any{map[string]any{"type": kind}},
		}}})
		if err == nil {
			t.Fatal("expected native-required refusal")
		}
		requireRecoveryLoss(t, result.Report, "input.0.content.0", LossUnsupported)
	}
}

func TestRecoveryUsageDetailsByTargetContract(t *testing.T) {
	for _, direction := range []string{"chat-to-responses", "responses-to-chat"} {
		t.Run(direction, func(t *testing.T) {
			inputKey, outputKey := "prompt_tokens_details", "completion_tokens_details"
			targetInput, targetOutput := "input_tokens_details", "output_tokens_details"
			if direction == "responses-to-chat" {
				inputKey, targetInput = targetInput, inputKey
				outputKey, targetOutput = targetOutput, outputKey
			}
			for _, detail := range []map[string]any{nil, {}, {"cached_tokens": nil, "reasoning_tokens": nil}, {"cached_tokens": 0}, {"cache_write_tokens": 0}, {"cached_tokens": 4, "cache_write_tokens": 3}, {"reasoning_tokens": 0}, {"reasoning_tokens": 2}, {"cached_tokens": json.Number("4"), "cache_write_tokens": json.Number("3"), "reasoning_tokens": json.Number("2")}} {
				u := map[string]any{"prompt_tokens": 11, "completion_tokens": 5, "input_tokens": 11, "output_tokens": 5, "total_tokens": 16}
				if detail != nil {
					u[inputKey], u[outputKey] = detail, detail
				}
				var got map[string]any
				if direction == "chat-to-responses" {
					got = ChatResponseToResponses("m", map[string]any{"usage": u})["usage"].(map[string]any)
				} else {
					got = ResponsesToChat("m", map[string]any{"usage": u})["usage"].(map[string]any)
				}
				for key, fields := range map[string][]string{targetInput: {"cached_tokens", "cache_write_tokens"}, targetOutput: {"reasoning_tokens"}} {
					want := map[string]any{}
					for _, field := range fields {
						if value := detail[field]; value != nil {
							want[field] = value
						}
					}
					if direction == "chat-to-responses" {
						// The pinned Responses target requires these fields. A
						// placeholder satisfies wire shape, not source measurement.
						required := "cached_tokens"
						if key == targetOutput {
							required = "reasoning_tokens"
						}
						if want[required] == nil {
							want[required] = 0
						}
					}
					actual, exists := got[key]
					if len(want) == 0 {
						if exists {
							t.Errorf("detail=%#v invented %s=%#v", detail, key, actual)
						}
					} else if !reflect.DeepEqual(actual, want) {
						t.Errorf("detail=%#v %s=%#v want=%#v", detail, key, actual, want)
					}
				}
				if got["total_tokens"] != 16 {
					t.Fatalf("aggregate changed: %#v", got)
				}
			}
		})
	}
}

func TestRecoveryResponsesRequiredUsageShape(t *testing.T) {
	// OpenAI's generated response_usage.py at openai-python v1.109.1
	// requires both detail objects and these counters (not Optional fields).
	for _, details := range []map[string]any{
		nil,
		{},
		{"prompt_tokens_details": nil, "completion_tokens_details": nil},
		{"prompt_tokens_details": map[string]any{"cached_tokens": nil, "cache_write_tokens": 3}},
	} {
		usage := map[string]any{"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
		for key, value := range details {
			usage[key] = value
		}
		before, err := json.Marshal(usage)
		if err != nil {
			t.Fatal(err)
		}
		got := ChatResponseToResponses("m", map[string]any{"usage": usage})["usage"].(map[string]any)
		for object, field := range map[string]string{"input_tokens_details": "cached_tokens", "output_tokens_details": "reasoning_tokens"} {
			counts, ok := got[object].(map[string]any)
			if !ok {
				t.Fatalf("required detail object %s missing: %#v", object, got)
			}
			if value, exists := counts[field]; !exists || value != 0 {
				t.Fatalf("required placeholder %s.%s=%#v", object, field, value)
			}
		}
		after, err := json.Marshal(usage)
		if err != nil {
			t.Fatal(err)
		}
		if string(before) != string(after) {
			t.Fatalf("conversion mutated source evidence: before=%s after=%s", before, after)
		}
	}
}

func TestRecoveryUsageAggregateCompatibility(t *testing.T) {
	for _, tc := range []struct {
		raw   string
		total int
	}{
		{`{}`, 0},
		{`{"usage":null}`, 0},
		{`{"usage":{}}`, 0},
		{`{"usage":{"input_tokens":0,"prompt_tokens":0,"completion_tokens":0,"output_tokens":0,"total_tokens":0}}`, 0},
		{`{"usage":{"input_tokens":7,"prompt_tokens":7,"completion_tokens":2,"output_tokens":2}}`, 9},
		{`{"usage":{"input_tokens":7,"prompt_tokens":7,"completion_tokens":2,"output_tokens":2,"total_tokens":12}}`, 12},
	} {
		source := decodeJSON[map[string]any](t, tc.raw)
		responses := ChatResponseToResponses("m", source)
		chat := ResponsesToChat("m", source)
		usage, hasUsage := source["usage"].(map[string]any)
		if !hasUsage {
			if responses["usage"] != nil {
				t.Fatalf("Responses invented absent usage: %#v", responses["usage"])
			}
			if _, ok := chat["usage"]; ok {
				t.Fatalf("Chat invented absent usage: %#v", chat["usage"])
			}
			continue
		}
		if len(usage) == 0 {
			if responses["usage"] != nil {
				t.Fatalf("Responses empty-usage behavior changed: %#v", responses["usage"])
			}
		} else {
			got := responses["usage"].(map[string]any)
			if got["total_tokens"] != tc.total {
				t.Fatalf("Responses total=%#v want=%d", got, tc.total)
			}
		}
		got := chat["usage"].(map[string]any)
		if got["total_tokens"] != tc.total {
			t.Fatalf("Chat total=%#v want=%d", got, tc.total)
		}
	}
}

func TestRecoveryRepresentableContentDoesNotGainLoss(t *testing.T) {
	result := OpenAIMessagesToAnthropicWithReport([]map[string]any{{"role": "user", "content": []any{"a", "b"}}})
	if len(result.Report.Losses) != 0 || result.Value.Messages[0]["content"] != "a\nb" {
		t.Fatalf("representable string parts: %+v", result)
	}
	for _, role := range []string{"system", "user", "assistant", "tool"} {
		messages := []map[string]any{{"role": role, "content": []any{map[string]any{"type": "text", "text": "keep"}}}}
		for _, report := range []Report{ChatToResponsesWithReport("m", messages, nil, false).Report, OpenAIMessagesToAnthropicWithReport(messages).Report} {
			if len(report.Losses) != 0 {
				t.Fatalf("role=%s report=%+v", role, report)
			}
		}
	}
}
