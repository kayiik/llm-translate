package translate

import "testing"

// helperCase pins a conversion helper that has no WithReport counterpart.
func helperCase(name, input string, convert func(t *testing.T, input any) any) goldenCase {
	return goldenCase{name: name, run: func(t *testing.T) map[string]any {
		value := decodeJSON[any](t, input)
		return map[string]any{"input": value, "value": convert(t, value)}
	}}
}

func helperGoldenCases() []goldenCase {
	list := func(v any) []any { l, _ := v.([]any); return l }
	object := func(v any) map[string]any { m, _ := v.(map[string]any); return m }
	mapEach := func(v any, convert func(any) any) any {
		out := []any{}
		for _, item := range list(v) {
			out = append(out, convert(item))
		}
		return out
	}
	return []goldenCase{
		helperCase("AnthropicToolsToOpenAI/mixed", `[{"name":"search","description":"find","input_schema":{"type":"object"}},{"name":"bare"},{"description":"no name"},"not an object"]`,
			func(t *testing.T, v any) any { return AnthropicToolsToOpenAI(list(v)) }),
		helperCase("AnthropicToolsToOpenAI/empty", `[]`,
			func(t *testing.T, v any) any { return AnthropicToolsToOpenAI(list(v)) }),
		helperCase("AnthropicToolChoiceToOpenAI/all", `[{"type":"auto"},{"type":"any"},{"type":"none"},{"type":"tool","name":"search"},{"type":"tool"},{"type":"other"},null]`,
			func(t *testing.T, v any) any {
				return mapEach(v, func(item any) any { return AnthropicToolChoiceToOpenAI(object(item)) })
			}),
		helperCase("OpenAIToolsToAnthropic/mixed", `[{"type":"function","function":{"name":"lookup","description":"d","parameters":{"type":"object"}}},{"name":"flat","parameters":{"type":"object"}},{"type":"function","function":{"name":""}},{"type":"function"}]`,
			func(t *testing.T, v any) any { return OpenAIToolsToAnthropic(list(v)) }),
		helperCase("OpenAIFinishReasonToAnthropic/all", `["stop","length","tool_calls","function_call","content_filter","other",null]`,
			func(t *testing.T, v any) any {
				return mapEach(v, func(item any) any { return OpenAIFinishReasonToAnthropic(item) })
			}),
		helperCase("AnthropicStopReasonToOpenAI/all", `["end_turn","max_tokens","tool_use","stop_sequence","pause_turn",null]`,
			func(t *testing.T, v any) any {
				return mapEach(v, func(item any) any { return AnthropicStopReasonToOpenAI(item) })
			}),
		helperCase("NewResponseEnvelope/defaults", `{"model":"gpt-test","id":"resp_fixture","status":"in_progress"}`,
			func(t *testing.T, v any) any {
				in := object(v)
				return NewResponseEnvelope(in["model"].(string), in["id"].(string), in["status"].(string), []any{}, nil)
			}),
		helperCase("NewResponseEnvelope/from_request", `{"model":"gpt-test","id":"","status":"completed","request":{"instructions":"x","tools":[{"type":"function","name":"f"}],"temperature":1,"text":{"format":{"type":"json_object"}},"user":"u"}}`,
			func(t *testing.T, v any) any {
				in := object(v)
				return NewResponseEnvelope(in["model"].(string), in["id"].(string), in["status"].(string), []any{}, object(in["request"]))
			}),
		helperCase("FailedResponseEnvelope/basic", `{"model":"gpt-test","id":"resp_failed","code":"server_error","message":"upstream failed"}`,
			func(t *testing.T, v any) any {
				in := object(v)
				return FailedResponseEnvelope(in["model"].(string), in["id"].(string), in["code"].(string), in["message"].(string), nil)
			}),
	}
}
