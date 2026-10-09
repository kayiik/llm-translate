package translate

import "testing"

// chatToResponsesCase pins ChatToResponses and ChatToResponsesWithReport.
func chatToResponsesCase(name, model, messages, kw string, stream bool) goldenCase {
	return goldenCase{name: "ChatToResponses/" + name, run: func(t *testing.T) map[string]any {
		msgs := decodeJSON[[]map[string]any](t, messages)
		options := decodeJSON[map[string]any](t, kw)
		plain := ChatToResponses(model, msgs, options, stream)
		reported := ChatToResponsesWithReport(model, msgs, options, stream)
		samePayload(t, plain, reported.Value)
		return map[string]any{
			"input":  map[string]any{"model": model, "messages": msgs, "kw": options, "stream": stream},
			"value":  reported.Value,
			"report": reported.Report,
		}
	}}
}

// chatResponseToResponsesCase pins ChatResponseToResponses(WithRequest) and
// their WithReport counterparts. A nil request uses the request-free pair.
func chatResponseToResponsesCase(name, model, chat, request string) goldenCase {
	prefix := "ChatResponseToResponses/"
	if request != "" {
		prefix = "ChatResponseToResponsesWithRequest/"
	}
	return goldenCase{name: prefix + name, run: func(t *testing.T) map[string]any {
		response := decodeJSON[map[string]any](t, chat)
		input := map[string]any{"model": model, "chat": response}
		var reported ConversionResult[map[string]any]
		if request == "" {
			reported = ChatResponseToResponsesWithReport(model, response)
			samePayload(t, ChatResponseToResponses(model, response), reported.Value)
		} else {
			req := decodeJSON[map[string]any](t, request)
			input["request"] = req
			reported = ChatResponseToResponsesWithRequestAndReport(model, response, req)
			samePayload(t, ChatResponseToResponsesWithRequest(model, response, req), reported.Value)
		}
		return map[string]any{"input": input, "value": reported.Value, "report": reported.Report}
	}}
}

func chatResponsesGoldenCases() []goldenCase {
	return []goldenCase{
		chatToResponsesCase("text_system", "gpt-test",
			`[{"role":"system","content":"be terse"},{"role":"user","content":"hello"}]`,
			`{"max_tokens":128,"temperature":0.5,"top_p":0.9,"reasoning_effort":"high"}`, false),
		chatToResponsesCase("developer_multipart_image", "gpt-test",
			`[{"role":"developer","content":[{"type":"text","text":"policy A"},{"type":"text","text":"policy B"}]},
			  {"role":"user","content":[{"type":"text","text":"what is this?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA==","detail":"low"}}]}]`,
			`{"max_completion_tokens":64}`, true),
		chatToResponsesCase("tools_history", "gpt-test",
			`[{"role":"user","content":"weather?"},
			  {"role":"assistant","content":"checking","tool_calls":[{"id":"call_a","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"x\"}"}}]},
			  {"role":"tool","tool_call_id":"call_a","content":"sunny"},
			  {"role":"assistant","content":"It is sunny."},
			  {"role":"user","content":[{"type":"text","text":"thanks"}]}]`,
			`{"tools":[{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object","properties":{"city":{"type":"string"}}},"strict":true}}],
			  "tool_choice":{"type":"function","function":{"name":"get_weather"}},"metadata":{"trace":"t1"}}`, false),
		chatToResponsesCase("missing_tool_output", "gpt-test",
			`[{"role":"assistant","content":null,"tool_calls":[{"id":"call_orphan","type":"function","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}},{"id":"call_done","type":"function","function":{"name":"bash","arguments":"{}"}}]},
			  {"role":"tool","tool_call_id":"call_done","content":[{"type":"text","text":"ok"}]},
			  {"role":"user","content":"next?"}]`,
			`{"tool_choice":"required","max_tokens":10}`, false),
		chatToResponsesCase("unsupported_options", "gpt-test",
			`[{"role":"user","content":"hi"}]`,
			`{"seed":7,"stop":["END"],"n":1,"response_format":{"type":"json_object"},"temperature":null}`, false),
		chatToResponsesCase("empty_role_and_unknown_part", "gpt-test",
			`[{"content":"no role"},{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AA==","format":"wav"}}]}]`,
			`{}`, false),

		chatResponseToResponsesCase("text_usage", "gpt-test",
			`{"id":"chatcmpl-fixture","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hello there"},"finish_reason":"stop"}],
			  "usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":2}}}`, ""),
		chatResponseToResponsesCase("tool_calls", "gpt-test",
			`{"id":"chatcmpl-fixture2","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"go\"}"}},{"id":"call_2","type":"function","function":{"name":"fetch","arguments":"{}"}}]},"finish_reason":"tool_calls"}],
			  "usage":{"input_tokens":5,"output_tokens":6}}`, ""),
		chatResponseToResponsesCase("length_truncated", "gpt-test",
			`{"id":"chatcmpl-fixture3","choices":[{"index":0,"message":{"role":"assistant","content":"partial"},"finish_reason":"length"}]}`, ""),
		chatResponseToResponsesCase("refusal_content_filter", "gpt-test",
			`{"id":"chatcmpl-fixture4","choices":[{"index":0,"message":{"role":"assistant","content":null,"refusal":"cannot comply"},"finish_reason":"content_filter"}]}`, ""),
		chatResponseToResponsesCase("multiple_choices", "gpt-test",
			`{"id":"chatcmpl-fixture5","choices":[{"index":0,"message":{"role":"assistant","content":"one"},"finish_reason":"stop"},{"index":1,"message":{"role":"assistant","content":"two"},"finish_reason":"stop"}]}`, ""),
		chatResponseToResponsesCase("no_id", "gpt-test",
			`{"choices":[{"index":0,"message":{"role":"assistant","content":"anonymous"},"finish_reason":"stop"}]}`, ""),
		chatResponseToResponsesCase("request_fields", "gpt-test",
			`{"id":"chatcmpl-fixture6","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`,
			`{"instructions":"be brief","max_output_tokens":8,"tools":[],"tool_choice":"auto","temperature":0.2,"metadata":{"k":"v"},"parallel_tool_calls":false,"store":false,"reasoning":{"effort":"low"}}`),
	}
}
