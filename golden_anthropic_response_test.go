package translate

import "testing"

// openAIResponseToAnthropicCase pins OpenAIResponseToAnthropic and its
// WithReport counterpart.
func openAIResponseToAnthropicCase(name, model, response string) goldenCase {
	return goldenCase{name: "OpenAIResponseToAnthropic/" + name, run: func(t *testing.T) map[string]any {
		value := decodeJSON[map[string]any](t, response)
		reported := OpenAIResponseToAnthropicWithReport(value, model)
		samePayload(t, OpenAIResponseToAnthropic(value, model), reported.Value)
		return map[string]any{"input": map[string]any{"model": model, "response": value}, "value": reported.Value, "report": reported.Report}
	}}
}

// anthropicResponseToOpenAICase pins AnthropicResponseToOpenAI and its
// WithReport counterpart.
func anthropicResponseToOpenAICase(name, model, response string) goldenCase {
	return goldenCase{name: "AnthropicResponseToOpenAI/" + name, run: func(t *testing.T) map[string]any {
		value := decodeJSON[map[string]any](t, response)
		reported := AnthropicResponseToOpenAIWithReport(value, model)
		samePayload(t, AnthropicResponseToOpenAI(value, model), reported.Value)
		return map[string]any{"input": map[string]any{"model": model, "response": value}, "value": reported.Value, "report": reported.Report}
	}}
}

func anthropicResponseGoldenCases() []goldenCase {
	return []goldenCase{
		openAIResponseToAnthropicCase("text_stop", "claude-test", `{"id":"chatcmpl-fixture","choices":[{"index":0,"finish_reason":"stop",
			"message":{"role":"assistant","content":"hello world"}}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`),
		openAIResponseToAnthropicCase("tool_calls", "claude-test", `{"id":"chatcmpl-tools","choices":[{"index":0,"finish_reason":"tool_calls",
			"message":{"role":"assistant","content":"calling","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"go\"}"}},
			{"type":"function","function":{"name":"broken","arguments":"{oops"}},{"id":"call_3","type":"function","function":{"name":"empty","arguments":""}}]}}]}`),
		openAIResponseToAnthropicCase("filter_and_choices", "claude-test", `{"id":"chatcmpl-multi","choices":[
			{"index":0,"finish_reason":"content_filter","message":{"role":"assistant","content":"partial"}},
			{"index":1,"finish_reason":"stop","message":{"role":"assistant","content":"other"}}]}`),
		openAIResponseToAnthropicCase("length_without_id", "claude-test", `{"choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","content":"cut"}}]}`),
		openAIResponseToAnthropicCase("empty", "claude-test", `{}`),

		anthropicResponseToOpenAICase("text_tool_use", "claude-test", `{"id":"msg_fixture","type":"message","role":"assistant","model":"claude-test",
			"content":[{"type":"text","text":"let me check"},{"type":"tool_use","id":"toolu_1","name":"search","input":{"q":"go"}}],
			"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":7}}`),
		anthropicResponseToOpenAICase("thinking_block", "claude-test", `{"id":"msg_think","type":"message","role":"assistant",
			"content":[{"type":"thinking","thinking":"hmm","signature":"sig"},{"type":"redacted_thinking","data":"opaque"},{"type":"text","text":"answer"}],
			"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":4}}`),
		anthropicResponseToOpenAICase("stop_reasons", "claude-test", `{"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}],"stop_reason":"max_tokens"}`),
		anthropicResponseToOpenAICase("unknown_stop_and_block", "claude-test", `{"id":"msg_server","content":[{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}},
			{"type":"tool_use","id":"toolu_2","name":"noinput"}],"stop_reason":"pause_turn"}`),
	}
}
