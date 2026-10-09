package translate

import "testing"

// anthropicRequestCase pins AnthropicRequestToOpenAI, its WithReport
// counterpart and the lenient AnthropicMessagesToOpenAI for one payload.
func anthropicRequestCase(name, payload string) []goldenCase {
	strict := goldenCase{name: "AnthropicRequestToOpenAI/" + name, run: func(t *testing.T) map[string]any {
		request := decodeJSON[map[string]any](t, payload)
		messages, kw, incompatible := AnthropicRequestToOpenAI(request)
		reported := AnthropicRequestToOpenAIWithReport(request)
		samePayload(t, OpenAIChatRequest{Messages: messages, Keywords: kw}, reported.Value)
		return map[string]any{"input": request, "value": reported.Value, "incompatible": incompatible, "report": reported.Report}
	}}
	lenient := goldenCase{name: "AnthropicMessagesToOpenAI/" + name, run: func(t *testing.T) map[string]any {
		request := decodeJSON[map[string]any](t, payload)
		messages, _ := request["messages"].([]any)
		return map[string]any{"input": request, "value": AnthropicMessagesToOpenAI(messages, request["system"])}
	}}
	return []goldenCase{strict, lenient}
}

// openAIMessagesToAnthropicCase pins OpenAIMessagesToAnthropic and its
// WithReport counterpart.
func openAIMessagesToAnthropicCase(name, messages string) goldenCase {
	return goldenCase{name: "OpenAIMessagesToAnthropic/" + name, run: func(t *testing.T) map[string]any {
		msgs := decodeJSON[[]map[string]any](t, messages)
		system, converted := OpenAIMessagesToAnthropic(msgs)
		reported := OpenAIMessagesToAnthropicWithReport(msgs)
		// The plain form returns the system as a string, so only the
		// WithReport form can carry SystemBlocks.
		samePayload(t, AnthropicMessages{System: system, Messages: converted}, AnthropicMessages{System: reported.Value.System, Messages: reported.Value.Messages})
		return map[string]any{"input": msgs, "value": reported.Value, "report": reported.Report}
	}}
}

func anthropicGoldenCases() []goldenCase {
	var cases []goldenCase
	for _, c := range [][]goldenCase{
		anthropicRequestCase("text_system_options", `{"model":"claude-test","system":"be nice","max_tokens":256,"temperature":0.7,"top_p":0.95,
			"stop_sequences":["STOP"],"metadata":{"user_id":"u1"},"stream":true,
			"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"},{"role":"user","content":"how are you?"}]}`),
		anthropicRequestCase("system_blocks_image", `{"model":"claude-test","max_tokens":16,
			"system":[{"type":"text","text":"rule one"},{"type":"text","text":"rule two"}],
			"messages":[{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}]}]}`),
		anthropicRequestCase("tools_round_trip", `{"model":"claude-test","max_tokens":32,
			"tools":[{"name":"search","description":"find","input_schema":{"type":"object","properties":{"q":{"type":"string"}}}}],
			"tool_choice":{"type":"tool","name":"search"},
			"messages":[{"role":"user","content":"find go"},
			  {"role":"assistant","content":[{"type":"text","text":"searching"},{"type":"tool_use","id":"toolu_fixture","name":"search","input":{"q":"go"}}]},
			  {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_fixture","content":[{"type":"text","text":"result A"},{"type":"text","text":"result B"}]},{"type":"text","text":"summarize"}]}]}`),
		anthropicRequestCase("tool_result_error", `{"model":"claude-test","messages":[
			{"role":"assistant","content":[{"type":"tool_use","id":"toolu_err","name":"bash","input":{"command":"exit 1"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_err","content":"failed","is_error":true}]}],
			"tool_choice":{"type":"any"}}`),
		anthropicRequestCase("thinking_and_effort", `{"model":"claude-test","max_tokens":64,"thinking":{"type":"disabled"},"output_config":{"effort":"high"},
			"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":"sig"},{"type":"text","text":"hello"}]}]}`),
		anthropicRequestCase("unsupported_values", `{"model":"claude-test","top_k":5,"thinking":{"type":"enabled","budget_tokens":1024},
			"messages":[{"role":"user","content":[{"type":"document","source":{"type":"text","data":"doc"}}],"extra":true},{"role":"system","content":"inline system"}],
			"tool_choice":{"type":"auto","name":"x"}}`),
		anthropicRequestCase("preamble_and_anonymous_tool", `{"model":"claude-test","_llmgw_preamble":"gateway preamble","system":"sys",
			"messages":[{"role":"assistant","content":[{"type":"tool_use","name":"noid","input":{}}]}]}`),
	} {
		cases = append(cases, c...)
	}
	return append(cases,
		openAIMessagesToAnthropicCase("system_developer_user", `[{"role":"system","content":"sys"},{"role":"developer","content":[{"type":"text","text":"dev"}]},{"role":"user","content":"hello"}]`),
		openAIMessagesToAnthropicCase("tool_round_trip", `[{"role":"user","content":"weather?"},
			{"role":"assistant","content":"checking","tool_calls":[{"id":"call_a","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"x\"}"}},{"id":"call_b","type":"function","function":{"name":"get_time","arguments":""}}]},
			{"role":"tool","tool_call_id":"call_a","content":"sunny"},{"role":"tool","tool_call_id":"call_b","content":[{"type":"text","text":"noon"}]},
			{"role":"assistant","content":"done"}]`),
		openAIMessagesToAnthropicCase("invalid_args_and_roles", `[{"role":"critic","content":"odd role"},
			{"role":"assistant","content":null,"tool_calls":[{"type":"function","function":{"name":"broken","arguments":"{not json"}},{"id":"call_arr","type":"function","function":{"name":"list","arguments":"[1,2]"}}]},
			{"role":"assistant","content":""}]`),
		openAIMessagesToAnthropicCase("multipart_image", `[{"role":"user","content":[{"type":"text","text":"see"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}},{"type":"text","text":"this"}]},
			{"role":"assistant","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]`),
	)
}
