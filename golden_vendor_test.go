package translate

// Vendor-field cases prove that each field is either carried or reported at
// its source path, in every direction where it can occur.

const vendorChatMessages = `[
	{"role":"system","content":[{"type":"text","text":"cached policy","cache_control":{"type":"ephemeral","ttl":"1h"}},{"type":"text","text":"tail"}]},
	{"role":"user","content":[
		{"type":"text","text":"look","cache_control":{"type":"ephemeral"}},
		{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="},"cache_control":{"type":"ephemeral"}},
		{"type":"input_audio","input_audio":{"data":"AA==","format":"wav"},"cache_control":{"type":"ephemeral"}}]},
	{"role":"assistant","content":null,
	 "reasoning_details":[{"type":"reasoning.text","text":"thinking","signature":"sig-r"}],
	 "tool_calls":[
		{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"go\"}"},"extra_content":{"google":{"thought_signature":"sig-thought"}}},
		{"id":"call_2","type":"function","function":{"name":"lookup","arguments":"{}"},"extra_content":{"google":{"thought_signature":""}}}]},
	{"role":"tool","tool_call_id":"call_1","content":[{"type":"text","text":"result","cache_control":{"type":"ephemeral"}}]},
	{"role":"tool","tool_call_id":"call_2","content":"second"},
	{"role":"user","content":"thanks"}]`

const vendorChatResponse = `{"id":"chatcmpl-vendor","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"calling",
	"reasoning_details":[{"type":"reasoning.encrypted","data":"opaque"}],
	"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"},"extra_content":{"google":{"thought_signature":"sig-thought"}}}]}}]}`

func vendorGoldenCases() []goldenCase {
	cases := []goldenCase{
		chatToResponsesCase("vendor_fields", "gpt-test", vendorChatMessages, `{}`, false),
		openAIMessagesToAnthropicCase("vendor_fields", vendorChatMessages),
		openAIMessagesToAnthropicCase("cache_control_breakpoints", `[
			{"role":"system","content":"first system"},
			{"role":"developer","content":[{"type":"text","text":"a"},{"type":"text","text":"b","cache_control":{"type":"ephemeral"}},{"type":"text","text":"c"}]},
			{"role":"system","content":[{"type":"text","text":"","cache_control":{"type":"ephemeral"}}]},
			{"role":"user","content":[{"type":"text","text":"one"},{"type":"text","text":"two","cache_control":{"type":"ephemeral"}},{"type":"text","text":"three","cache_control":{"type":"ephemeral"}},{"type":"text","text":"four"}]},
			{"role":"assistant","content":[{"type":"text","text":"answer","cache_control":{"type":"ephemeral"}}],"tool_calls":[{"id":"call_x","type":"function","function":{"name":"f","arguments":"{}"}}]},
			{"role":"tool","tool_call_id":"call_x","content":"plain result"},
			{"role":"user","content":[{"type":"input_text","text":"not a Chat text part","cache_control":{"type":"ephemeral"}}]}]`),
		openAIResponseToAnthropicCase("vendor_fields", "claude-test", vendorChatResponse),
		chatResponseToResponsesCase("vendor_fields", "gpt-test", vendorChatResponse, ""),
		openAIStreamCase("vendor_fields",
			`{"choices":[{"index":0,"delta":{"role":"assistant","reasoning_details":[{"type":"reasoning.text","text":"hmm"}]},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_1","type":"function","function":{"name":"lookup","arguments":""},"extra_content":{"google":{"thought_signature":"sig-thought"}}}]},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{}"},"extra_content":{"google":{"thought_signature":"sig-thought"}}}]},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`),
		anthropicStreamCase("redacted_thinking",
			`data: {"type":"message_start","message":{"id":"msg_v","content":[]}}`,
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque"}}`,
			`data: {"type":"content_block_stop","index":0}`,
			`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":"","cache_control":{"type":"ephemeral"}}}`,
			`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hi"}}`,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}`),
		anthropicResponseToOpenAICase("thinking_blocks_between_text", "gpt-test", `{"id":"msg_v2","content":[{"type":"text","text":"a"},{"type":"thinking","thinking":"t","signature":"s"},{"type":"text","text":"b"}],"stop_reason":"end_turn"}`),
	}
	cases = append(cases, anthropicRequestCase("cache_control_everywhere", `{"model":"claude-test","max_tokens":8,
		"system":[{"type":"text","text":"sys","cache_control":{"type":"ephemeral"}}],
		"tools":[{"name":"search","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}],
		"messages":[
			{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="},"cache_control":{"type":"ephemeral"}}]},
			{"role":"assistant","content":[{"type":"redacted_thinking","data":"opaque"},{"type":"tool_use","id":"toolu_v","name":"search","input":{},"cache_control":{"type":"ephemeral"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_v","content":[{"type":"text","text":"r","cache_control":{"type":"ephemeral"}}],"cache_control":{"type":"ephemeral"}}]}]}`)...)
	cases = append(cases, responsesToChatCase("reasoning_items", "gpt-test", `{"id":"resp_v","output":[
		{"type":"reasoning","id":"rs_1","summary":[]},
		{"type":"function_call","call_id":"call_1","name":"f","arguments":"{}"},
		{"type":"reasoning","id":"rs_2","encrypted_content":"opaque"}]}`)...)
	return cases
}
