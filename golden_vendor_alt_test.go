package translate

// Alternate thought-signature locations and reasoning_content, which some
// providers emit instead of or next to the documented fields.

const altChatMessages = `[
	{"role":"user","content":"weather?"},
	{"role":"assistant","content":"checking","reasoning_content":"need a lookup",
	 "tool_calls":[
		{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}","thought_signature":"sig-function"}},
		{"id":"call_2","type":"function","function":{"name":"lookup","arguments":"{}"},"thought_signature":"sig-call"},
		{"id":"call_3","type":"function","function":{"name":"lookup","arguments":"{}","thought_signature":""},"thought_signature":"",
		 "extra_content":{"google":{"thought_signature":"sig-google"}}}]},
	{"role":"tool","tool_call_id":"call_1","content":"one"},
	{"role":"tool","tool_call_id":"call_2","content":"two"},
	{"role":"tool","tool_call_id":"call_3","content":"three"},
	{"role":"assistant","content":"done","reasoning_content":""}]`

const altChatResponse = `{"id":"chatcmpl-alt","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,
	"reasoning_content":"thinking out loud",
	"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}","thought_signature":"sig-function"},"thought_signature":"sig-call"}]}}]}`

func alternateVendorGoldenCases() []goldenCase {
	return []goldenCase{
		chatToResponsesCase("alternate_signatures", "gpt-test", altChatMessages, `{}`, false),
		openAIMessagesToAnthropicCase("alternate_signatures", altChatMessages),
		openAIResponseToAnthropicCase("alternate_signatures", "claude-test", altChatResponse),
		chatResponseToResponsesCase("alternate_signatures", "gpt-test", altChatResponse, ""),
		openAIStreamCase("alternate_signatures",
			`{"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"hmm"},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{"reasoning_content":""},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"","thought_signature":"sig-function"}}]},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"fetch","arguments":"{}"},"thought_signature":"sig-call"}]},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":5}}`),
	}
}
