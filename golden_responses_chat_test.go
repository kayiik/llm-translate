package translate

import "testing"

// responsesRequestToChatCase pins ResponsesRequestToChat and its WithReport
// counterpart, including the validation error both return.
func responsesRequestToChatCase(name, payload string) goldenCase {
	return goldenCase{name: "ResponsesRequestToChat/" + name, run: func(t *testing.T) map[string]any {
		request := decodeJSON[map[string]any](t, payload)
		messages, kw, plainErr := ResponsesRequestToChat(request)
		reported, err := ResponsesRequestToChatWithReport(request)
		samePayload(t, OpenAIChatRequest{Messages: messages, Keywords: kw}, reported.Value)
		record := map[string]any{"input": request, "value": reported.Value, "report": reported.Report}
		if (plainErr == nil) != (err == nil) || (err != nil && plainErr.Error() != err.Error()) {
			t.Fatalf("plain error %v, WithReport error %v", plainErr, err)
		}
		if err != nil {
			record["error"] = err.Error()
		}
		return record
	}}
}

// responsesToChatCase pins ResponsesToChat and ResponsesToChatChunks with
// their WithReport counterparts for one Responses object.
func responsesToChatCase(name, model, response string) []goldenCase {
	chat := goldenCase{name: "ResponsesToChat/" + name, run: func(t *testing.T) map[string]any {
		value := decodeJSON[map[string]any](t, response)
		reported := ResponsesToChatWithReport(model, value)
		samePayload(t, ResponsesToChat(model, value), reported.Value)
		return map[string]any{"input": map[string]any{"model": model, "response": value}, "value": reported.Value, "report": reported.Report}
	}}
	chunks := goldenCase{name: "ResponsesToChatChunks/" + name, run: func(t *testing.T) map[string]any {
		value := decodeJSON[map[string]any](t, response)
		reported := ResponsesToChatChunksWithReport(model, value)
		samePayload(t, ResponsesToChatChunks(model, value), reported.Value)
		return map[string]any{"input": map[string]any{"model": model, "response": value}, "value": reported.Value, "report": reported.Report}
	}}
	return []goldenCase{chat, chunks}
}

func responsesChatGoldenCases() []goldenCase {
	cases := []goldenCase{
		responsesRequestToChatCase("string_input", `{"model":"gpt-test","input":"hello","instructions":"be terse","max_output_tokens":64,
			"reasoning":{"effort":"medium"},"temperature":0.3,"top_p":1,"stream":true}`),
		responsesRequestToChatCase("structured_history", `{"model":"gpt-test","input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"look"},{"type":"input_image","image_url":"data:image/png;base64,AA==","detail":"low"}]},
			{"type":"function_call","call_id":"call_1","name":"one","arguments":"{\"a\":1}"},
			{"type":"function_call","call_id":"call_2","name":"two","arguments":"{\"b\":2}"},
			{"type":"function_call_output","call_id":"call_1","output":[{"type":"output_text","text":"done"}]},
			{"type":"function_call_output","call_id":"call_2","output":"also done"},
			{"role":"assistant","content":[{"type":"output_text","text":"all done"}]}],
			"tools":[{"type":"function","name":"one","description":"first","parameters":{"type":"object"},"strict":false}],
			"tool_choice":{"type":"function","name":"one"},"parallel_tool_calls":true,"metadata":{"k":"v"}}`),
		responsesRequestToChatCase("text_parts_collapse", `{"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello "},{"type":"input_text","text":"world"}]}],"tool_choice":"auto"}`),
		responsesRequestToChatCase("object_input", `{"input":{"role":"developer","content":"rules"}}`),
		responsesRequestToChatCase("client_hints", `{"input":"hi","client_metadata":{"client":"fixture"},"include":[],"prompt_cache_key":"fixture-key","store":false,"background":false}`),
		responsesRequestToChatCase("rejects_unknown_field", `{"input":"hi","text":{"format":{"type":"json_schema"}}}`),
		responsesRequestToChatCase("rejects_reasoning_item", `{"input":[{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"thinking"}],"encrypted_content":"opaque"},{"role":"user","content":"hi"}]}`),
		responsesRequestToChatCase("rejects_previous_response", `{"input":"hi","previous_response_id":"resp_prev"}`),
		responsesRequestToChatCase("rejects_builtin_tool", `{"input":"hi","tools":[{"type":"web_search"}]}`),
		responsesRequestToChatCase("rejects_missing_input", `{"model":"gpt-test"}`),
	}
	for _, c := range [][]goldenCase{
		responsesToChatCase("text_usage", "gpt-test", `{"id":"resp_fixture","object":"response","status":"completed",
			"output":[{"type":"message","id":"msg_fixture","role":"assistant","content":[{"type":"output_text","text":"Hi there!","annotations":[]}]}],
			"usage":{"input_tokens":10,"output_tokens":3,"total_tokens":13}}`),
		responsesToChatCase("reasoning_then_message", "gpt-test", `{"id":"resp_reason","status":"completed","output":[
			{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"thinking"}],"encrypted_content":"opaque"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}],
			"usage":{"input_tokens":4,"output_tokens":9}}`),
		responsesToChatCase("function_calls", "gpt-test", `{"id":"resp_tools","output":[
			{"type":"function_call","call_id":"call_9","name":"do_thing","arguments":"{\"a\":1}"},
			{"type":"function_call","id":"fc_2","name":"other","arguments":"{}"}]}`),
		responsesToChatCase("incomplete_max_tokens", "gpt-test", `{"id":"resp_cut","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},
			"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"partial"}]}]}`),
		responsesToChatCase("incomplete_content_filter", "gpt-test", `{"id":"resp_filter","status":"incomplete","incomplete_details":{"reason":"content_filter"},"output":[]}`),
		responsesToChatCase("annotations_refusal_unknown_item", "gpt-test", `{"id":"resp_mixed","output":[
			{"type":"web_search_call","id":"ws_1","status":"completed"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"cited","annotations":[{"type":"url_citation","url":"https://example.com"}]},{"type":"refusal","refusal":"no"}]}]}`),
		responsesToChatCase("output_text_fallback", "gpt-test", `{"output_text":"from convenience field"}`),
	} {
		cases = append(cases, c...)
	}
	return cases
}
