package translate

import (
	"bytes"
	"reflect"
	"testing"
)

// openAIStreamCase pins OpenAIStreamToAnthropicSSE and its WithReport
// counterpart for one sequence of Chat chunk payloads.
func openAIStreamCase(name string, chunks ...string) goldenCase {
	return goldenCase{name: "OpenAIStreamToAnthropicSSE/" + name, run: func(t *testing.T) map[string]any {
		var plain, events []string
		OpenAIStreamToAnthropicSSE(pullItems(chunks), "claude-test", func(e string) { plain = append(plain, e) })
		report := OpenAIStreamToAnthropicSSEWithReport(pullItems(chunks), "claude-test", func(e string) { events = append(events, e) })
		samePayload(t, plain, events)
		return map[string]any{"input": chunks, "value": events, "report": report}
	}}
}

// anthropicStreamCase pins AnthropicSSEToOpenAIChunks and its WithReport
// counterpart for one sequence of SSE lines.
func anthropicStreamCase(name string, lines ...string) goldenCase {
	return goldenCase{name: "AnthropicSSEToOpenAIChunks/" + name, run: func(t *testing.T) map[string]any {
		var plain, chunks []string
		AnthropicSSEToOpenAIChunks(pullItems(lines), "gpt-test", func(c string) { plain = append(plain, c) })
		report := AnthropicSSEToOpenAIChunksWithReport(pullItems(lines), "gpt-test", func(c string) { chunks = append(chunks, c) })
		samePayload(t, plain, chunks)
		return map[string]any{"input": lines, "value": chunks, "report": report}
	}}
}

// anthropicStreamOptionsCase pins AnthropicSSEToOpenAIChunksWithOptions for
// one sequence of SSE lines, without options and with IncludeUsage. Without
// options its chunks are the plain conversion's, cut where an incomplete
// source stopped; IncludeUsage may change the chunks but never the report.
func anthropicStreamOptionsCase(name string, lines ...string) goldenCase {
	return goldenCase{name: "AnthropicSSEToOpenAIChunksWithOptions/" + name, run: func(t *testing.T) map[string]any {
		plain, chunks, usageChunks := []string{}, []string{}, []string{}
		AnthropicSSEToOpenAIChunks(pullItems(lines), "gpt-test", func(c string) { plain = append(plain, c) })
		report := AnthropicSSEToOpenAIChunksWithOptions(pullItems(lines), "gpt-test", func(c string) { chunks = append(chunks, c) }, StreamOptions{})
		usageReport := AnthropicSSEToOpenAIChunksWithOptions(pullItems(lines), "gpt-test", func(c string) { usageChunks = append(usageChunks, c) }, StreamOptions{IncludeUsage: true})
		if !reflect.DeepEqual(report, usageReport) {
			t.Fatalf("IncludeUsage changed the report: %+v, %+v", report, usageReport)
		}
		if len(chunks) > len(plain) || (report.Completed && len(chunks) != len(plain)) {
			t.Fatalf("got %d chunks, the plain conversion %d", len(chunks), len(plain))
		}
		if got, want := renderGolden(t, chunks), renderGolden(t, plain[:len(chunks)]); !bytes.Equal(got, want) {
			t.Fatalf("chunks differ from the plain conversion\ngot:\n%s\nwant:\n%s", got, want)
		}
		return map[string]any{"input": lines, "value": chunks, "value_include_usage": usageChunks, "report": streamReportRecord(report)}
	}}
}

// streamReportRecord renders a StreamReport for a golden file.
func streamReportRecord(report *StreamReport) map[string]any {
	record := map[string]any{"usage": report.Usage, "completed": report.Completed, "error": nil}
	if report.Err != nil {
		record["error"] = report.Err.Error()
	}
	return record
}

func streamGoldenCases() []goldenCase {
	return []goldenCase{
		openAIStreamCase("text_estimated_usage",
			`{"id":"chatcmpl-s","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
			`{"id":"chatcmpl-s","choices":[{"index":0,"delta":{"content":"Hello, "},"finish_reason":null}]}`,
			`{"id":"chatcmpl-s","choices":[{"index":0,"delta":{"content":"world"},"finish_reason":null}]}`,
			`{"id":"chatcmpl-s","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`),
		openAIStreamCase("text_with_usage",
			`{"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
			`{"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11}}`),
		openAIStreamCase("tool_calls",
			`{"choices":[{"index":0,"delta":{"role":"assistant","content":"checking"},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":""}}]},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"go\"}"}},{"index":1,"id":"call_2","function":{"name":"fetch","arguments":"{}"}}]},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":8}}`),
		openAIStreamCase("tool_without_id_or_arguments",
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"ping"}}]},"finish_reason":null}]}`),
		openAIStreamCase("malformed_and_blank", ``, `not json`, `{"choices":"bad"}`,
			`{"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"content_filter"}]}`),
		openAIStreamCase("empty"),

		anthropicStreamCase("text_with_usage",
			`event: message_start`,
			`data: {"type":"message_start","message":{"id":"msg_s","type":"message","role":"assistant","content":[],"model":"claude-test","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`,
			``, `event: content_block_start`,
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`data: {"type":"ping"}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}`,
			`data: {"type":"content_block_stop","index":0}`,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}`,
			`data: {"type":"message_stop"}`),
		anthropicStreamCase("tool_use",
			`data: {"type":"message_start","message":{"id":"msg_t","content":[]}}`,
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Calling"}}`,
			`data: {"type":"content_block_stop","index":0}`,
			`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_s","name":"search","input":{}}}`,
			`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}`,
			`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"go\"}"}}`,
			`data: {"type":"content_block_stop","index":1}`,
			`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}`,
			`data: {"type":"message_stop"}`),
		anthropicStreamCase("thinking_then_text",
			`data: {"type":"message_start","message":{"id":"msg_r","content":[]}}`,
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me think"}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}`,
			`data: {"type":"content_block_stop","index":0}`,
			`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
			`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}`,
			`data: {"type":"content_block_stop","index":1}`,
			`data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"}}`),
		anthropicStreamCase("text_before_start_and_malformed",
			`data: not json`, `data:`, `: comment`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"early"}}`,
			`data: {"type":"message_delta","delta":{"stop_reason":"stop_sequence","stop_sequence":"END"}}`),
		anthropicStreamCase("empty"),

		anthropicStreamOptionsCase("usage_with_cache",
			`data: {"type":"message_start","message":{"id":"msg_c","type":"message","role":"assistant","content":[],"model":"claude-test","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"cache_creation_input_tokens":20,"cache_read_input_tokens":30,"output_tokens":1}}}`,
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"cached"}}`,
			`data: {"type":"content_block_stop","index":0}`,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":7}}`,
			`data: {"type":"message_stop"}`),
		anthropicStreamOptionsCase("usage_without_cache",
			`event: message_start`,
			`data: {"type":"message_start","message":{"id":"msg_u","type":"message","role":"assistant","content":[],"model":"claude-test","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`,
			``, `event: content_block_start`,
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			``, `event: content_block_delta`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
			``, `event: content_block_stop`,
			`data: {"type":"content_block_stop","index":0}`,
			``, `event: message_delta`,
			`data: {"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":3}}`,
			``, `event: message_stop`,
			`data: {"type":"message_stop"}`),
		anthropicStreamOptionsCase("usage_in_message_delta",
			`data: {"type":"message_start","message":{"id":"msg_d","content":[],"usage":{"input_tokens":5,"cache_creation_input_tokens":2,"output_tokens":1}}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"searched"}}`,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":12,"cache_read_input_tokens":3,"output_tokens":9,"server_tool_use":{"web_search_requests":1}}}`,
			`data: {"type":"message_stop"}`),
		anthropicStreamOptionsCase("usage_absent",
			`data: {"type":"message_start","message":{"id":"msg_t","content":[]}}`,
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_s","name":"search","input":{}}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"go\"}"}}`,
			`data: {"type":"content_block_stop","index":0}`,
			`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}`,
			`data: {"type":"message_stop"}`),
		anthropicStreamOptionsCase("error_event",
			`data: {"type":"message_start","message":{"id":"msg_e","content":[],"usage":{"input_tokens":8,"output_tokens":1}}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`,
			``, `event: error`,
			`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`),
		anthropicStreamOptionsCase("error_before_start",
			`event: error`,
			`data: {"type":"error","error":{"type":"api_error","message":"Internal server error"}}`),
		anthropicStreamOptionsCase("truncated",
			`data: {"type":"message_start","message":{"id":"msg_x","content":[],"usage":{"input_tokens":6,"cache_read_input_tokens":4,"output_tokens":1}}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"cut"}}`,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`),
		anthropicStreamOptionsCase("empty"),
	}
}
