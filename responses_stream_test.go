package translate

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

func bridgeEvents(t *testing.T, records []string) []map[string]any {
	t.Helper()
	var result []map[string]any
	for _, record := range records {
		_, raw, ok := strings.Cut(record, "\ndata: ")
		if !ok {
			t.Fatalf("not complete Responses SSE: %q", record)
		}
		body, ok := decodeObservedObject(strings.TrimSpace(raw))
		if !ok {
			t.Fatalf("bad SSE JSON: %q", record)
		}
		if !strings.HasPrefix(record, "event: "+body["type"].(string)+"\n") {
			t.Fatalf("event type mismatch: %q", record)
		}
		result = append(result, body)
	}
	return result
}

func bridgeJSON(t *testing.T, records []string) []map[string]any {
	t.Helper()
	var result []map[string]any
	for _, record := range records {
		body, ok := decodeObservedObject(record)
		if !ok {
			t.Fatalf("bad Chat JSON (DONE is caller-owned): %q", record)
		}
		result = append(result, body)
	}
	return result
}

func TestChatResponsesStreamIncrementalToolsRefusalAndExactUsage(t *testing.T) {
	source := []string{
		`{"id":"chatcmpl-source","choices":[{"index":0,"delta":{"content":"checking "}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":4,"id":"call_a","type":"function","function":{"name":"one","arguments":"{\"a\":"}},{"index":1,"id":"call_b","type":"function","function":{"name":"two","arguments":"{"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"}"}},{"index":4,"function":{"arguments":"1}"}}],"refusal":"cannot "}}]}`,
		`{"choices":[{"index":0,"delta":{"refusal":"comply"},"finish_reason":"tool_calls"}]}`,
		`{"choices":[],"usage":{"prompt_tokens":9007199254740993,"completion_tokens":0,"total_tokens":9007199254740993,"prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":3},"completion_tokens_details":{"reasoning_tokens":0}}}`,
		"[DONE]",
	}
	read := 0
	early := false
	var output []string
	result := OpenAIStreamToResponsesSSEObserved(func() (string, bool) {
		if read == len(source) {
			return "", false
		}
		raw := source[read]
		read++
		return raw, true
	}, "m", nil, func(raw string) {
		output = append(output, raw)
		if strings.Contains(raw, "event: response.output_text.delta\n") {
			early = read < len(source)-1
		}
	}, StreamObservationOptions{})
	if !result.Completed || result.Err != nil || !early || result.Assessment.Conversion != ConversionResponsesToChat {
		t.Fatalf("not incremental/complete: %+v early=%v", result, early)
	}
	if result.Usage.InputTokens.Value != 9007199254740993 || result.Usage.OutputTokens.Presence != CountPresent || result.Usage.OutputTokens.Value != 0 {
		t.Fatalf("source usage=%+v", result.Usage)
	}
	events := bridgeEvents(t, output)
	var terminal map[string]any
	for i, event := range events {
		if toInt(event["sequence_number"]) != i {
			t.Fatalf("sequence=%v index=%d", event, i)
		}
		if event["type"] == "response.completed" {
			terminal = event["response"].(map[string]any)
		}
	}
	if terminal == nil || terminal["id"] != "resp_source" {
		t.Fatalf("terminal identity=%v", terminal)
	}
	gotTools := map[string]string{}
	var text, refusal string
	for _, raw := range mustList(terminal["output"]) {
		item, _ := asMap(raw)
		if item["type"] == "function_call" {
			gotTools[item["call_id"].(string)] = item["arguments"].(string)
		}
		for _, rawPart := range mustList(item["content"]) {
			part, _ := asMap(rawPart)
			if part["type"] == "output_text" {
				text += part["text"].(string)
			}
			if part["type"] == "refusal" {
				refusal += part["refusal"].(string)
			}
		}
	}
	if !reflect.DeepEqual(gotTools, map[string]string{"call_a": `{"a":1}`, "call_b": `{}`}) || text != "checking " || refusal != "cannot comply" {
		t.Fatalf("terminal loss: tools=%v text=%q refusal=%q", gotTools, text, refusal)
	}
	usage := terminal["usage"].(map[string]any)
	if n, ok := usageInteger(usage["input_tokens"]); !ok || n != 9007199254740993 {
		t.Fatalf("wire usage rounded: %v", usage)
	}
}

func TestChatResponsesStreamTerminalAndUnsupported(t *testing.T) {
	for _, tc := range []struct {
		name     string
		chunks   []string
		complete bool
		terminal string
	}{
		{"truncated", []string{`{"choices":[{"index":0,"delta":{"content":"partial"}}]}`}, false, ""},
		{"error", []string{`{"error":{"type":"overloaded_error","message":"synthetic"}}`}, false, ""},
		{"length", []string{`{"choices":[{"index":0,"delta":{"content":"cut"},"finish_reason":"length"}]}`, "[DONE]"}, true, "response.incomplete"},
		{"multiple choices", []string{`{"choices":[{"index":0,"delta":{"content":"one"}},{"index":1,"delta":{"content":"two"}}]}`, "[DONE]"}, false, ""},
		{"opaque", []string{`{"choices":[{"index":0,"delta":{"reasoning_details":[{"type":"reasoning.encrypted","data":"opaque","summary":[]}]}}]}`, "[DONE]"}, false, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var output []string
			result := OpenAIStreamToResponsesSSEObserved(pullItems(tc.chunks), "m", nil, func(raw string) { output = append(output, raw) }, StreamObservationOptions{})
			if result.Completed != tc.complete || (!tc.complete && result.Err == nil) {
				t.Fatalf("terminal=%+v", result)
			}
			found := false
			for _, event := range bridgeEvents(t, output) {
				if event["type"] == tc.terminal {
					found = true
				}
				if !tc.complete && (event["type"] == "response.completed" || event["type"] == "response.incomplete") {
					t.Fatal("fabricated success terminal")
				}
			}
			if tc.complete && !found {
				t.Fatalf("missing terminal=%s", tc.terminal)
			}
		})
	}
}

func TestResponsesChatStreamIncrementalRoundTripAndUsageTail(t *testing.T) {
	source := []string{
		`{"type":"response.created","response":{"id":"resp_native","model":"actual","status":"in_progress"}}`,
		`{"type":"response.output_item.added","output_index":3,"item":{"id":"msg1","type":"message","role":"assistant","content":[]}}`,
		`{"type":"response.output_text.delta","output_index":3,"item_id":"msg1","content_index":0,"delta":"hello"}`,
		`{"type":"response.output_item.added","output_index":8,"item":{"id":"fc_a","type":"function_call","call_id":"call_a","name":"one","arguments":""}}`,
		`{"type":"response.output_item.added","output_index":2,"item":{"id":"fc_b","type":"function_call","call_id":"call_b","name":"two","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","output_index":8,"item_id":"fc_a","delta":"{\"a\":"}`,
		`{"type":"response.function_call_arguments.delta","output_index":2,"item_id":"fc_b","delta":"{}"}`,
		`{"type":"response.function_call_arguments.delta","output_index":8,"item_id":"fc_a","delta":"1}"}`,
		`{"type":"response.refusal.delta","output_index":3,"item_id":"msg1","content_index":1,"delta":"no"}`,
		`{"type":"response.completed","response":{"id":"resp_native","status":"completed","output":[{"id":"msg1","type":"message","role":"assistant","content":[{"type":"output_text","text":"hello","annotations":[]},{"type":"refusal","refusal":"no"}]},{"id":"fc_a","type":"function_call","call_id":"call_a","name":"one","arguments":"{\"a\":1}"},{"id":"fc_b","type":"function_call","call_id":"call_b","name":"two","arguments":"{}"}],"usage":{"input_tokens":9007199254740993,"output_tokens":0,"total_tokens":9007199254740993,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":3},"output_tokens_details":{"reasoning_tokens":0}}}}`,
	}
	read := 0
	early := false
	var output []string
	result := ResponsesSSEToOpenAIChunksObserved(func() (string, bool) {
		if read == len(source) {
			return "", false
		}
		raw := source[read]
		read++
		return raw, true
	}, "m", func(raw string) {
		output = append(output, raw)
		if strings.Contains(raw, `"content":"hello"`) {
			early = read < len(source)
		}
	}, StreamObservationOptions{IncludeUsage: true})
	if !result.Completed || result.Err != nil || !early || result.Assessment.Conversion != ConversionChatToResponses {
		t.Fatalf("not incremental: %+v early=%v", result, early)
	}
	var text, refusal string
	tools := map[int]string{}
	ids := map[int]string{}
	sawFinish, sawUsage := false, false
	for _, chunk := range bridgeJSON(t, output) {
		if chunk["id"] != "resp_native" {
			t.Fatalf("identity switched: %v", chunk)
		}
		if usage, ok := asMap(chunk["usage"]); ok {
			sawUsage = true
			if !sawFinish {
				t.Fatal("usage before finish")
			}
			if n, ok := usageInteger(usage["prompt_tokens"]); !ok || n != 9007199254740993 {
				t.Fatalf("usage rounded: %v", usage)
			}
		}
		for _, rawChoice := range mustList(chunk["choices"]) {
			choice, _ := asMap(rawChoice)
			if present(choice["finish_reason"]) {
				sawFinish = true
			}
			delta, _ := asMap(choice["delta"])
			if value, _ := delta["content"].(string); value != "" {
				text += value
			}
			if value, _ := delta["refusal"].(string); value != "" {
				refusal += value
			}
			for _, rawCall := range mustList(delta["tool_calls"]) {
				call, _ := asMap(rawCall)
				index := toInt(call["index"])
				if value, _ := call["id"].(string); value != "" {
					ids[index] = value
				}
				fn, _ := asMap(call["function"])
				if value, _ := fn["arguments"].(string); value != "" {
					tools[index] += value
				}
			}
		}
	}
	if text != "hello" || refusal != "no" || !reflect.DeepEqual(ids, map[int]string{0: "call_a", 1: "call_b"}) || !reflect.DeepEqual(tools, map[int]string{0: `{"a":1}`, 1: `{}`}) || !sawUsage {
		t.Fatalf("loss/duplicate delta: text=%q refusal=%q ids=%v args=%v usage=%v", text, refusal, ids, tools, sawUsage)
	}
}

func TestResponsesChatStreamOpaqueUnsupportedAndTermination(t *testing.T) {
	for _, tc := range []struct {
		name     string
		events   []string
		complete bool
	}{
		{"truncated", []string{`{"type":"response.created","response":{"id":"r1"}}`}, false},
		{"error", []string{`{"type":"error","code":"synthetic_error","message":"synthetic"}`}, false},
		{"incomplete", []string{`{"type":"response.incomplete","response":{"id":"r1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[]}}`}, true},
		{"opaque", []string{`{"type":"response.output_item.added","output_index":0,"item":{"id":"rs1","type":"reasoning","summary":[],"encrypted_content":"opaque"}}`}, false},
		{"hosted tool", []string{`{"type":"response.output_item.added","output_index":0,"item":{"id":"ws1","type":"web_search_call"}}`}, false},
		{"citation", []string{`{"type":"response.output_text.annotation.added","output_index":3,"content_index":2,"annotation":{"type":"url_citation","url":"https://example.invalid"}}`}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var output []string
			observed := false
			result := ResponsesSSEToOpenAIChunksObserved(pullItems(tc.events), "m", func(raw string) {
				if tc.name == "opaque" || tc.name == "hosted tool" || tc.name == "citation" {
					if !observed {
						t.Fatal("affected output preceded loss observation")
					}
				}
				output = append(output, raw)
			}, StreamObservationOptions{Observe: func(o StreamObservation) {
				if o.Assessment.Feasibility != FeasibilitySupported {
					observed = true
				}
			}})
			if result.Completed != tc.complete || (!tc.complete && result.Err == nil) {
				t.Fatalf("terminal=%+v", result)
			}
			for _, chunk := range bridgeJSON(t, output) {
				for _, raw := range mustList(chunk["choices"]) {
					choice, _ := asMap(raw)
					if !tc.complete && present(choice["finish_reason"]) {
						t.Fatal("fabricated finish on failure")
					}
				}
			}
			if tc.name == "truncated" && !errors.Is(result.Err, ErrResponsesStreamTruncated) {
				t.Fatalf("wrong truncation=%v", result.Err)
			}
		})
	}
}

func TestResponsesStreamBridgeIdentityAndRequestEcho(t *testing.T) {
	var identities []string
	for i := 0; i < 2; i++ {
		var output []string
		result := OpenAIStreamToResponsesSSEObserved(pullItems([]string{`{"choices":[{"index":0,"delta":{"content":"early"}}]}`, `{"id":"chatcmpl-late","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`, "[DONE]"}), "m", map[string]any{"metadata": map[string]any{"k": "v"}, "store": false}, func(raw string) { output = append(output, raw) }, StreamObservationOptions{})
		if !result.Completed {
			t.Fatalf("identity run=%+v", result)
		}
		events := bridgeEvents(t, output)
		created := events[0]["response"].(map[string]any)
		id := created["id"].(string)
		identities = append(identities, id)
		for _, event := range events {
			if response, ok := asMap(event["response"]); ok {
				if response["id"] != id {
					t.Fatal("late ID changed output")
				}
				if response["previous_response_id"] != nil || response["store"] != false {
					t.Fatalf("fabricated stored state: %v", response)
				}
			}
		}
	}
	if identities[0] == identities[1] {
		t.Fatal("constant synthetic stream identity")
	}
}

func TestResponsesStreamBridgeFiniteNestedScopeAndNoRepair(t *testing.T) {
	for _, raw := range []string{
		`{"choices":[{"index":0,"delta":{"tool_calls":"not-a-list"},"finish_reason":"stop"}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","extra_content":{"unreviewed":true},"function":{"name":"f","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"f","arguments":"{"}}]},"finish_reason":"tool_calls"}]}`,
	} {
		var output []string
		result := OpenAIStreamToResponsesSSEObserved(pullItems([]string{raw, "[DONE]"}), "m", nil, func(s string) { output = append(output, s) }, StreamObservationOptions{})
		if result.Completed || result.Err == nil {
			t.Fatalf("unassessed/invalid tool stream completed: %s %+v", raw, result)
		}
	}
	for _, raw := range []string{
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m1","role":"assistant","content":"lost"}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"f1","call_id":"c1","name":"f","arguments":"{}","unreviewed":true}}`,
	} {
		result := ResponsesSSEToOpenAIChunksObserved(pullItems([]string{raw, `{"type":"response.completed","response":{"output":[]}}`}), "m", func(string) {}, StreamObservationOptions{})
		if result.Completed || result.Err == nil {
			t.Fatalf("unsupported native item completed: %s %+v", raw, result)
		}
	}
}

func TestResponsesStreamBridgeRejectsStoredLineageBeforeReading(t *testing.T) {
	read := 0
	result := OpenAIStreamToResponsesSSEObserved(func() (string, bool) { read++; return "[DONE]", true }, "m", map[string]any{"previous_response_id": "native_only"}, func(string) { t.Fatal("lineage rejection emitted data") }, StreamObservationOptions{})
	if result.Err == nil || read != 0 {
		t.Fatalf("stored lineage fabricated or source consumed: reads=%d %+v", read, result)
	}
}

func TestResponsesStreamBridgeLateIdentityAndNativeSequence(t *testing.T) {
	for run := 0; run < 2; run++ {
		var output []string
		result := ResponsesSSEToOpenAIChunksObserved(pullItems([]string{
			`{"type":"response.output_text.delta","sequence_number":0,"output_index":0,"item_id":"m1","content_index":0,"delta":"early"}`,
			`{"type":"response.created","sequence_number":1,"response":{"id":"late-native","created_at":123}}`,
			`{"type":"response.completed","sequence_number":2,"response":{"id":"late-native","output":[]}}`,
		}), "m", func(raw string) { output = append(output, raw) }, StreamObservationOptions{})
		if !result.Completed {
			t.Fatalf("late identity=%+v", result)
		}
		var id string
		for _, chunk := range bridgeJSON(t, output) {
			if id == "" {
				id = chunk["id"].(string)
			}
			if chunk["id"] != id || chunk["created"] == nil {
				t.Fatalf("unstable/invalid required shape: %v", chunk)
			}
		}
	}
	result := ResponsesSSEToOpenAIChunksObserved(pullItems([]string{`{"type":"response.created","sequence_number":1,"response":{"id":"r1"}}`, `{"type":"response.completed","sequence_number":1,"response":{"id":"r1","output":[]}}`}), "m", func(string) {}, StreamObservationOptions{})
	if result.Completed || result.Err == nil {
		t.Fatalf("duplicate/replayed native sequence accepted: %+v", result)
	}
}

func TestResponsesStreamBridgeMissingUsageAndTerminalSnapshots(t *testing.T) {
	var output []string
	result := ResponsesSSEToOpenAIChunksObserved(pullItems([]string{
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"m1","type":"message","role":"assistant","content":[]}}`,
		`{"type":"response.output_text.delta","output_index":0,"item_id":"m1","content_index":0,"delta":"hello"}`,
		`{"type":"response.output_text.done","output_index":0,"item_id":"m1","content_index":0,"text":"hello"}`,
		`{"type":"response.content_part.done","output_index":0,"item_id":"m1","content_index":0,"part":{"type":"output_text","text":"hello","annotations":[]}}`,
		`{"type":"response.completed","response":{"output":[{"id":"m1","type":"message","role":"assistant","content":[{"type":"output_text","text":"hello","annotations":[]}]}]}}`,
	}), "m", func(raw string) { output = append(output, raw) }, StreamObservationOptions{IncludeUsage: true})
	if !result.Completed || result.Usage.InputTokens.Presence != CountAbsent {
		t.Fatalf("missing source usage invented: %+v", result)
	}
	text := ""
	for _, chunk := range bridgeJSON(t, output) {
		if chunk["usage"] != nil {
			t.Fatalf("fabricated missing usage: %v", chunk)
		}
		for _, raw := range mustList(chunk["choices"]) {
			choice, _ := asMap(raw)
			delta, _ := asMap(choice["delta"])
			value, _ := delta["content"].(string)
			text += value
		}
	}
	if text != "hello" {
		t.Fatalf("done snapshots duplicated content: %q", text)
	}
}

func TestResponsesStreamBridgeRejectsMalformedSemanticContainers(t *testing.T) {
	for _, delta := range []string{`"lost"`, `{"role":42}`, `{"reasoning_details":"unassessed"}`} {
		result := OpenAIStreamToResponsesSSEObserved(pullItems([]string{`{"choices":[{"index":0,"delta":` + delta + `,"finish_reason":"stop"}]}`, "[DONE]"}), "m", nil, func(string) {}, StreamObservationOptions{})
		if result.Completed || result.Err == nil {
			t.Fatalf("malformed source passed: %s %+v", delta, result)
		}
	}
	result := ResponsesSSEToOpenAIChunksObserved(pullItems([]string{`{"type":"response.content_part.added","output_index":0,"item_id":"m1","content_index":0,"part":{"type":"output_text","text":"answer","annotations":{"url":"https://example.invalid"}}}`, `{"type":"response.completed","response":{"output":[]}}`}), "m", func(string) {}, StreamObservationOptions{})
	if result.Completed || result.Err == nil {
		t.Fatalf("unassessed annotation shape passed: %+v", result)
	}
}

func TestResponsesStreamBridgeIncompleteToolRoundTrip(t *testing.T) {
	var records []string
	forward := OpenAIStreamToResponsesSSEObserved(pullItems([]string{`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":3,"id":"c1","type":"function","function":{"name":"f","arguments":"{"}}]},"finish_reason":"length"}]}`, "[DONE]"}), "m", nil, func(raw string) { records = append(records, raw) }, StreamObservationOptions{})
	if !forward.Completed {
		t.Fatalf("incomplete transport failed: %+v", forward)
	}
	var source []string
	for _, event := range bridgeEvents(t, records) {
		if event["type"] == "response.function_call_arguments.done" || event["type"] == "response.output_item.done" {
			t.Fatalf("partial tool claimed done: %v", event)
		}
		encoded, err := json.Marshal(event)
		if err != nil {
			t.Fatal(err)
		}
		source = append(source, string(encoded))
	}
	var chunks []string
	back := ResponsesSSEToOpenAIChunksObserved(pullItems(source), "m", func(raw string) { chunks = append(chunks, raw) }, StreamObservationOptions{})
	if !back.Completed || back.Err != nil {
		t.Fatalf("incomplete bridge roundtrip failed: %+v", back)
	}
	arguments := ""
	finish := ""
	for _, chunk := range bridgeJSON(t, chunks) {
		for _, rawChoice := range mustList(chunk["choices"]) {
			choice, _ := asMap(rawChoice)
			if value, _ := choice["finish_reason"].(string); value != "" {
				finish = value
			}
			delta, _ := asMap(choice["delta"])
			for _, rawCall := range mustList(delta["tool_calls"]) {
				call, _ := asMap(rawCall)
				fn, _ := asMap(call["function"])
				value, _ := fn["arguments"].(string)
				arguments += value
			}
		}
	}
	if arguments != "{" || finish != "length" {
		t.Fatalf("partial tool repaired or lost: arguments=%q finish=%q", arguments, finish)
	}
}

func TestResponsesStreamBridgeRejectsDuplicateToolIdentity(t *testing.T) {
	result := OpenAIStreamToResponsesSSEObserved(pullItems([]string{`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"duplicate","type":"function","function":{"name":"f","arguments":"{}"}},{"index":1,"id":"duplicate","type":"function","function":{"name":"g","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`, "[DONE]"}), "m", nil, func(string) {}, StreamObservationOptions{})
	if result.Completed || result.Err == nil {
		t.Fatalf("duplicate tool link accepted: %+v", result)
	}
	result = ResponsesSSEToOpenAIChunksObserved(pullItems([]string{`{"type":"response.output_item.added","output_index":0,"item":{"id":"f1","type":"function_call","call_id":"duplicate","name":"f","arguments":"{}"}}`, `{"type":"response.output_item.added","output_index":1,"item":{"id":"f2","type":"function_call","call_id":"duplicate","name":"g","arguments":"{}"}}`, `{"type":"response.completed","response":{"output":[]}}`}), "m", func(string) {}, StreamObservationOptions{})
	if result.Completed || result.Err == nil {
		t.Fatalf("duplicate native tool link accepted: %+v", result)
	}
}

func TestResponsesStreamBridgeAcceptsDocumentedDeltaObfuscation(t *testing.T) {
	var output []string
	result := ResponsesSSEToOpenAIChunksObserved(pullItems([]string{
		`{"type":"response.output_text.delta","output_index":0,"item_id":"m1","content_index":0,"delta":"hi","obfuscation":"padding-text"}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"id":"f1","type":"function_call","call_id":"c1","name":"f","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","output_index":1,"item_id":"f1","delta":"{}","obfuscation":"padding-arguments"}`,
		`{"type":"response.completed","response":{"output":[]}}`,
	}), "m", func(raw string) { output = append(output, raw) }, StreamObservationOptions{})
	if !result.Completed || result.Err != nil || result.Assessment.Feasibility != FeasibilitySupported {
		t.Fatalf("default source padding rejected: %+v", result)
	}
	text, args := "", ""
	for _, chunk := range bridgeJSON(t, output) {
		for _, rawChoice := range mustList(chunk["choices"]) {
			choice, _ := asMap(rawChoice)
			delta, _ := asMap(choice["delta"])
			value, _ := delta["content"].(string)
			text += value
			for _, rawCall := range mustList(delta["tool_calls"]) {
				call, _ := asMap(rawCall)
				fn, _ := asMap(call["function"])
				value, _ := fn["arguments"].(string)
				args += value
			}
		}
	}
	if text != "hi" || args != "{}" {
		t.Fatalf("padding changed semantic deltas: text=%q args=%q", text, args)
	}
}

func TestResponsesStreamBridgeRejectsMalformedResponseContainers(t *testing.T) {
	for _, source := range [][]string{
		{`{"type":"response.completed","response":{"id":"r1","status":"completed","output":{"type":"message","id":"m1","role":"assistant","content":[{"type":"output_text","text":"lost"}]}}}`},
		{`{"type":"response.created","response":"not-an-object"}`, `{"type":"response.completed","response":{"output":[]}}`},
		{`{"type":"response.in_progress","response":[]}`, `{"type":"response.completed","response":{"output":[]}}`},
	} {
		var output []string
		result := ResponsesSSEToOpenAIChunksObserved(pullItems(source), "m", func(raw string) { output = append(output, raw) }, StreamObservationOptions{})
		if result.Completed || result.Err == nil {
			t.Fatalf("malformed response lost with successful terminal: %+v", result)
		}
		for _, chunk := range bridgeJSON(t, output) {
			for _, rawChoice := range mustList(chunk["choices"]) {
				choice, _ := asMap(rawChoice)
				if present(choice["finish_reason"]) {
					t.Fatalf("malformed response emitted finish: %v", chunk)
				}
			}
		}
	}
}
