package translate

import (
	"encoding/json"
	"testing"
)

func TestChatResponsesUsagePreservesJSONNumbers(t *testing.T) {
	for _, tc := range []struct{ input, output, total string }{
		{"1", "1", "2"},
		{"9007199254740993", "0", "9007199254740993"},
		{"9223372036854775807", "0", "9223372036854775807"},
	} {
		chat := map[string]any{"id": "chatcmpl-usage", "usage": map[string]any{"prompt_tokens": json.Number(tc.input), "completion_tokens": json.Number(tc.output), "total_tokens": json.Number(tc.total)}}
		for name, value := range map[string]map[string]any{
			"plain":            ChatResponseToResponses("m", chat),
			"reported":         ChatResponseToResponsesWithReport("m", chat).Value,
			"request plain":    ChatResponseToResponsesWithRequest("m", chat, map[string]any{}),
			"request reported": ChatResponseToResponsesWithRequestAndReport("m", chat, map[string]any{}).Value,
		} {
			usage, ok := asMap(value["usage"])
			if !ok {
				t.Fatalf("%s lost valid usage: %v", name, value)
			}
			encoded, err := json.Marshal(usage)
			if err != nil {
				t.Fatal(err)
			}
			decoded, ok := decodeObservedObject(string(encoded))
			if !ok {
				t.Fatalf("invalid target usage: %s", encoded)
			}
			for key, want := range map[string]string{"input_tokens": tc.input, "output_tokens": tc.output, "total_tokens": tc.total} {
				if decoded[key] != json.Number(want) {
					t.Fatalf("%s %s=%#v want exact %s (usage=%s)", name, key, decoded[key], want, encoded)
				}
			}
		}
		if report := ChatResponseToResponsesWithReport("m", chat).Report; report.HasMaterialLoss() {
			t.Fatalf("valid counts rejected: %+v", report)
		}
	}
}

func TestChatResponsesUsageRejectsInvalidCountsWithoutZero(t *testing.T) {
	for _, tc := range []struct {
		key   string
		value any
	}{
		{"prompt_tokens", nil}, {"prompt_tokens", "1"}, {"prompt_tokens", -1}, {"completion_tokens", 1.5},
		{"prompt_tokens", json.Number("9223372036854775808")}, {"total_tokens", json.Number("bad")},
	} {
		u := map[string]any{"prompt_tokens": json.Number("1"), "completion_tokens": json.Number("1"), "total_tokens": json.Number("2")}
		u[tc.key] = tc.value
		chat := map[string]any{"usage": u}
		plain := ChatResponseToResponses("m", chat)
		reported := ChatResponseToResponsesWithReport("m", chat)
		if plain["usage"] != nil || reported.Value["usage"] != nil {
			t.Fatalf("invalid source manufactured target zeros: key=%s raw=%#v plain=%v", tc.key, tc.value, plain["usage"])
		}
		requireRecoveryLoss(t, reported.Report, "usage."+tc.key, LossUnsupported)
	}
	chat := map[string]any{"usage": map[string]any{"prompt_tokens": json.Number("9223372036854775807"), "completion_tokens": json.Number("1")}}
	result := ChatResponseToResponsesWithReport("m", chat)
	if result.Value["usage"] != nil {
		t.Fatalf("derived total overflow emitted a value: %v", result.Value["usage"])
	}
	requireRecoveryLoss(t, result.Report, "usage.total_tokens", LossUnsupported)
}

func TestResponsesChatUsagePreservesExactCountsAndReportedZero(t *testing.T) {
	for _, tc := range []struct{ input, output, total string }{
		{"1", "1", "2"}, {"0", "0", "0"}, {"1", "1", "0"},
		{"9007199254740993", "0", "9007199254740993"},
		{"9223372036854775807", "0", "9223372036854775807"},
	} {
		source := map[string]any{"id": "resp-counts", "output": []any{}, "usage": map[string]any{"input_tokens": json.Number(tc.input), "output_tokens": json.Number(tc.output), "total_tokens": json.Number(tc.total)}}
		reported := ResponsesToChatWithReport("m", source)
		for name, value := range map[string]map[string]any{"plain": ResponsesToChat("m", source), "reported": reported.Value} {
			encoded, err := json.Marshal(value["usage"])
			if err != nil {
				t.Fatal(err)
			}
			usage, ok := decodeObservedObject(string(encoded))
			if !ok {
				t.Fatalf("%s lost valid usage: %s", name, encoded)
			}
			for key, want := range map[string]string{"prompt_tokens": tc.input, "completion_tokens": tc.output, "total_tokens": tc.total} {
				if usage[key] != json.Number(want) {
					t.Fatalf("%s %s=%#v want exact %s", name, key, usage[key], want)
				}
			}
		}
		if reported.Report.HasMaterialLoss() {
			t.Fatalf("valid source counts rejected: %+v", reported.Report)
		}
	}
}

func TestResponsesChatUsageRejectsInvalidAggregatesWithoutZero(t *testing.T) {
	for _, tc := range []struct {
		key   string
		value any
	}{
		{"input_tokens", nil}, {"input_tokens", -1}, {"output_tokens", 1.5}, {"total_tokens", "2"},
		{"input_tokens", json.Number("9223372036854775808")}, {"total_tokens", json.Number("invalid")},
	} {
		usage := map[string]any{"input_tokens": json.Number("1"), "output_tokens": json.Number("1"), "total_tokens": json.Number("2")}
		usage[tc.key] = tc.value
		source := map[string]any{"output": []any{}, "usage": usage}
		reported := ResponsesToChatWithReport("m", source)
		for name, value := range map[string]map[string]any{"plain": ResponsesToChat("m", source), "reported": reported.Value} {
			if _, exists := value["usage"]; exists {
				t.Errorf("%s fabricated usage from invalid source %s=%#v: %v", name, tc.key, tc.value, value["usage"])
			}
		}
		requireRecoveryLoss(t, reported.Report, "usage."+tc.key, LossUnsupported)
	}
	source := map[string]any{"output": []any{}, "usage": map[string]any{"input_tokens": json.Number("9223372036854775807"), "output_tokens": json.Number("1")}}
	result := ResponsesToChatWithReport("m", source)
	if _, exists := result.Value["usage"]; exists {
		t.Fatalf("derived overflow emitted usage: %v", result.Value["usage"])
	}
	requireRecoveryLoss(t, result.Report, "usage.total_tokens", LossUnsupported)
}

func TestResponsesChatUsageMissingFallbackAndSubsetDetails(t *testing.T) {
	source := map[string]any{"output": []any{}, "usage": map[string]any{"input_tokens": json.Number("11"), "output_tokens": json.Number("5"), "input_tokens_details": map[string]any{"cached_tokens": json.Number("4"), "cache_write_tokens": json.Number("3")}, "output_tokens_details": map[string]any{"reasoning_tokens": json.Number("2")}}}
	result := ResponsesToChatWithReport("m", source)
	usage := result.Value["usage"].(map[string]any)
	if usage["prompt_tokens"] != 11 || usage["completion_tokens"] != 5 || usage["total_tokens"] != 16 {
		t.Fatalf("missing-total fallback/subsets changed aggregate: %v", usage)
	}
	input := usage["prompt_tokens_details"].(map[string]any)
	output := usage["completion_tokens_details"].(map[string]any)
	if input["cached_tokens"] != json.Number("4") || input["cache_write_tokens"] != json.Number("3") || output["reasoning_tokens"] != json.Number("2") {
		t.Fatalf("subset details lost: %v", usage)
	}
	if result.Report.HasMaterialLoss() {
		t.Fatalf("valid sparse usage rejected: %+v", result.Report)
	}
	for _, raw := range []any{nil, map[string]any{}, map[string]any{"input_tokens": 0}} {
		value := ResponsesToChat("m", map[string]any{"usage": raw})
		if raw == nil {
			if _, exists := value["usage"]; exists {
				t.Fatal("absent source usage fabricated an object")
			}
		} else if usage := value["usage"].(map[string]any); usage["prompt_tokens"] != 0 || usage["completion_tokens"] != 0 || usage["total_tokens"] != 0 {
			t.Fatalf("legacy absent-field placeholders changed: %v", usage)
		}
	}
}
