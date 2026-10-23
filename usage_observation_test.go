package translate

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"
)

func TestUsageObservationPresenceAndTargetPlaceholders(t *testing.T) {
	for _, dialect := range []UsageDialect{UsageDialectChat, UsageDialectResponses, UsageDialectMessages} {
		o := ObserveUsage(dialect, nil, CompletionUnknown)
		for _, c := range []CountObservation{o.InputTokens, o.OutputTokens, o.TotalTokens, o.BaseInputTokens, o.CacheReadTokens, o.CacheWriteTokens, o.ReasoningTokens, o.CacheWrite5mTokens, o.CacheWrite1hTokens} {
			if c.Presence != CountAbsent {
				t.Errorf("dialect=%s absent=%+v", dialect, c)
			}
		}
	}
	for _, raw := range []string{`{"input_tokens":0,"output_tokens":0,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}`, `{"input_tokens":11,"output_tokens":5,"input_tokens_details":{"cached_tokens":4,"cache_write_tokens":3},"output_tokens_details":{"reasoning_tokens":2}}`} {
		o := ObserveUsage(UsageDialectResponses, decodeJSON[map[string]any](t, raw), CompletionComplete)
		if o.InputTokens.Presence != CountPresent || o.CacheReadTokens.Presence != CountPresent || o.CacheWriteTokens.Presence != CountPresent || o.TotalTokens.Provenance != UsageDerived {
			t.Fatalf("observation=%+v", o)
		}
		if o.InputTokens.Value > 0 && (o.TotalTokens.Value != 16 || o.BaseInputTokens.Value != 4) {
			t.Fatalf("overlapping subsets counted twice: %+v", o)
		}
	}
	chat := map[string]any{"usage": map[string]any{"prompt_tokens": 11, "completion_tokens": 5}}
	native := ObserveUsage(UsageDialectChat, chat["usage"], CompletionComplete)
	if native.CacheReadTokens.Presence != CountAbsent {
		t.Fatalf("native source invented detail: %+v", native)
	}
	wire := ChatResponseToResponses("m", chat)["usage"]
	synthetic := ObserveUsageWithSource(UsageDialectResponses, wire, CompletionComplete, UsageSynthetic)
	if synthetic.CacheReadTokens.Presence != CountPresent || synthetic.CacheReadTokens.Provenance != UsageSynthetic {
		t.Fatalf("placeholder labeled reported: %+v", synthetic)
	}
}

func TestUsageObservationDisjointAndTTL(t *testing.T) {
	raw := decodeJSON[map[string]any](t, `{"input_tokens":7,"cache_read_input_tokens":11,"cache_creation_input_tokens":13,"output_tokens":5,"cache_creation":{"ephemeral_5m_input_tokens":5,"ephemeral_1h_input_tokens":8}}`)
	o := ObserveUsage(UsageDialectMessages, raw, CompletionComplete)
	if o.InputTokens.Value != 31 || o.TotalTokens.Value != 36 || o.CacheWrite5mTokens.Value != 5 || len(o.Issues) != 0 {
		t.Fatalf("disjoint/TTL=%+v", o)
	}
	raw["cache_creation"] = map[string]any{"ephemeral_5m_input_tokens": 5, "ephemeral_1h_input_tokens": 7}
	o = ObserveUsage(UsageDialectMessages, raw, CompletionPartial)
	if len(o.Issues) == 0 || o.CacheWrite1hTokens.Presence != CountInvalid || o.CacheWriteTokens.Value != 13 {
		t.Fatalf("invalid partition=%+v", o)
	}
	delete(raw, "cache_read_input_tokens")
	if o = ObserveUsage(UsageDialectMessages, raw, CompletionPartial); o.InputTokens.Presence != CountAbsent {
		t.Fatalf("missing category treated zero: %+v", o)
	}
}

func TestUsageObservationInvalidAndCheckedDerivation(t *testing.T) {
	for _, v := range []any{nil, -1, 1.5, "4", true, math.Inf(1), float64(1<<53) + 2, json.Number("9223372036854775808")} {
		o := ObserveUsage(UsageDialectResponses, map[string]any{"input_tokens": v}, CompletionPartial)
		if o.InputTokens.Presence != CountInvalid || len(o.Issues) == 0 {
			t.Fatalf("invalid=%#v observation=%+v", v, o)
		}
	}
	o := ObserveUsage(UsageDialectResponses, map[string]any{"input_tokens": json.Number("9223372036854775807"), "output_tokens": 1}, CompletionTruncated)
	if o.TotalTokens.Presence != CountInvalid || o.Quality != CompletionTruncated {
		t.Fatalf("overflow/quality=%+v", o)
	}
	o = ObserveUsage(UsageDialectResponses, map[string]any{"input_tokens": json.Number("9007199254740993"), "output_tokens": 0}, CompletionComplete)
	if o.InputTokens.Value != 9007199254740993 || o.InputTokens.Presence != CountPresent {
		t.Fatalf("precise integer=%+v", o)
	}
}

func TestUsageAccumulatorReplacesCumulativeCountsAndBoundsEvidence(t *testing.T) {
	a := NewUsageAccumulator(UsageDialectMessages)
	a.Merge(decodeJSON[map[string]any](t, `{"input_tokens":7,"cache_creation_input_tokens":13,"cache_read_input_tokens":11,"output_tokens":2}`))
	a.Merge(map[string]any{"output_tokens": 5})
	if o := a.Snapshot(CompletionPartial); o.TotalTokens.Value != 36 {
		t.Fatalf("cumulative added rather than replaced: %+v", o)
	}
	a.Merge(map[string]any{"output_tokens": 0})
	o := a.Snapshot(CompletionTruncated)
	if o.OutputTokens.Presence != CountPresent || o.OutputTokens.Value != 0 || o.InputTokens.Value != 31 || o.Quality != CompletionTruncated {
		t.Fatalf("zero/omitted=%+v", o)
	}
	a.Merge(map[string]any{"output_tokens": nil})
	if o = a.Snapshot(CompletionFailed); o.OutputTokens.Presence != CountInvalid {
		t.Fatalf("explicit invalid retained previous count: %+v", o)
	}
	a.Merge(map[string]any{"unreviewed_large": strings.Repeat("x", 20000)})
	o = a.Snapshot(CompletionFailed)
	if !o.RawTruncated {
		t.Fatal("unbounded raw evidence")
	}
	for k := range o.Raw {
		o.Raw[k] = json.RawMessage(`0`)
	}
	if next := a.Snapshot(CompletionFailed); string(next.Raw["input_tokens"]) != "7" {
		t.Fatal("snapshot aliases retained source")
	}
}

func TestUsageAccumulatorCounterSurvivesEvidenceEviction(t *testing.T) {
	a := NewUsageAccumulator(UsageDialectMessages)
	a.Merge(map[string]any{"output_tokens": 7})
	extensions := map[string]any{}
	for i := 0; i < 40; i++ {
		extensions[fmt.Sprintf("a%02d", i)] = 0
	}
	a.Merge(extensions)
	o := a.Snapshot(CompletionPartial)
	if o.OutputTokens.Presence != CountPresent || o.OutputTokens.Value != 7 || !o.RawTruncated {
		t.Fatalf("evidence cap evicted counter: %+v", o)
	}
	a.Merge(map[string]any{"cache_creation": map[string]any{"ephemeral_5m_input_tokens": 5, "huge": strings.Repeat("x", 20000)}})
	o = a.Snapshot(CompletionPartial)
	if o.CacheWrite5mTokens.Presence != CountPresent || o.CacheWrite5mTokens.Value != 5 {
		t.Fatalf("nested evidence cap evicted recognized counter: %+v", o)
	}
}
