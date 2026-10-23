package translate

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// Compile-time compatibility guards: the existing public layouts remain usable
// by callers with unkeyed literals; new detail fields belong to the sidecar.
var _ = UsageObservation{UsageDialectChat, "", CompletionUnknown,
	CountObservation{}, CountObservation{}, CountObservation{}, CountObservation{},
	CountObservation{}, CountObservation{}, CountObservation{}, CountObservation{}, CountObservation{}, nil, false, nil}
var _ = Usage{0, 0, 0, 0}

type detailCase struct {
	name          string
	dialect       UsageDialect
	parent, field string
	get           func(UsageDetails) CountObservation
	nullable      bool
}

var detailCases = []detailCase{
	{"input audio", UsageDialectChat, "prompt_tokens_details", "audio_tokens", func(d UsageDetails) CountObservation { return d.InputAudioTokens }, true},
	{"output audio", UsageDialectChat, "completion_tokens_details", "audio_tokens", func(d UsageDetails) CountObservation { return d.OutputAudioTokens }, true},
	{"accepted prediction", UsageDialectChat, "completion_tokens_details", "accepted_prediction_tokens", func(d UsageDetails) CountObservation { return d.AcceptedPredictionTokens }, true},
	{"rejected prediction", UsageDialectChat, "completion_tokens_details", "rejected_prediction_tokens", func(d UsageDetails) CountObservation { return d.RejectedPredictionTokens }, true},
	{"web search", UsageDialectMessages, "server_tool_use", "web_search_requests", func(d UsageDetails) CountObservation { return d.WebSearchRequests }, false},
	{"web fetch beta", UsageDialectMessages, "server_tool_use", "web_fetch_requests", func(d UsageDetails) CountObservation { return d.WebFetchRequests }, false},
}

func TestUsageDetailsDeclaredPathsPresenceAndProvenance(t *testing.T) {
	for _, tc := range detailCases {
		t.Run(tc.name, func(t *testing.T) {
			for _, sample := range []struct {
				value any
				want  int64
			}{{0, 0}, {json.Number("9007199254740993"), 9007199254740993}} {
				u := map[string]any{tc.parent: map[string]any{tc.field: sample.value}}
				o := ObserveUsageWithDetails(tc.dialect, u, CompletionPartial)
				c := tc.get(o.Details)
				if c.Presence != CountPresent || c.Value != sample.want || c.Provenance != UsageReported || c.Path != tc.parent+"."+tc.field || o.Quality != CompletionPartial {
					t.Fatalf("reported detail lost: %+v %+v", c, o)
				}
				synthetic := ObserveUsageWithSourceAndDetails(tc.dialect, u, CompletionFailed, UsageSynthetic)
				if c = tc.get(synthetic.Details); c.Provenance != UsageSynthetic || c.Presence != CountPresent {
					t.Fatalf("synthetic detail labeled measured: %+v", c)
				}
			}
			for _, u := range []any{nil, map[string]any{}, map[string]any{tc.parent: nil}, map[string]any{tc.parent: map[string]any{}}} {
				o := ObserveUsageWithDetails(tc.dialect, u, CompletionUnknown)
				c := tc.get(o.Details)
				if c.Presence != CountAbsent || c.Path != tc.parent+"."+tc.field {
					t.Fatalf("missing detail became zero/invalid: raw=%#v count=%+v", u, c)
				}
			}
			c := tc.get(ObserveUsageWithDetails(tc.dialect, map[string]any{tc.parent: map[string]any{tc.field: nil}}, CompletionPartial).Details)
			want := CountInvalid
			if tc.nullable {
				want = CountAbsent
			}
			if c.Presence != want {
				t.Fatalf("schema nullability lost: %+v want=%s", c, want)
			}
		})
	}
}

func TestUsageDetailsInvalidAndOverflow(t *testing.T) {
	for _, tc := range detailCases {
		t.Run(tc.name, func(t *testing.T) {
			for _, value := range []any{-1, 1.5, "3", false, json.Number("9223372036854775808"), json.Number("1e99999999")} {
				o := ObserveUsageWithDetails(tc.dialect, map[string]any{tc.parent: map[string]any{tc.field: value}}, CompletionPartial)
				c := tc.get(o.Details)
				if c.Presence != CountInvalid || len(o.Issues) == 0 {
					t.Fatalf("invalid/overflow count accepted: value=%#v count=%+v issues=%v", value, c, o.Issues)
				}
			}
			for _, parent := range []any{false, "invalid", []any{}} {
				o := ObserveUsageWithDetails(tc.dialect, map[string]any{tc.parent: parent}, CompletionPartial)
				if tc.get(o.Details).Presence != CountInvalid {
					t.Fatalf("malformed parent masked: %+v", o)
				}
			}
		})
	}
}

func TestUsageDetailsParentRelationshipsDoNotAddTotals(t *testing.T) {
	raw := map[string]any{"prompt_tokens": 4, "completion_tokens": 10, "total_tokens": 14, "prompt_tokens_details": map[string]any{"audio_tokens": 5}, "completion_tokens_details": map[string]any{"audio_tokens": 11, "accepted_prediction_tokens": 8, "rejected_prediction_tokens": 4}}
	o := ObserveUsageWithDetails(UsageDialectChat, raw, CompletionComplete)
	if len(o.Issues) < 3 || o.TotalTokens.Value != 14 || o.OutputTokens.Value != 10 || o.Details.InputAudioTokens.Value != 5 || o.Details.InputAudioTokens.Presence != CountPresent {
		t.Fatalf("parent contradiction lost or details counted twice: %+v", o)
	}
	raw["prompt_tokens_details"] = map[string]any{"audio_tokens": 3}
	raw["completion_tokens_details"] = map[string]any{"audio_tokens": 9, "reasoning_tokens": 9, "accepted_prediction_tokens": 7, "rejected_prediction_tokens": 3}
	o = ObserveUsageWithDetails(UsageDialectChat, raw, CompletionComplete)
	if len(o.Issues) != 0 || o.TotalTokens.Value != 14 {
		t.Fatalf("undocumented disjoint partition invented: %+v", o)
	}
	o = ObserveUsageWithDetails(UsageDialectChat, map[string]any{"completion_tokens": json.Number("9223372036854775807"), "completion_tokens_details": map[string]any{"accepted_prediction_tokens": json.Number("9223372036854775807"), "rejected_prediction_tokens": 1}}, CompletionComplete)
	if len(o.Issues) == 0 || o.Details.AcceptedPredictionTokens.Presence != CountPresent {
		t.Fatalf("prediction sum overflow not reported: %+v", o)
	}
	web := ObserveUsageWithDetails(UsageDialectMessages, map[string]any{"input_tokens": 0, "output_tokens": 0, "server_tool_use": map[string]any{"web_search_requests": 3, "web_fetch_requests": 2}}, CompletionComplete)
	if len(web.Issues) != 0 || web.Details.WebSearchRequests.Value != 3 || web.Details.WebFetchRequests.Value != 2 || web.OutputTokens.Value != 0 {
		t.Fatalf("operation counts equated to tokens: %+v", web)
	}
}

func TestUsageDetailsNullableContainersAndUnestablishedDialects(t *testing.T) {
	o := ObserveUsageWithDetails(UsageDialectChat, map[string]any{"prompt_tokens": 2, "completion_tokens": 1, "prompt_tokens_details": nil, "completion_tokens_details": nil}, CompletionComplete)
	if len(o.Issues) != 0 || o.Details.InputAudioTokens.Presence != CountAbsent || o.Details.OutputAudioTokens.Presence != CountAbsent {
		t.Fatalf("nullable absent details called invalid: %+v", o)
	}
	o = ObserveUsageWithDetails(UsageDialectResponses, map[string]any{"input_tokens_details": map[string]any{"audio_tokens": 4}, "output_tokens_details": map[string]any{"audio_tokens": 5, "accepted_prediction_tokens": 3}, "server_tool_use": map[string]any{"web_search_requests": 1, "web_fetch_requests": 1}}, CompletionComplete)
	for _, tc := range detailCases {
		if c := tc.get(o.Details); c.Presence != CountAbsent || c.Path != "" {
			t.Fatalf("unestablished Responses mapping invented: %+v", c)
		}
	}
}

func TestUsageDetailsCumulativeRetentionAndEvidenceEviction(t *testing.T) {
	for _, tc := range detailCases {
		t.Run(tc.name, func(t *testing.T) {
			a := NewUsageAccumulator(tc.dialect)
			a.Merge(map[string]any{tc.parent: map[string]any{tc.field: 7}})
			a.Merge(map[string]any{"unrelated": 1})
			if c := tc.get(a.SnapshotWithDetails(CompletionPartial).Details); c.Presence != CountPresent || c.Value != 7 {
				t.Fatalf("omitted cumulative count lost: %+v", c)
			}
			a.Merge(map[string]any{tc.parent: nil})
			if c := tc.get(a.SnapshotWithDetails(CompletionPartial).Details); c.Presence != CountPresent || c.Value != 7 {
				t.Fatalf("nullable absent container erased cumulative evidence: %+v", c)
			}
			if tc.nullable {
				a.Merge(map[string]any{tc.parent: map[string]any{tc.field: nil}})
				if c := tc.get(a.SnapshotWithDetails(CompletionPartial).Details); c.Presence != CountPresent || c.Value != 7 {
					t.Fatalf("nullable absent leaf erased cumulative evidence: %+v", c)
				}
			}
			a.Merge(map[string]any{tc.parent: map[string]any{tc.field: 0}})
			if c := tc.get(a.SnapshotWithDetails(CompletionTruncated).Details); c.Presence != CountPresent || c.Value != 0 {
				t.Fatalf("explicit zero did not replace: %+v", c)
			}
			a.Merge(map[string]any{tc.parent: map[string]any{tc.field: "invalid"}})
			if c := tc.get(a.SnapshotWithDetails(CompletionFailed).Details); c.Presence != CountInvalid {
				t.Fatalf("invalid update masked by nullable absence: %+v", c)
			}
			a.Merge(map[string]any{tc.parent: map[string]any{tc.field: json.Number("9007199254740993"), "oversized": strings.Repeat("x", 20000)}})
			extensions := map[string]any{}
			for i := 0; i < 40; i++ {
				extensions[fmt.Sprintf("a%02d", i)] = 0
			}
			a.Merge(extensions)
			o := a.SnapshotWithDetails(CompletionFailed)
			c := tc.get(o.Details)
			if !o.RawTruncated || c.Presence != CountPresent || c.Value != 9007199254740993 || c.Provenance != UsageReported || o.Quality != CompletionFailed {
				t.Fatalf("bounded evidence erased authoritative detail: %+v", o)
			}
			for key := range o.Raw {
				o.Raw[key] = json.RawMessage(`0`)
			}
			o.Details = UsageDetails{}
			if c := tc.get(a.SnapshotWithDetails(CompletionPartial).Details); c.Presence != CountPresent || c.Value != 9007199254740993 {
				t.Fatalf("snapshot aliases normalized counters: %+v", c)
			}
			a.Merge(map[string]any{tc.parent: false})
			if c := tc.get(a.SnapshotWithDetails(CompletionPartial).Details); c.Presence != CountInvalid {
				t.Fatalf("malformed parent update masked: %+v", c)
			}
			a.Merge(map[string]any{tc.parent: nil})
			if c := tc.get(a.SnapshotWithDetails(CompletionPartial).Details); c.Presence != CountInvalid {
				t.Fatalf("absence erased prior invalid evidence without a new measurement: %+v", c)
			}
		})
	}
}

func TestUsageDetailsNativeAndSnapshotDetachment(t *testing.T) {
	raw := map[string]any{"prompt_tokens": 10, "completion_tokens": 10, "completion_tokens_details": map[string]any{"accepted_prediction_tokens": 2, "rejected_prediction_tokens": 3}}
	before, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	o := ObserveUsageWithDetails(UsageDialectChat, raw, CompletionComplete)
	for _, value := range o.Raw {
		if len(value) > 0 {
			value[0] = '9'
		}
	}
	o.Details.AcceptedPredictionTokens.Value = 99
	after, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatalf("source mutated through observation: before=%s after=%s", before, after)
	}
	a := NewUsageAccumulator(UsageDialectChat)
	a.Merge(raw)
	snapshot := a.SnapshotWithDetails(CompletionPartial)
	for _, value := range snapshot.Raw {
		if len(value) > 0 {
			value[0] = '9'
		}
	}
	snapshot.Details.RejectedPredictionTokens.Value = 99
	next := a.SnapshotWithDetails(CompletionComplete)
	if next.Details.AcceptedPredictionTokens.Value != 2 || next.Details.RejectedPredictionTokens.Value != 3 || string(next.Raw["completion_tokens_details"]) != `{"accepted_prediction_tokens":2,"rejected_prediction_tokens":3}` {
		t.Fatalf("snapshot aliases retained source data: %+v", next)
	}
}
