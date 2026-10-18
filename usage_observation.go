package translate

import (
	"bytes"
	"encoding/json"
	"math"
	"math/big"
	"sort"
	"strconv"
	"strings"
)

type UsageDialect string

const (
	UsageDialectChat      UsageDialect = "chat"
	UsageDialectResponses UsageDialect = "responses"
	UsageDialectMessages  UsageDialect = "messages"
)

type CountPresence string

const (
	CountAbsent  CountPresence = "absent"
	CountPresent CountPresence = "present"
	CountInvalid CountPresence = "invalid"
)

type UsageProvenance string

const (
	UsageReported  UsageProvenance = "reported"
	UsageDerived   UsageProvenance = "derived"
	UsageSynthetic UsageProvenance = "synthetic"
)

type CompletionQuality string

const (
	CompletionUnknown   CompletionQuality = "unknown"
	CompletionPartial   CompletionQuality = "partial"
	CompletionComplete  CompletionQuality = "complete"
	CompletionTruncated CompletionQuality = "truncated"
	CompletionFailed    CompletionQuality = "failed"
)

type CountObservation struct {
	Value      int64
	Presence   CountPresence
	Provenance UsageProvenance
	Path       string
}

// UsageObservation describes source evidence, independent of target-required
// wire placeholders. Raw is a bounded, detached copy of usage members only.
type UsageObservation struct {
	Dialect                                                             UsageDialect
	Source                                                              string
	Quality                                                             CompletionQuality
	InputTokens, OutputTokens, TotalTokens                              CountObservation
	BaseInputTokens, CacheReadTokens, CacheWriteTokens, ReasoningTokens CountObservation
	CacheWrite5mTokens, CacheWrite1hTokens                              CountObservation
	Raw                                                                 map[string]json.RawMessage
	RawTruncated                                                        bool
	Issues                                                              []string
}

func ObserveUsage(dialect UsageDialect, raw any, quality CompletionQuality) UsageObservation {
	return ObserveUsageWithSource(dialect, raw, quality, UsageReported)
}

func ObserveUsageWithSource(dialect UsageDialect, raw any, quality CompletionQuality, source UsageProvenance) UsageObservation {
	o := UsageObservation{Dialect: dialect, Source: string(source), Quality: quality}
	absent := CountObservation{Presence: CountAbsent, Provenance: source}
	o.InputTokens = absent
	o.OutputTokens = absent
	o.TotalTokens = absent
	o.BaseInputTokens = absent
	o.CacheReadTokens = absent
	o.CacheWriteTokens = absent
	o.ReasoningTokens = absent
	o.CacheWrite5mTokens = absent
	o.CacheWrite1hTokens = absent
	u, ok := asMap(raw)
	if raw != nil && !ok {
		o.Issues = append(o.Issues, "usage must be an object")
	}
	read := func(path string) CountObservation { return observedCount(u, path, source) }
	switch dialect {
	case UsageDialectChat:
		o.InputTokens = read("prompt_tokens")
		o.OutputTokens = read("completion_tokens")
		o.TotalTokens = read("total_tokens")
		o.CacheReadTokens = read("prompt_tokens_details.cached_tokens")
		o.CacheWriteTokens = read("prompt_tokens_details.cache_write_tokens")
		o.ReasoningTokens = read("completion_tokens_details.reasoning_tokens")
	case UsageDialectResponses:
		o.InputTokens = read("input_tokens")
		o.OutputTokens = read("output_tokens")
		o.TotalTokens = read("total_tokens")
		o.CacheReadTokens = read("input_tokens_details.cached_tokens")
		o.CacheWriteTokens = read("input_tokens_details.cache_write_tokens")
		o.ReasoningTokens = read("output_tokens_details.reasoning_tokens")
	case UsageDialectMessages:
		o.BaseInputTokens = read("input_tokens")
		o.OutputTokens = read("output_tokens")
		o.CacheReadTokens = read("cache_read_input_tokens")
		o.CacheWriteTokens = read("cache_creation_input_tokens")
		o.CacheWrite5mTokens = read("cache_creation.ephemeral_5m_input_tokens")
		o.CacheWrite1hTokens = read("cache_creation.ephemeral_1h_input_tokens")
		o.InputTokens = deriveSum("input_tokens+cache_read_input_tokens+cache_creation_input_tokens", o.BaseInputTokens, o.CacheReadTokens, o.CacheWriteTokens)
		o.TotalTokens = deriveSum("inclusive_input+output_tokens", o.InputTokens, o.OutputTokens)
	default:
		o.Issues = append(o.Issues, "unknown usage dialect")
	}
	if dialect == UsageDialectChat || dialect == UsageDialectResponses {
		if o.TotalTokens.Presence == CountAbsent {
			o.TotalTokens = deriveSum("input+output", o.InputTokens, o.OutputTokens)
		}
		if allPresent(o.InputTokens, o.CacheReadTokens, o.CacheWriteTokens) {
			cache := deriveSum("cache_read+cache_write", o.CacheReadTokens, o.CacheWriteTokens)
			if cache.Presence == CountInvalid || cache.Value > o.InputTokens.Value {
				o.Issues = append(o.Issues, "cache subsets exceed inclusive input")
			} else {
				o.BaseInputTokens = CountObservation{Value: o.InputTokens.Value - cache.Value, Presence: CountPresent, Provenance: UsageDerived, Path: "input-cache_read-cache_write"}
			}
		}
		if allPresent(o.OutputTokens, o.ReasoningTokens) && o.ReasoningTokens.Value > o.OutputTokens.Value {
			o.Issues = append(o.Issues, "reasoning subset exceeds output")
		}
	}
	if dialect == UsageDialectMessages && allPresent(o.CacheWrite5mTokens, o.CacheWrite1hTokens, o.CacheWriteTokens) {
		ttl := deriveSum("cache_write_ttl", o.CacheWrite5mTokens, o.CacheWrite1hTokens)
		if ttl.Presence == CountInvalid || ttl.Value != o.CacheWriteTokens.Value {
			o.Issues = append(o.Issues, "cache write TTL children do not partition parent")
			// Keep raw source evidence; an inconsistent normalized partition
			// is invalid, never an invented default TTL allocation.
			o.CacheWrite5mTokens.Presence = CountInvalid
			o.CacheWrite1hTokens.Presence = CountInvalid
		}
	}
	for _, c := range []CountObservation{o.InputTokens, o.OutputTokens, o.TotalTokens, o.BaseInputTokens, o.CacheReadTokens, o.CacheWriteTokens, o.ReasoningTokens, o.CacheWrite5mTokens, o.CacheWrite1hTokens} {
		if c.Presence == CountInvalid {
			o.Issues = append(o.Issues, "invalid count at "+c.Path)
		}
	}
	o.Raw, o.RawTruncated = boundedUsageRaw(u)
	return o
}

func observedCount(u map[string]any, path string, source UsageProvenance) CountObservation {
	c := CountObservation{Presence: CountAbsent, Provenance: source, Path: path}
	var v any = u
	fields := splitFields(strings.ReplaceAll(path, ".", " "))
	for _, field := range fields {
		// Optional nested detail objects may be explicitly null. No child
		// measurement was supplied; a null numeric leaf remains invalid.
		if v == nil {
			return c
		}
		m, ok := asMap(v)
		if !ok {
			c.Presence = CountInvalid
			return c
		}
		var exists bool
		v, exists = m[field]
		if !exists {
			return c
		}
	}
	if n, ok := usageInteger(v); ok {
		c.Value = n
		c.Presence = CountPresent
	} else {
		c.Presence = CountInvalid
	}
	return c
}

func usageInteger(v any) (int64, bool) {
	switch n := v.(type) {
	case int:
		return int64(n), n >= 0
	case int64:
		return n, n >= 0
	case uint64:
		if n <= math.MaxInt64 {
			return int64(n), true
		}
	case json.Number:
		if len(n) > 64 || !json.Valid([]byte(n)) {
			return 0, false
		}
		if i := strings.IndexAny(string(n), "eE"); i >= 0 {
			e, err := strconv.Atoi(string(n)[i+1:])
			if err != nil || e > 19 || e < -64 {
				return 0, false
			}
		}
		r, ok := new(big.Rat).SetString(string(n))
		if ok && r.IsInt() && r.Num().IsInt64() && r.Sign() >= 0 {
			return r.Num().Int64(), true
		}
	case float64:
		// Beyond the exact JSON float range the original integer evidence
		// is unavailable. Use json.Number for larger native counters.
		if n >= 0 && n <= 1<<53 && math.Trunc(n) == n {
			return int64(n), true
		}
	}
	return 0, false
}

func allPresent(counts ...CountObservation) bool {
	for _, c := range counts {
		if c.Presence != CountPresent {
			return false
		}
	}
	return true
}

func deriveSum(path string, counts ...CountObservation) CountObservation {
	c := CountObservation{Presence: CountPresent, Provenance: UsageDerived, Path: path}
	for _, v := range counts {
		if v.Presence == CountInvalid {
			c.Value = 0
			c.Presence = CountInvalid
			return c
		}
		if v.Presence != CountPresent {
			c.Value = 0
			c.Presence = CountAbsent
			return c
		}
		if v.Value > math.MaxInt64-c.Value {
			c.Value = 0
			c.Presence = CountInvalid
			return c
		}
		c.Value += v.Value
	}
	return c
}

const maxUsageRawBytes = 16384

func boundedUsageRaw(u map[string]any) (map[string]json.RawMessage, bool) {
	if u == nil {
		return nil, false
	}
	keys := make([]string, 0, len(u))
	for k := range u {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	result := map[string]json.RawMessage{}
	size := 0
	truncated := false
	for _, k := range keys {
		b, err := json.Marshal(u[k])
		if err != nil || len(b) > 4096 || len(k) > 128 || len(result) >= 32 || size+len(k)+len(b) > maxUsageRawBytes {
			truncated = true
			continue
		}
		result[k] = b
		size += len(k) + len(b)
	}
	return result, truncated
}

// UsageAccumulator replaces cumulative source fields, retaining omitted fields.
// It is single-consumer state, like the existing stream converters.
type UsageAccumulator struct {
	dialect   UsageDialect
	raw       map[string]any
	evidence  map[string]any
	truncated bool
	invalid   bool
}

func NewUsageAccumulator(dialect UsageDialect) *UsageAccumulator {
	return &UsageAccumulator{dialect: dialect, raw: map[string]any{}, evidence: map[string]any{}}
}

func (a *UsageAccumulator) Merge(raw any) {
	if a.raw == nil {
		a.raw = map[string]any{}
	}
	if a.evidence == nil {
		a.evidence = map[string]any{}
	}
	u, ok := asMap(raw)
	if !ok {
		if raw != nil {
			a.invalid = true
		}
		return
	}
	mergeKnownUsage(a.raw, u)
	mergeUsageMembers(a.evidence, u)
	// Re-decode bounded evidence, so a stream cannot retain an unbounded
	// sequence of usage extensions or references to mutable caller objects.
	bounded, truncated := boundedUsageRaw(a.evidence)
	a.truncated = a.truncated || truncated
	a.evidence = map[string]any{}
	for k, b := range bounded {
		var v any
		d := json.NewDecoder(bytes.NewReader(b))
		d.UseNumber()
		if d.Decode(&v) == nil {
			a.evidence[k] = v
		}
	}
}

func mergeUsageMembers(target, source map[string]any) {
	for k, v := range source {
		if child, ok := asMap(v); ok {
			if child == nil {
				target[k] = nil
				continue
			}
			dst, _ := asMap(target[k])
			if dst == nil {
				dst = map[string]any{}
			}
			mergeUsageMembers(dst, child)
			target[k] = dst
		} else {
			target[k] = v
		}
	}
}

func (a *UsageAccumulator) Snapshot(quality CompletionQuality) UsageObservation {
	o := ObserveUsage(a.dialect, a.raw, quality)
	o.Raw, o.RawTruncated = boundedUsageRaw(a.evidence)
	o.RawTruncated = o.RawTruncated || a.truncated
	if a.invalid {
		o.Issues = append(o.Issues, "nonobject usage snapshot observed")
	}
	return o
}

func mergeKnownUsage(target, source map[string]any) {
	for _, key := range []string{"input_tokens", "prompt_tokens", "output_tokens", "completion_tokens", "total_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"} {
		if v, ok := source[key]; ok {
			target[key] = boundedCountValue(v)
		}
	}
	for key, fields := range map[string][]string{"prompt_tokens_details": {"cached_tokens", "cache_write_tokens", "audio_tokens"}, "input_tokens_details": {"cached_tokens", "cache_write_tokens"}, "completion_tokens_details": {"reasoning_tokens", "audio_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens"}, "output_tokens_details": {"reasoning_tokens"}, "cache_creation": {"ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"}, "server_tool_use": {"web_search_requests", "web_fetch_requests"}} {
		v, exists := source[key]
		if !exists {
			continue
		}
		child, ok := asMap(v)
		if !ok || child == nil {
			if v == nil || (ok && child == nil) {
				// A nullable absent container supplies no cumulative child
				// measurements; retain earlier reported values like omission.
				continue
			} else {
				target[key] = false
			}
			continue
		}
		dst, _ := asMap(target[key])
		if dst == nil {
			dst = map[string]any{}
		}
		for _, field := range fields {
			if v, ok := child[field]; ok {
				if detailedUsageField(key, field) {
					if v == nil && nullableDetailedUsageField(key, field) {
						continue
					}
					dst[field] = boundedDetailValue(v)
				} else {
					dst[field] = boundedCountValue(v)
				}
			}
		}
		target[key] = dst
	}
}

func boundedCountValue(v any) any {
	// Invalid values need only a marker in authoritative normalized state;
	// their bounded original representation lives separately in Raw evidence.
	if _, ok := usageInteger(v); ok {
		return v
	}
	return nil
}
