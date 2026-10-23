package translate

import "strings"

// UsageDetails contains only documented source details, not extra totals.
// Web request counters have operation units; all other fields have token units.
type UsageDetails struct {
	InputAudioTokens, OutputAudioTokens                CountObservation
	AcceptedPredictionTokens, RejectedPredictionTokens CountObservation
	WebSearchRequests, WebFetchRequests                CountObservation
}

// UsageObservationWithDetails extends observations without changing the layout
// of the existing UsageObservation or the legacy integer Usage type.
type UsageObservationWithDetails struct {
	UsageObservation
	Details UsageDetails
}

func ObserveUsageWithDetails(dialect UsageDialect, raw any, quality CompletionQuality) UsageObservationWithDetails {
	return ObserveUsageWithSourceAndDetails(dialect, raw, quality, UsageReported)
}

func ObserveUsageWithSourceAndDetails(dialect UsageDialect, raw any, quality CompletionQuality, source UsageProvenance) UsageObservationWithDetails {
	base := ObserveUsageWithSource(dialect, raw, quality, source)
	return observeDetails(base, raw, source)
}

func (a *UsageAccumulator) SnapshotWithDetails(quality CompletionQuality) UsageObservationWithDetails {
	return observeDetails(a.Snapshot(quality), a.raw, UsageReported)
}

func observeDetails(base UsageObservation, raw any, source UsageProvenance) UsageObservationWithDetails {
	absent := CountObservation{Presence: CountAbsent, Provenance: source}
	d := UsageDetails{InputAudioTokens: absent, OutputAudioTokens: absent, AcceptedPredictionTokens: absent, RejectedPredictionTokens: absent, WebSearchRequests: absent, WebFetchRequests: absent}
	u, _ := asMap(raw)
	read := func(path string, nullable bool) CountObservation {
		count := observedCount(u, path, source)
		// Chat's optional int details are nullable in the pinned schema.
		// A missing/null optional container is handled by observedCount;
		// a null numeric leaf is absence only where that leaf is nullable.
		if nullable && count.Presence == CountInvalid {
			parent, field, _ := strings.Cut(path, ".")
			if object, ok := asMap(u[parent]); ok {
				if value, exists := object[field]; exists && value == nil {
					count.Presence = CountAbsent
				}
			}
		}
		return count
	}
	switch base.Dialect {
	case UsageDialectChat:
		d.InputAudioTokens = read("prompt_tokens_details.audio_tokens", true)
		d.OutputAudioTokens = read("completion_tokens_details.audio_tokens", true)
		d.AcceptedPredictionTokens = read("completion_tokens_details.accepted_prediction_tokens", true)
		d.RejectedPredictionTokens = read("completion_tokens_details.rejected_prediction_tokens", true)
		for _, relation := range []struct{ child, parent CountObservation }{{d.InputAudioTokens, base.InputTokens}, {d.OutputAudioTokens, base.OutputTokens}, {d.AcceptedPredictionTokens, base.OutputTokens}, {d.RejectedPredictionTokens, base.OutputTokens}} {
			if allPresent(relation.child, relation.parent) && relation.child.Value > relation.parent.Value {
				base.Issues = append(base.Issues, relation.child.Path+" exceeds "+relation.parent.Path)
			}
		}
		if allPresent(d.AcceptedPredictionTokens, d.RejectedPredictionTokens, base.OutputTokens) {
			prediction := deriveSum("accepted_prediction_tokens+rejected_prediction_tokens", d.AcceptedPredictionTokens, d.RejectedPredictionTokens)
			if prediction.Presence == CountInvalid || prediction.Value > base.OutputTokens.Value {
				base.Issues = append(base.Issues, "prediction token subsets exceed completion_tokens")
			}
		}
	case UsageDialectMessages:
		// The stable pinned schema has search; its beta schema additionally
		// has fetch. A reported field is evidence, not feature availability.
		d.WebSearchRequests = read("server_tool_use.web_search_requests", false)
		d.WebFetchRequests = read("server_tool_use.web_fetch_requests", false)
	}
	for _, count := range []CountObservation{d.InputAudioTokens, d.OutputAudioTokens, d.AcceptedPredictionTokens, d.RejectedPredictionTokens, d.WebSearchRequests, d.WebFetchRequests} {
		if count.Presence == CountInvalid {
			base.Issues = append(base.Issues, "invalid count at "+count.Path)
		}
	}
	return UsageObservationWithDetails{UsageObservation: base, Details: d}
}

func detailedUsageField(parent, field string) bool {
	switch parent {
	case "prompt_tokens_details":
		return field == "audio_tokens"
	case "completion_tokens_details":
		return field == "audio_tokens" || field == "accepted_prediction_tokens" || field == "rejected_prediction_tokens"
	case "server_tool_use":
		return field == "web_search_requests" || field == "web_fetch_requests"
	}
	return false
}

func nullableDetailedUsageField(parent, field string) bool {
	return (parent == "prompt_tokens_details" || parent == "completion_tokens_details") && detailedUsageField(parent, field)
}

func boundedDetailValue(value any) any {
	if value == nil {
		return nil
	}
	if _, ok := usageInteger(value); ok {
		return value
	}
	// Unlike an optional null, false cannot be a valid numeric observation.
	// Retain invalidity without retaining unbounded malformed source values.
	return false
}
