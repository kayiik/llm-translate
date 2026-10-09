package translate

import (
	"errors"
	"testing"
)

func TestAnthropicStreamReportTypesFailures(t *testing.T) {
	translateLines := func(lines ...string) *StreamReport {
		return AnthropicSSEToOpenAIChunksWithOptions(pullItems(lines), "m", func(string) {}, StreamOptions{})
	}

	t.Run("error event", func(t *testing.T) {
		report := translateLines(`event: error`, `data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
		var streamErr *StreamError
		if !errors.As(report.Err, &streamErr) || *streamErr != (StreamError{Type: "overloaded_error", Message: "Overloaded"}) || report.Completed {
			t.Fatalf("got %+v", report)
		}
	})

	t.Run("first error event wins over later events", func(t *testing.T) {
		report := translateLines(
			`data: {"type":"error","error":{"type":"api_error","message":"first"}}`,
			`data: {"type":"error","error":{"type":"api_error","message":"second"}}`,
			`data: {"type":"message_stop"}`)
		var streamErr *StreamError
		if !errors.As(report.Err, &streamErr) || streamErr.Message != "first" || report.Completed {
			t.Fatalf("got %+v", report)
		}
	})

	t.Run("source ends without message_stop", func(t *testing.T) {
		report := translateLines(
			`data: {"type":"message_start","message":{"usage":{"input_tokens":4,"output_tokens":1}}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"cut"}}`)
		if !errors.Is(report.Err, ErrStreamTruncated) || report.Completed || report.Usage.InputTokens != 4 {
			t.Fatalf("got %+v", report)
		}
	})

	t.Run("message_stop completes", func(t *testing.T) {
		if report := translateLines(`data: {"type":"message_stop"}`); report.Err != nil || !report.Completed {
			t.Fatalf("got %+v", report)
		}
	})
}
