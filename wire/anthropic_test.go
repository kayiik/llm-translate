package wire

import (
	"encoding/json"
	"testing"
)

func TestAnthropicRequestRoundTrip(t *testing.T) {
	r := assertRoundTrip(t, fixture(t, "anthropic_request.json"), DecodeAnthropicRequest)
	if r.Temperature != "1.0" || r.TopK != "40" || r.Stream == nil || *r.Stream || r.Extra["x_top"] == nil {
		t.Fatalf("scalars and Extra: %+v", r)
	}
	system := r.System.Blocks
	if len(system) != 2 || system[0].CacheControl == nil || system[0].CacheControl.TTL != "1h" || system[1].CacheControl != nil {
		t.Fatalf("system blocks: %+v", system)
	}
	if r.Tools[0].CacheControl == nil || r.Tools[1].Type != "web_search_20250305" || r.Tools[1].Extra["max_uses"] == nil {
		t.Fatalf("tools: %+v", r.Tools)
	}
	if r.ToolChoice.DisableParallelToolUse == nil || !*r.ToolChoice.DisableParallelToolUse {
		t.Fatalf("tool_choice: %+v", r.ToolChoice)
	}
	user := r.Messages[0].Content.Blocks
	if user[0].CacheControl.Extra["x_cc"] == nil || user[1].CacheControl == nil || user[2].Citations == nil {
		t.Fatalf("user blocks: %+v", user)
	}
	assistant := r.Messages[1].Content.Blocks
	if *assistant[0].Signature != "sig-a" || *assistant[1].Data != "opaque==" || assistant[2].Text == nil || assistant[3].Extra["x_block"] == nil {
		t.Fatalf("assistant blocks: %+v", assistant)
	}
	result := r.Messages[2].Content.Blocks[0]
	if result.CacheControl == nil || result.Content.Blocks[0].CacheControl == nil || result.IsError == nil || *result.IsError {
		t.Fatalf("tool_result: %+v", result)
	}
	if text := r.Messages[3].Content.Text; text == nil || *text != "string content" {
		t.Fatalf("string content: %+v", r.Messages[3])
	}
}

func TestAnthropicResponseRoundTrip(t *testing.T) {
	r := assertRoundTrip(t, fixture(t, "anthropic_response.json"), DecodeAnthropicResponse)
	if r.StopReason != "tool_use" || r.StopSequence != nil || string(r.Extra["stop_sequence"]) != "null" {
		t.Fatalf("stop fields: %+v", r)
	}
	if len(r.Content) != 5 || r.Content[4].Content == nil || r.Usage.CacheReadInputTokens != "80" {
		t.Fatalf("content and usage: %+v", r)
	}
}

func TestAnthropicEventsRoundTrip(t *testing.T) {
	events := assertEachRoundTrips(t, fixture(t, "anthropic_events.json"), DecodeAnthropicEvent)
	if events[0].Message == nil || events[0].Message.Content == nil || events[0].Message.Usage.InputTokens != "10" {
		t.Fatalf("message_start: %+v", events[0])
	}
	if block := events[1].ContentBlock; block == nil || block.Thinking == nil || *block.Thinking != "" || block.Signature == nil {
		t.Fatalf("thinking start keeps its empty strings: %+v", events[1])
	}
	if d := events[11].Delta; d == nil || d.PartialJSON == nil || *d.PartialJSON != `{"q":1.0}` {
		t.Fatalf("input_json_delta: %+v", events[11])
	}
	if d := events[12].Delta; d.StopReason != "tool_use" || events[12].Usage.OutputTokens != "15" {
		t.Fatalf("message_delta: %+v", events[12])
	}
}

func TestAnthropicMalformedMembersSurvive(t *testing.T) {
	// A member of the wrong shape must not fail the decode or be dropped.
	data := []byte(`{"model":"m","system":{"not":"blocks"},"messages":[{"role":"user","content":7}],"tools":"none","cache_control":true}`)
	r := assertRoundTrip(t, data, DecodeAnthropicRequest)
	if r.System != nil || r.Tools != nil || r.Messages[0].Content != nil {
		t.Fatalf("malformed members were modeled: %+v", r)
	}
	for _, key := range []string{"system", "tools", "cache_control"} {
		if _, ok := r.Extra[key]; !ok {
			t.Fatalf("%s not kept in Extra: %v", key, r.Extra)
		}
	}
	if _, err := DecodeAnthropicRequest([]byte(`[1]`)); err == nil {
		t.Fatal("a non-object request decoded")
	}
	var block AnthropicBlock
	if err := json.Unmarshal([]byte(`null`), &block); err == nil {
		t.Fatal("null decoded as a block")
	}
}
