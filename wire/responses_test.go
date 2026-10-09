package wire

import "testing"

func TestResponsesRequestRoundTrip(t *testing.T) {
	r := assertRoundTrip(t, fixture(t, "responses_request.json"), DecodeResponsesRequest)
	if r.Instructions == nil || *r.Instructions != "" || r.Temperature != "0.20" || r.Store == nil || *r.Store {
		t.Fatalf("scalars: %+v", r)
	}
	if r.ToolChoice == nil || r.ToolChoice.Name != "lookup" || r.Reasoning.Effort != "high" || r.Reasoning.Extra["x_reason"] == nil {
		t.Fatalf("tool_choice and reasoning: %+v %+v", r.ToolChoice, r.Reasoning)
	}
	if r.Tools[1].Type != "web_search" || r.Tools[1].Extra["search_context_size"] == nil {
		t.Fatalf("built-in tool: %+v", r.Tools[1])
	}
	items := r.Input.Items
	if items[0].Content.Parts[0].Extra["cache_control"] == nil || items[0].Content.Parts[1].ImageURL == "" {
		t.Fatalf("message parts: %+v", items[0].Content.Parts)
	}
	if items[1].EncryptedContent == nil || len(items[1].Summary) != 1 || string(items[1].Extra["content"]) != "null" {
		t.Fatalf("reasoning item: %+v", items[1])
	}
	if items[2].CallID != "call_1" || items[2].Extra["extra_content"] == nil {
		t.Fatalf("function_call keeps vendor members in Extra: %+v", items[2])
	}
	if items[3].Output.Parts == nil || items[4].Output.Text == nil || *items[4].Output.Text != "" {
		t.Fatalf("function outputs: %+v %+v", items[3].Output, items[4].Output)
	}
	if items[5].Type != "" || items[5].Content.Text == nil || items[5].Extra["reasoning_details"] == nil {
		t.Fatalf("typeless message: %+v", items[5])
	}
}

func TestResponsesStringInputRoundTrip(t *testing.T) {
	r := assertRoundTrip(t, []byte(`{"model":"m","input":"hello","tool_choice":"required","reasoning":{"effort":null}}`), DecodeResponsesRequest)
	if r.Input.Text == nil || *r.Input.Text != "hello" || r.ToolChoice.Mode != "required" || string(r.Reasoning.Extra["effort"]) != "null" {
		t.Fatalf("request: %+v", r)
	}
}

func TestResponsesResponseRoundTrip(t *testing.T) {
	r := assertRoundTrip(t, fixture(t, "responses_response.json"), DecodeResponsesResponse)
	if r.Status != "incomplete" || r.IncompleteDetails == nil || r.Instructions != nil || string(r.Extra["instructions"]) != "null" {
		t.Fatalf("envelope: %+v", r)
	}
	if r.Output[0].Content.Parts[0].Type != "reasoning_text" || r.Output[1].Content.Parts[1].Refusal == nil {
		t.Fatalf("output: %+v", r.Output)
	}
	if r.Output[2].Arguments == nil || *r.Output[2].Arguments != "" || r.Usage.TotalTokens != "120" {
		t.Fatalf("function call and usage: %+v %+v", r.Output[2], r.Usage)
	}
}

func TestResponsesEventsRoundTrip(t *testing.T) {
	events := assertEachRoundTrips(t, fixture(t, "responses_events.json"), DecodeResponsesEvent)
	if events[0].Response == nil || events[0].Response.Output == nil || string(events[0].Response.Extra["usage"]) != "null" {
		t.Fatalf("response.created: %+v", events[0])
	}
	if events[2].SummaryIndex != "0" || events[2].Extra["obfuscation"] == nil {
		t.Fatalf("summary delta: %+v", events[2])
	}
	if events[5].Delta == nil || *events[5].Delta != "" || events[6].Text == nil {
		t.Fatalf("text events: %+v %+v", events[5], events[6])
	}
	if events[9].Arguments == nil || events[10].Response.Usage.OutputTokens != "2" {
		t.Fatalf("done events: %+v %+v", events[9], events[10])
	}
}
