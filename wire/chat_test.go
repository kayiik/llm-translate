package wire

import (
	"encoding/json"
	"testing"
)

func TestChatRequestRoundTrip(t *testing.T) {
	r := assertRoundTrip(t, fixture(t, "chat_request.json"), DecodeChatRequest)
	if r.Model != "gemini-test" || r.Temperature != "0.70" || r.MaxTokens != "1e3" || r.Seed != "12345678901234567890" {
		t.Fatalf("scalars: %+v", r)
	}
	if r.Stop == nil || r.Stop.Sequence == nil || *r.Stop.Sequence != "END" {
		t.Fatalf("stop: %+v", r.Stop)
	}
	if r.ToolChoice == nil || r.ToolChoice.Mode != "" || r.ToolChoice.Function.Name != "lookup" || string(r.ToolChoice.Extra["x_choice"]) != "1" {
		t.Fatalf("tool_choice: %+v", r.ToolChoice)
	}
	if _, ok := r.Extra["vendor_top"]; !ok {
		t.Fatalf("unknown top-level member not kept: %v", r.Extra)
	}
	system := r.Messages[0].Content.Parts[0]
	if system.CacheControl == nil || system.CacheControl.Type != "ephemeral" || system.CacheControl.TTL != "1h" {
		t.Fatalf("system part cache_control: %+v", system.CacheControl)
	}
	user := r.Messages[1]
	if user.Content.Parts[0].CacheControl == nil || user.Name != "" || string(user.Extra["name"]) != `""` {
		t.Fatalf("user message: %+v", user)
	}
	assistant := r.Messages[2]
	if assistant.Content != nil || string(assistant.Extra["content"]) != "null" {
		t.Fatalf("null content must stay in Extra: %+v", assistant)
	}
	var details []map[string]any
	if err := json.Unmarshal(assistant.ReasoningDetails, &details); err != nil || len(details) != 2 {
		t.Fatalf("reasoning_details: %s", assistant.ReasoningDetails)
	}
	call := assistant.ToolCalls[0]
	if call.ThoughtSignature() != "sig-thought-1" || call.ExtraContent.Extra["x_vendor"] == nil || call.ExtraContent.Google.Extra["x_google"] == nil {
		t.Fatalf("thought signature: %+v", call.ExtraContent)
	}
	if args := assistant.ToolCalls[1].Function.Arguments; args == nil || *args != "" {
		t.Fatalf("empty arguments must survive as a string: %v", args)
	}
}

func TestChatRequestFormsRoundTrip(t *testing.T) {
	r := assertRoundTrip(t, fixture(t, "chat_request_forms.json"), DecodeChatRequest)
	if r.ToolChoice == nil || r.ToolChoice.Mode != "auto" || r.Stop == nil || len(r.Stop.Sequences) != 2 {
		t.Fatalf("polymorphic forms: %+v %+v", r.ToolChoice, r.Stop)
	}
	if r.Stream == nil || !*r.Stream || r.Tools == nil || len(r.Tools) != 0 {
		t.Fatalf("stream and empty tools: %+v", r)
	}
	parts := r.Messages[1].Content.Parts
	if parts[2].Text == nil || *parts[2].Text != "" || parts[3].CacheControl != nil || string(parts[3].Extra["cache_control"]) != "null" {
		t.Fatalf("parts: %+v", parts)
	}
	assistant := r.Messages[2]
	if assistant.Content == nil || assistant.Content.Text != nil || assistant.Content.Parts == nil || assistant.ToolCalls == nil {
		t.Fatalf("empty arrays must stay arrays: %+v", assistant)
	}
}

func TestChatResponseRoundTrip(t *testing.T) {
	r := assertRoundTrip(t, fixture(t, "chat_response.json"), DecodeChatResponse)
	message := r.Choices[0].Message
	if message.ToolCalls[0].ThoughtSignature() != "sig-thought-2" || message.ReasoningDetails == nil {
		t.Fatalf("vendor fields: %+v", message)
	}
	if r.Usage.TotalTokens != "17" || r.SystemFingerprint != "" || string(r.Extra["system_fingerprint"]) != "null" {
		t.Fatalf("usage and null member: %+v", r)
	}
}

func TestChatChunksRoundTrip(t *testing.T) {
	chunks := assertEachRoundTrips(t, fixture(t, "chat_chunks.json"), DecodeChatChunk)
	if chunks[0].Choices[0].Delta.Content.Text == nil || chunks[0].Choices[0].FinishReason != "" {
		t.Fatalf("first chunk: %+v", chunks[0].Choices[0])
	}
	if chunks[1].Choices[0].Delta.ReasoningDetails == nil {
		t.Fatal("reasoning_details delta not modeled")
	}
	if chunks[2].Choices[0].Delta.ToolCalls[0].ThoughtSignature() != "sig-thought-3" {
		t.Fatal("thought signature delta not modeled")
	}
	if chunks[5].Usage == nil || chunks[5].Choices == nil || len(chunks[5].Choices) != 0 {
		t.Fatalf("final chunk: %+v", chunks[5])
	}
}

func TestChatEncodeFromScratch(t *testing.T) {
	message := ChatMessage{
		Role:    "assistant",
		Content: &ChatContent{Text: str("")},
		ToolCalls: []ChatToolCall{{
			ID: "call_1", Type: "function",
			Function:     &ChatFunctionCall{Name: "lookup", Arguments: str("{}")},
			ExtraContent: &ChatExtraContent{Google: &GoogleExtraContent{ThoughtSignature: "sig"}},
		}},
		Extra: map[string]json.RawMessage{"role": json.RawMessage(`"user"`), "x": json.RawMessage(`1`)},
	}
	got, err := json.Marshal(message)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"},"extra_content":{"google":{"thought_signature":"sig"}}}],"x":1}`
	if string(got) != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
}
