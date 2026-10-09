package wire

import (
	"encoding/json"
	"testing"
)

func TestDecodeRejectsNonObjects(t *testing.T) {
	for _, input := range []string{``, `nul`, `null`, `[]`, `"text"`, `{"model":`} {
		if _, err := DecodeChatRequest([]byte(input)); err == nil {
			t.Errorf("DecodeChatRequest(%q) succeeded", input)
		}
	}
}

func TestEncodeExtraEdgeCases(t *testing.T) {
	// Keys that need escaping are escaped; typed members win over Extra.
	part := ChatContentPart{Type: "text", Text: str("hi"), Extra: map[string]json.RawMessage{
		"type": json.RawMessage(`"ignored"`), "quote\"key": json.RawMessage(`{"a":[1, 2]}`),
	}}
	got, err := json.Marshal(part)
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"type":"text","text":"hi","quote\"key":{"a":[1,2]}}`; string(got) != want {
		t.Fatalf("got %s, want %s", got, want)
	}
	// Invalid raw JSON in Extra is an encode error, not silent corruption.
	part.Extra = map[string]json.RawMessage{"bad": json.RawMessage(`{`)}
	if _, err := json.Marshal(part); err == nil {
		t.Fatal("invalid Extra encoded")
	}
	// An empty object stays an object.
	if got, err := json.Marshal(CacheControl{}); err != nil || string(got) != `{}` {
		t.Fatalf("empty object: %s %v", got, err)
	}
}

func TestExactNumbersSurvive(t *testing.T) {
	data := []byte(`{"temperature":1.10,"max_tokens":1E2,"top_p":-0,"seed":100000000000000000001,"metadata":{"n":0.1000}}`)
	r := assertRoundTrip(t, data, DecodeChatRequest)
	if r.Temperature != "1.10" || r.MaxTokens != "1E2" || r.TopP != "-0" || r.Seed != "100000000000000000001" {
		t.Fatalf("numbers: %+v", r)
	}
}
