package wire

import "testing"

func TestChatToolCallThoughtSignatureOrder(t *testing.T) {
	cases := []struct {
		name, call, want string
	}{
		{"google first", `{"extra_content":{"google":{"thought_signature":"g"}},"function":{"thought_signature":"f"},"thought_signature":"c"}`, "g"},
		{"function next", `{"extra_content":{"google":{"thought_signature":""}},"function":{"name":"x","thought_signature":"f"},"thought_signature":"c"}`, "f"},
		{"call level last", `{"function":{"thought_signature":7},"thought_signature":"c"}`, "c"},
		{"none", `{"function":{"name":"x"},"thought_signature":null}`, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			decode := func(data []byte) (ChatToolCall, error) {
				var call ChatToolCall
				err := call.UnmarshalJSON(data)
				return call, err
			}
			call := assertRoundTrip(t, []byte(tc.call), decode)
			if got := call.ThoughtSignature(); got != tc.want {
				t.Fatalf("ThoughtSignature() = %q, want %q", got, tc.want)
			}
		})
	}
}
