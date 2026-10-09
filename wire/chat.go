package wire

import "encoding/json"

// ChatRequest is a Chat Completions request body.
type ChatRequest struct {
	Model               string
	Messages            []ChatMessage
	Tools               []ChatTool
	ToolChoice          *ChatToolChoice
	Stream              *bool
	StreamOptions       json.RawMessage
	MaxTokens           json.Number
	MaxCompletionTokens json.Number
	Temperature         json.Number
	TopP                json.Number
	Stop                *ChatStop
	N                   json.Number
	Seed                json.Number
	ReasoningEffort     string
	ParallelToolCalls   *bool
	ResponseFormat      json.RawMessage
	Metadata            json.RawMessage
	User                string
	Extra               map[string]json.RawMessage
}

// DecodeChatRequest decodes a Chat Completions request body.
func DecodeChatRequest(data []byte) (ChatRequest, error) {
	var r ChatRequest
	err := json.Unmarshal(data, &r)
	return r, err
}

func (r *ChatRequest) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*r = ChatRequest{}
	m.name("model", &r.Model)
	take(m, "messages", &r.Messages)
	take(m, "tools", &r.Tools)
	take(m, "tool_choice", &r.ToolChoice)
	m.flag("stream", &r.Stream)
	m.raw("stream_options", &r.StreamOptions)
	m.number("max_tokens", &r.MaxTokens)
	m.number("max_completion_tokens", &r.MaxCompletionTokens)
	m.number("temperature", &r.Temperature)
	m.number("top_p", &r.TopP)
	take(m, "stop", &r.Stop)
	m.number("n", &r.N)
	m.number("seed", &r.Seed)
	m.name("reasoning_effort", &r.ReasoningEffort)
	m.flag("parallel_tool_calls", &r.ParallelToolCalls)
	m.raw("response_format", &r.ResponseFormat)
	m.raw("metadata", &r.Metadata)
	m.name("user", &r.User)
	r.Extra = m.extra()
	return nil
}

func (r ChatRequest) MarshalJSON() ([]byte, error) {
	var o object
	o.name("model", r.Model)
	putSlice(&o, "messages", r.Messages)
	putSlice(&o, "tools", r.Tools)
	put(&o, "tool_choice", r.ToolChoice)
	o.flag("stream", r.Stream)
	o.raw("stream_options", r.StreamOptions)
	o.number("max_tokens", r.MaxTokens)
	o.number("max_completion_tokens", r.MaxCompletionTokens)
	o.number("temperature", r.Temperature)
	o.number("top_p", r.TopP)
	put(&o, "stop", r.Stop)
	o.number("n", r.N)
	o.number("seed", r.Seed)
	o.name("reasoning_effort", r.ReasoningEffort)
	o.flag("parallel_tool_calls", r.ParallelToolCalls)
	o.raw("response_format", r.ResponseFormat)
	o.raw("metadata", r.Metadata)
	o.name("user", r.User)
	return o.finish(r.Extra)
}

// ChatMessage is one Chat message. It also serves as the delta of a stream
// chunk, which is a partial message.
type ChatMessage struct {
	Role       string
	Content    *ChatContent
	Name       string
	ToolCalls  []ChatToolCall
	ToolCallID string
	Refusal    *string
	// ReasoningDetails is the vendor reasoning_details array that some
	// providers return on assistant messages and expect back verbatim on
	// the next turn. It is kept as raw JSON.
	ReasoningDetails json.RawMessage
	Extra            map[string]json.RawMessage
}

func (c *ChatMessage) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*c = ChatMessage{}
	m.name("role", &c.Role)
	take(m, "content", &c.Content)
	m.name("name", &c.Name)
	take(m, "tool_calls", &c.ToolCalls)
	m.name("tool_call_id", &c.ToolCallID)
	m.text("refusal", &c.Refusal)
	m.raw("reasoning_details", &c.ReasoningDetails)
	c.Extra = m.extra()
	return nil
}

func (c ChatMessage) MarshalJSON() ([]byte, error) {
	var o object
	o.name("role", c.Role)
	put(&o, "content", c.Content)
	o.name("name", c.Name)
	putSlice(&o, "tool_calls", c.ToolCalls)
	o.name("tool_call_id", c.ToolCallID)
	o.text("refusal", c.Refusal)
	o.raw("reasoning_details", c.ReasoningDetails)
	return o.finish(c.Extra)
}

// ChatContent is message content: a string when Text is set, and otherwise
// the Parts array.
type ChatContent struct {
	Text  *string
	Parts []ChatContentPart
}

func (c *ChatContent) UnmarshalJSON(data []byte) error {
	return decodeTextOr(data, &c.Text, &c.Parts)
}

func (c ChatContent) MarshalJSON() ([]byte, error) {
	return encodeTextOr(c.Text, c.Parts)
}
