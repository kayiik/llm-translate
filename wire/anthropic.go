package wire

import "encoding/json"

// AnthropicRequest is an Anthropic Messages request body.
type AnthropicRequest struct {
	Model string
	// System is a string or an array of text blocks, which may carry
	// cache_control.
	System        *AnthropicContent
	Messages      []AnthropicMessage
	MaxTokens     json.Number
	Tools         []AnthropicTool
	ToolChoice    *AnthropicToolChoice
	Stream        *bool
	Temperature   json.Number
	TopP          json.Number
	TopK          json.Number
	StopSequences []string
	Metadata      json.RawMessage
	Thinking      json.RawMessage
	Extra         map[string]json.RawMessage
}

// DecodeAnthropicRequest decodes an Anthropic Messages request body.
func DecodeAnthropicRequest(data []byte) (AnthropicRequest, error) {
	var r AnthropicRequest
	err := json.Unmarshal(data, &r)
	return r, err
}

func (r *AnthropicRequest) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*r = AnthropicRequest{}
	m.name("model", &r.Model)
	take(m, "system", &r.System)
	take(m, "messages", &r.Messages)
	m.number("max_tokens", &r.MaxTokens)
	take(m, "tools", &r.Tools)
	take(m, "tool_choice", &r.ToolChoice)
	m.flag("stream", &r.Stream)
	m.number("temperature", &r.Temperature)
	m.number("top_p", &r.TopP)
	m.number("top_k", &r.TopK)
	take(m, "stop_sequences", &r.StopSequences)
	m.raw("metadata", &r.Metadata)
	m.raw("thinking", &r.Thinking)
	r.Extra = m.extra()
	return nil
}

func (r AnthropicRequest) MarshalJSON() ([]byte, error) {
	var o object
	o.name("model", r.Model)
	put(&o, "system", r.System)
	putSlice(&o, "messages", r.Messages)
	o.number("max_tokens", r.MaxTokens)
	putSlice(&o, "tools", r.Tools)
	put(&o, "tool_choice", r.ToolChoice)
	o.flag("stream", r.Stream)
	o.number("temperature", r.Temperature)
	o.number("top_p", r.TopP)
	o.number("top_k", r.TopK)
	putSlice(&o, "stop_sequences", r.StopSequences)
	o.raw("metadata", r.Metadata)
	o.raw("thinking", r.Thinking)
	return o.finish(r.Extra)
}

// AnthropicMessage is one request message.
type AnthropicMessage struct {
	Role    string
	Content *AnthropicContent
	Extra   map[string]json.RawMessage
}

func (a *AnthropicMessage) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*a = AnthropicMessage{}
	m.name("role", &a.Role)
	take(m, "content", &a.Content)
	a.Extra = m.extra()
	return nil
}

func (a AnthropicMessage) MarshalJSON() ([]byte, error) {
	var o object
	o.name("role", a.Role)
	put(&o, "content", a.Content)
	return o.finish(a.Extra)
}

// AnthropicContent is message, system or tool_result content: a string when
// Text is set, and otherwise the Blocks array.
type AnthropicContent struct {
	Text   *string
	Blocks []AnthropicBlock
}

func (c *AnthropicContent) UnmarshalJSON(data []byte) error {
	return decodeTextOr(data, &c.Text, &c.Blocks)
}

func (c AnthropicContent) MarshalJSON() ([]byte, error) {
	return encodeTextOr(c.Text, c.Blocks)
}

// AnthropicTool is one tool definition, client or server.
type AnthropicTool struct {
	Type         string
	Name         string
	Description  *string
	InputSchema  json.RawMessage
	CacheControl *CacheControl
	Extra        map[string]json.RawMessage
}

func (t *AnthropicTool) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*t = AnthropicTool{}
	m.name("type", &t.Type)
	m.name("name", &t.Name)
	m.text("description", &t.Description)
	m.raw("input_schema", &t.InputSchema)
	take(m, "cache_control", &t.CacheControl)
	t.Extra = m.extra()
	return nil
}

func (t AnthropicTool) MarshalJSON() ([]byte, error) {
	var o object
	o.name("type", t.Type)
	o.name("name", t.Name)
	o.text("description", t.Description)
	o.raw("input_schema", t.InputSchema)
	put(&o, "cache_control", t.CacheControl)
	return o.finish(t.Extra)
}

// AnthropicToolChoice is tool_choice, which Anthropic always sends as an
// object such as {"type":"tool","name":"lookup"}.
type AnthropicToolChoice struct {
	Type                   string
	Name                   string
	DisableParallelToolUse *bool
	Extra                  map[string]json.RawMessage
}

func (c *AnthropicToolChoice) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*c = AnthropicToolChoice{}
	m.name("type", &c.Type)
	m.name("name", &c.Name)
	m.flag("disable_parallel_tool_use", &c.DisableParallelToolUse)
	c.Extra = m.extra()
	return nil
}

func (c AnthropicToolChoice) MarshalJSON() ([]byte, error) {
	var o object
	o.name("type", c.Type)
	o.name("name", c.Name)
	o.flag("disable_parallel_tool_use", c.DisableParallelToolUse)
	return o.finish(c.Extra)
}
