package wire

import "encoding/json"

// AnthropicBlock is one content block of any type: text, image, document,
// tool_use, tool_result, thinking, redacted_thinking and so on. Fields that
// a block type does not use stay unset.
type AnthropicBlock struct {
	Type string
	// Text is set on text blocks.
	Text      *string
	Citations json.RawMessage
	// Source is the image or document source.
	Source json.RawMessage
	// ID, Name and Input are set on tool_use and server_tool_use blocks.
	ID    string
	Name  string
	Input json.RawMessage
	// ToolUseID, Content and IsError are set on tool_result blocks.
	ToolUseID string
	Content   *AnthropicContent
	IsError   *bool
	// Thinking and Signature are set on thinking blocks; Data on
	// redacted_thinking blocks.
	Thinking  *string
	Signature *string
	Data      *string
	// CacheControl marks a prompt-caching breakpoint after this block.
	CacheControl *CacheControl
	Extra        map[string]json.RawMessage
}

func (b *AnthropicBlock) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*b = AnthropicBlock{}
	m.name("type", &b.Type)
	m.text("text", &b.Text)
	m.raw("citations", &b.Citations)
	m.raw("source", &b.Source)
	m.name("id", &b.ID)
	m.name("name", &b.Name)
	m.raw("input", &b.Input)
	m.name("tool_use_id", &b.ToolUseID)
	take(m, "content", &b.Content)
	m.flag("is_error", &b.IsError)
	m.text("thinking", &b.Thinking)
	m.text("signature", &b.Signature)
	m.text("data", &b.Data)
	take(m, "cache_control", &b.CacheControl)
	b.Extra = m.extra()
	return nil
}

func (b AnthropicBlock) MarshalJSON() ([]byte, error) {
	var o object
	o.name("type", b.Type)
	o.text("text", b.Text)
	o.raw("citations", b.Citations)
	o.raw("source", b.Source)
	o.name("id", b.ID)
	o.name("name", b.Name)
	o.raw("input", b.Input)
	o.name("tool_use_id", b.ToolUseID)
	put(&o, "content", b.Content)
	o.flag("is_error", b.IsError)
	o.text("thinking", b.Thinking)
	o.text("signature", b.Signature)
	o.text("data", b.Data)
	put(&o, "cache_control", b.CacheControl)
	return o.finish(b.Extra)
}

// AnthropicResponse is a Messages response object. It is also the message
// of a message_start stream event.
type AnthropicResponse struct {
	ID           string
	Type         string
	Role         string
	Model        string
	Content      []AnthropicBlock
	StopReason   string
	StopSequence *string
	Usage        *AnthropicUsage
	Extra        map[string]json.RawMessage
}

// DecodeAnthropicResponse decodes a Messages response object.
func DecodeAnthropicResponse(data []byte) (AnthropicResponse, error) {
	var r AnthropicResponse
	err := json.Unmarshal(data, &r)
	return r, err
}

func (r *AnthropicResponse) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*r = AnthropicResponse{}
	m.name("id", &r.ID)
	m.name("type", &r.Type)
	m.name("role", &r.Role)
	m.name("model", &r.Model)
	take(m, "content", &r.Content)
	m.name("stop_reason", &r.StopReason)
	m.text("stop_sequence", &r.StopSequence)
	take(m, "usage", &r.Usage)
	r.Extra = m.extra()
	return nil
}

func (r AnthropicResponse) MarshalJSON() ([]byte, error) {
	var o object
	o.name("id", r.ID)
	o.name("type", r.Type)
	o.name("role", r.Role)
	o.name("model", r.Model)
	putSlice(&o, "content", r.Content)
	o.name("stop_reason", r.StopReason)
	o.text("stop_sequence", r.StopSequence)
	put(&o, "usage", r.Usage)
	return o.finish(r.Extra)
}

// AnthropicUsage is the usage object of a response or message_delta event.
type AnthropicUsage struct {
	InputTokens              json.Number
	OutputTokens             json.Number
	CacheCreationInputTokens json.Number
	CacheReadInputTokens     json.Number
	CacheCreation            json.RawMessage
	ServerToolUse            json.RawMessage
	ServiceTier              string
	Extra                    map[string]json.RawMessage
}

func (u *AnthropicUsage) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*u = AnthropicUsage{}
	m.number("input_tokens", &u.InputTokens)
	m.number("output_tokens", &u.OutputTokens)
	m.number("cache_creation_input_tokens", &u.CacheCreationInputTokens)
	m.number("cache_read_input_tokens", &u.CacheReadInputTokens)
	m.raw("cache_creation", &u.CacheCreation)
	m.raw("server_tool_use", &u.ServerToolUse)
	m.name("service_tier", &u.ServiceTier)
	u.Extra = m.extra()
	return nil
}

func (u AnthropicUsage) MarshalJSON() ([]byte, error) {
	var o object
	o.number("input_tokens", u.InputTokens)
	o.number("output_tokens", u.OutputTokens)
	o.number("cache_creation_input_tokens", u.CacheCreationInputTokens)
	o.number("cache_read_input_tokens", u.CacheReadInputTokens)
	o.raw("cache_creation", u.CacheCreation)
	o.raw("server_tool_use", u.ServerToolUse)
	o.name("service_tier", u.ServiceTier)
	return o.finish(u.Extra)
}
