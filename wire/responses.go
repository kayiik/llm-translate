package wire

import (
	"encoding/json"
	"errors"
)

// ResponsesRequest is an OpenAI Responses request body.
type ResponsesRequest struct {
	Model              string
	Input              *ResponsesInput
	Instructions       *string
	Stream             *bool
	MaxOutputTokens    json.Number
	Temperature        json.Number
	TopP               json.Number
	Tools              []ResponsesTool
	ToolChoice         *ResponsesToolChoice
	ParallelToolCalls  *bool
	Reasoning          *ResponsesReasoning
	Metadata           json.RawMessage
	Store              *bool
	PreviousResponseID string
	Include            json.RawMessage
	Text               json.RawMessage
	Truncation         string
	User               string
	PromptCacheKey     string
	Extra              map[string]json.RawMessage
}

// DecodeResponsesRequest decodes a Responses request body.
func DecodeResponsesRequest(data []byte) (ResponsesRequest, error) {
	var r ResponsesRequest
	err := json.Unmarshal(data, &r)
	return r, err
}

func (r *ResponsesRequest) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*r = ResponsesRequest{}
	m.name("model", &r.Model)
	take(m, "input", &r.Input)
	m.text("instructions", &r.Instructions)
	m.flag("stream", &r.Stream)
	m.number("max_output_tokens", &r.MaxOutputTokens)
	m.number("temperature", &r.Temperature)
	m.number("top_p", &r.TopP)
	take(m, "tools", &r.Tools)
	take(m, "tool_choice", &r.ToolChoice)
	m.flag("parallel_tool_calls", &r.ParallelToolCalls)
	take(m, "reasoning", &r.Reasoning)
	m.raw("metadata", &r.Metadata)
	m.flag("store", &r.Store)
	m.name("previous_response_id", &r.PreviousResponseID)
	m.raw("include", &r.Include)
	m.raw("text", &r.Text)
	m.name("truncation", &r.Truncation)
	m.name("user", &r.User)
	m.name("prompt_cache_key", &r.PromptCacheKey)
	r.Extra = m.extra()
	return nil
}

func (r ResponsesRequest) MarshalJSON() ([]byte, error) {
	var o object
	o.name("model", r.Model)
	put(&o, "input", r.Input)
	o.text("instructions", r.Instructions)
	o.flag("stream", r.Stream)
	o.number("max_output_tokens", r.MaxOutputTokens)
	o.number("temperature", r.Temperature)
	o.number("top_p", r.TopP)
	putSlice(&o, "tools", r.Tools)
	put(&o, "tool_choice", r.ToolChoice)
	o.flag("parallel_tool_calls", r.ParallelToolCalls)
	put(&o, "reasoning", r.Reasoning)
	o.raw("metadata", r.Metadata)
	o.flag("store", r.Store)
	o.name("previous_response_id", r.PreviousResponseID)
	o.raw("include", r.Include)
	o.raw("text", r.Text)
	o.name("truncation", r.Truncation)
	o.name("user", r.User)
	o.name("prompt_cache_key", r.PromptCacheKey)
	return o.finish(r.Extra)
}

// ResponsesInput is input: a string when Text is set, and otherwise the
// Items array.
type ResponsesInput struct {
	Text  *string
	Items []ResponsesItem
}

func (i *ResponsesInput) UnmarshalJSON(data []byte) error {
	return decodeTextOr(data, &i.Text, &i.Items)
}

func (i ResponsesInput) MarshalJSON() ([]byte, error) {
	return encodeTextOr(i.Text, i.Items)
}

// ResponsesTool is one tool definition. Function tools are flat, with Name
// and Parameters at the top level; built-in tools keep their options in Extra.
type ResponsesTool struct {
	Type        string
	Name        string
	Description *string
	Parameters  json.RawMessage
	Strict      *bool
	Extra       map[string]json.RawMessage
}

func (t *ResponsesTool) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*t = ResponsesTool{}
	m.name("type", &t.Type)
	m.name("name", &t.Name)
	m.text("description", &t.Description)
	m.raw("parameters", &t.Parameters)
	m.flag("strict", &t.Strict)
	t.Extra = m.extra()
	return nil
}

func (t ResponsesTool) MarshalJSON() ([]byte, error) {
	var o object
	o.name("type", t.Type)
	o.name("name", t.Name)
	o.text("description", t.Description)
	o.raw("parameters", t.Parameters)
	o.flag("strict", t.Strict)
	return o.finish(t.Extra)
}

// ResponsesToolChoice is tool_choice: the string Mode, such as "auto", when
// Mode is set, and otherwise an object such as {"type":"function","name":"f"}.
type ResponsesToolChoice struct {
	Mode  string
	Type  string
	Name  string
	Extra map[string]json.RawMessage
}

func (c *ResponsesToolChoice) UnmarshalJSON(data []byte) error {
	*c = ResponsesToolChoice{}
	if len(data) > 0 && data[0] == '"' {
		if err := json.Unmarshal(data, &c.Mode); err != nil {
			return err
		}
		if c.Mode == "" {
			return errors.New("wire: empty tool_choice mode")
		}
		return nil
	}
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	m.name("type", &c.Type)
	m.name("name", &c.Name)
	c.Extra = m.extra()
	return nil
}

func (c ResponsesToolChoice) MarshalJSON() ([]byte, error) {
	if c.Mode != "" {
		return json.Marshal(c.Mode)
	}
	var o object
	o.name("type", c.Type)
	o.name("name", c.Name)
	return o.finish(c.Extra)
}

// ResponsesReasoning is the reasoning configuration of a request or response.
type ResponsesReasoning struct {
	Effort  string
	Summary string
	Extra   map[string]json.RawMessage
}

func (r *ResponsesReasoning) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*r = ResponsesReasoning{}
	m.name("effort", &r.Effort)
	m.name("summary", &r.Summary)
	r.Extra = m.extra()
	return nil
}

func (r ResponsesReasoning) MarshalJSON() ([]byte, error) {
	var o object
	o.name("effort", r.Effort)
	o.name("summary", r.Summary)
	return o.finish(r.Extra)
}
