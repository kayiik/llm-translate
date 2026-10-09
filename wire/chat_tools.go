package wire

import (
	"encoding/json"
	"errors"
)

// ChatExtraContent is the vendor extra_content object of a tool call.
type ChatExtraContent struct {
	Google *GoogleExtraContent
	Extra  map[string]json.RawMessage
}

func (e *ChatExtraContent) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*e = ChatExtraContent{}
	take(m, "google", &e.Google)
	e.Extra = m.extra()
	return nil
}

func (e ChatExtraContent) MarshalJSON() ([]byte, error) {
	var o object
	put(&o, "google", e.Google)
	return o.finish(e.Extra)
}

// GoogleExtraContent is extra_content.google. ThoughtSignature is opaque and
// must be sent back unchanged with its tool call, or Gemini rejects or
// degrades the next turn.
type GoogleExtraContent struct {
	ThoughtSignature string
	Extra            map[string]json.RawMessage
}

func (g *GoogleExtraContent) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*g = GoogleExtraContent{}
	m.name("thought_signature", &g.ThoughtSignature)
	g.Extra = m.extra()
	return nil
}

func (g GoogleExtraContent) MarshalJSON() ([]byte, error) {
	var o object
	o.name("thought_signature", g.ThoughtSignature)
	return o.finish(g.Extra)
}

// ChatTool is one entry of the tools array.
type ChatTool struct {
	Type     string
	Function *ChatFunction
	Extra    map[string]json.RawMessage
}

func (t *ChatTool) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*t = ChatTool{}
	m.name("type", &t.Type)
	take(m, "function", &t.Function)
	t.Extra = m.extra()
	return nil
}

func (t ChatTool) MarshalJSON() ([]byte, error) {
	var o object
	o.name("type", t.Type)
	put(&o, "function", t.Function)
	return o.finish(t.Extra)
}

// ChatFunction is a function definition. A named tool_choice reuses it with
// only Name set.
type ChatFunction struct {
	Name        string
	Description *string
	Parameters  json.RawMessage
	Strict      *bool
	Extra       map[string]json.RawMessage
}

func (f *ChatFunction) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*f = ChatFunction{}
	m.name("name", &f.Name)
	m.text("description", &f.Description)
	m.raw("parameters", &f.Parameters)
	m.flag("strict", &f.Strict)
	f.Extra = m.extra()
	return nil
}

func (f ChatFunction) MarshalJSON() ([]byte, error) {
	var o object
	o.name("name", f.Name)
	o.text("description", f.Description)
	o.raw("parameters", f.Parameters)
	o.flag("strict", f.Strict)
	return o.finish(f.Extra)
}

// ChatToolChoice is tool_choice: the string Mode, such as "auto", when Mode
// is set, and otherwise an object such as
// {"type":"function","function":{"name":"lookup"}}.
type ChatToolChoice struct {
	Mode     string
	Type     string
	Function *ChatFunction
	Extra    map[string]json.RawMessage
}

func (c *ChatToolChoice) UnmarshalJSON(data []byte) error {
	*c = ChatToolChoice{}
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
	take(m, "function", &c.Function)
	c.Extra = m.extra()
	return nil
}

func (c ChatToolChoice) MarshalJSON() ([]byte, error) {
	if c.Mode != "" {
		return json.Marshal(c.Mode)
	}
	var o object
	o.name("type", c.Type)
	put(&o, "function", c.Function)
	return o.finish(c.Extra)
}

// ChatStop is stop: one sequence when Sequence is set, and otherwise the
// Sequences array.
type ChatStop struct {
	Sequence  *string
	Sequences []string
}

func (s *ChatStop) UnmarshalJSON(data []byte) error {
	return decodeTextOr(data, &s.Sequence, &s.Sequences)
}

func (s ChatStop) MarshalJSON() ([]byte, error) {
	return encodeTextOr(s.Sequence, s.Sequences)
}
