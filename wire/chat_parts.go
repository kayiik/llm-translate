package wire

import "encoding/json"

// ChatContentPart is one element of array-form Chat content.
type ChatContentPart struct {
	Type       string
	Text       *string
	ImageURL   *ChatImageURL
	InputAudio json.RawMessage
	File       json.RawMessage
	// CacheControl is the Anthropic prompt-caching breakpoint that some
	// Chat proxies accept on a content part.
	CacheControl *CacheControl
	Extra        map[string]json.RawMessage
}

func (p *ChatContentPart) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*p = ChatContentPart{}
	m.name("type", &p.Type)
	m.text("text", &p.Text)
	take(m, "image_url", &p.ImageURL)
	m.raw("input_audio", &p.InputAudio)
	m.raw("file", &p.File)
	take(m, "cache_control", &p.CacheControl)
	p.Extra = m.extra()
	return nil
}

func (p ChatContentPart) MarshalJSON() ([]byte, error) {
	var o object
	o.name("type", p.Type)
	o.text("text", p.Text)
	put(&o, "image_url", p.ImageURL)
	o.raw("input_audio", p.InputAudio)
	o.raw("file", p.File)
	put(&o, "cache_control", p.CacheControl)
	return o.finish(p.Extra)
}

// ChatImageURL is the image_url object of an image content part.
type ChatImageURL struct {
	URL    string
	Detail string
	Extra  map[string]json.RawMessage
}

func (u *ChatImageURL) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*u = ChatImageURL{}
	m.name("url", &u.URL)
	m.name("detail", &u.Detail)
	u.Extra = m.extra()
	return nil
}

func (u ChatImageURL) MarshalJSON() ([]byte, error) {
	var o object
	o.name("url", u.URL)
	o.name("detail", u.Detail)
	return o.finish(u.Extra)
}

// ChatToolCall is one tool call of an assistant message or stream delta.
type ChatToolCall struct {
	// Index identifies the call across the deltas of a stream.
	Index    json.Number
	ID       string
	Type     string
	Function *ChatFunctionCall
	// ExtraContent carries vendor data. Gemini puts the thought signature
	// that must accompany the call on the next turn at
	// extra_content.google.thought_signature.
	ExtraContent *ChatExtraContent
	Extra        map[string]json.RawMessage
}

func (c *ChatToolCall) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*c = ChatToolCall{}
	m.number("index", &c.Index)
	m.name("id", &c.ID)
	m.name("type", &c.Type)
	take(m, "function", &c.Function)
	take(m, "extra_content", &c.ExtraContent)
	c.Extra = m.extra()
	return nil
}

func (c ChatToolCall) MarshalJSON() ([]byte, error) {
	var o object
	o.number("index", c.Index)
	o.name("id", c.ID)
	o.name("type", c.Type)
	put(&o, "function", c.Function)
	put(&o, "extra_content", c.ExtraContent)
	return o.finish(c.Extra)
}

// ThoughtSignature returns the Gemini thought signature of the call, or "".
// Providers store it in different places; the first non-empty string wins,
// in this order: extra_content.google.thought_signature (Google's documented
// location), function.thought_signature, then a call-level
// thought_signature. The last two are not modeled and stay in Extra.
func (c ChatToolCall) ThoughtSignature() string {
	if c.ExtraContent != nil && c.ExtraContent.Google != nil && c.ExtraContent.Google.ThoughtSignature != "" {
		return c.ExtraContent.Google.ThoughtSignature
	}
	if c.Function != nil {
		if s := extraString(c.Function.Extra, "thought_signature"); s != "" {
			return s
		}
	}
	return extraString(c.Extra, "thought_signature")
}

// extraString returns the string value of an Extra member, or "".
func extraString(extra map[string]json.RawMessage, key string) string {
	var s string
	if raw, ok := extra[key]; ok && json.Unmarshal(raw, &s) == nil {
		return s
	}
	return ""
}

// ChatFunctionCall is the function of a tool call. Arguments is a pointer
// because stream deltas often carry an explicit empty string.
type ChatFunctionCall struct {
	Name      string
	Arguments *string
	Extra     map[string]json.RawMessage
}

func (f *ChatFunctionCall) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*f = ChatFunctionCall{}
	m.name("name", &f.Name)
	m.text("arguments", &f.Arguments)
	f.Extra = m.extra()
	return nil
}

func (f ChatFunctionCall) MarshalJSON() ([]byte, error) {
	var o object
	o.name("name", f.Name)
	o.text("arguments", f.Arguments)
	return o.finish(f.Extra)
}
