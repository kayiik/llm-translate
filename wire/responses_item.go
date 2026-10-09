package wire

import "encoding/json"

// ResponsesItem is one input or output item of any type: message,
// function_call, function_call_output, reasoning and so on. Fields that an
// item type does not use stay unset.
type ResponsesItem struct {
	Type   string
	ID     string
	Status string
	// Role and Content are set on messages. Reasoning items also use
	// Content, for reasoning_text parts.
	Role    string
	Content *ResponsesContent
	// CallID, Name and Arguments are set on function calls; CallID and
	// Output on function call outputs.
	CallID    string
	Name      string
	Arguments *string
	Output    *ResponsesContent
	// Summary and EncryptedContent are set on reasoning items.
	Summary          []ResponsesContentPart
	EncryptedContent *string
	Extra            map[string]json.RawMessage
}

func (i *ResponsesItem) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*i = ResponsesItem{}
	m.name("type", &i.Type)
	m.name("id", &i.ID)
	m.name("status", &i.Status)
	m.name("role", &i.Role)
	take(m, "content", &i.Content)
	m.name("call_id", &i.CallID)
	m.name("name", &i.Name)
	m.text("arguments", &i.Arguments)
	take(m, "output", &i.Output)
	take(m, "summary", &i.Summary)
	m.text("encrypted_content", &i.EncryptedContent)
	i.Extra = m.extra()
	return nil
}

func (i ResponsesItem) MarshalJSON() ([]byte, error) {
	var o object
	o.name("type", i.Type)
	o.name("id", i.ID)
	o.name("status", i.Status)
	o.name("role", i.Role)
	put(&o, "content", i.Content)
	o.name("call_id", i.CallID)
	o.name("name", i.Name)
	o.text("arguments", i.Arguments)
	put(&o, "output", i.Output)
	putSlice(&o, "summary", i.Summary)
	o.text("encrypted_content", i.EncryptedContent)
	return o.finish(i.Extra)
}

// ResponsesContent is item content or function output: a string when Text
// is set, and otherwise the Parts array.
type ResponsesContent struct {
	Text  *string
	Parts []ResponsesContentPart
}

func (c *ResponsesContent) UnmarshalJSON(data []byte) error {
	return decodeTextOr(data, &c.Text, &c.Parts)
}

func (c ResponsesContent) MarshalJSON() ([]byte, error) {
	return encodeTextOr(c.Text, c.Parts)
}

// ResponsesContentPart is one content part: input_text, output_text,
// input_image, input_file, refusal, summary_text, reasoning_text and so on.
type ResponsesContentPart struct {
	Type        string
	Text        *string
	ImageURL    string
	FileID      string
	Detail      string
	Annotations json.RawMessage
	Logprobs    json.RawMessage
	Refusal     *string
	Extra       map[string]json.RawMessage
}

func (p *ResponsesContentPart) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*p = ResponsesContentPart{}
	m.name("type", &p.Type)
	m.text("text", &p.Text)
	m.name("image_url", &p.ImageURL)
	m.name("file_id", &p.FileID)
	m.name("detail", &p.Detail)
	m.raw("annotations", &p.Annotations)
	m.raw("logprobs", &p.Logprobs)
	m.text("refusal", &p.Refusal)
	p.Extra = m.extra()
	return nil
}

func (p ResponsesContentPart) MarshalJSON() ([]byte, error) {
	var o object
	o.name("type", p.Type)
	o.text("text", p.Text)
	o.name("image_url", p.ImageURL)
	o.name("file_id", p.FileID)
	o.name("detail", p.Detail)
	o.raw("annotations", p.Annotations)
	o.raw("logprobs", p.Logprobs)
	o.text("refusal", p.Refusal)
	return o.finish(p.Extra)
}
