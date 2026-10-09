package wire

import "encoding/json"

// ChatResponse is a chat.completion object.
type ChatResponse struct {
	ID                string
	Object            string
	Created           json.Number
	Model             string
	Choices           []ChatChoice
	Usage             *ChatUsage
	SystemFingerprint string
	ServiceTier       string
	Extra             map[string]json.RawMessage
}

// DecodeChatResponse decodes a chat.completion object.
func DecodeChatResponse(data []byte) (ChatResponse, error) {
	var r ChatResponse
	err := json.Unmarshal(data, &r)
	return r, err
}

func (r *ChatResponse) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*r = ChatResponse{}
	m.name("id", &r.ID)
	m.name("object", &r.Object)
	m.number("created", &r.Created)
	m.name("model", &r.Model)
	take(m, "choices", &r.Choices)
	take(m, "usage", &r.Usage)
	m.name("system_fingerprint", &r.SystemFingerprint)
	m.name("service_tier", &r.ServiceTier)
	r.Extra = m.extra()
	return nil
}

func (r ChatResponse) MarshalJSON() ([]byte, error) {
	var o object
	o.name("id", r.ID)
	o.name("object", r.Object)
	o.number("created", r.Created)
	o.name("model", r.Model)
	putSlice(&o, "choices", r.Choices)
	put(&o, "usage", r.Usage)
	o.name("system_fingerprint", r.SystemFingerprint)
	o.name("service_tier", r.ServiceTier)
	return o.finish(r.Extra)
}

// ChatChunk is one chat.completion.chunk of a stream: the data of one SSE
// event, without the "data: " framing. Its choices carry Delta rather than
// Message.
type ChatChunk ChatResponse

// DecodeChatChunk decodes one chat.completion.chunk.
func DecodeChatChunk(data []byte) (ChatChunk, error) {
	var c ChatChunk
	err := json.Unmarshal(data, &c)
	return c, err
}

func (c *ChatChunk) UnmarshalJSON(data []byte) error {
	return (*ChatResponse)(c).UnmarshalJSON(data)
}

func (c ChatChunk) MarshalJSON() ([]byte, error) {
	return ChatResponse(c).MarshalJSON()
}

// ChatChoice is one choice of a response (Message) or chunk (Delta).
type ChatChoice struct {
	Index        json.Number
	Message      *ChatMessage
	Delta        *ChatMessage
	FinishReason string
	Logprobs     json.RawMessage
	Extra        map[string]json.RawMessage
}

func (c *ChatChoice) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*c = ChatChoice{}
	m.number("index", &c.Index)
	take(m, "message", &c.Message)
	take(m, "delta", &c.Delta)
	m.name("finish_reason", &c.FinishReason)
	m.raw("logprobs", &c.Logprobs)
	c.Extra = m.extra()
	return nil
}

func (c ChatChoice) MarshalJSON() ([]byte, error) {
	var o object
	o.number("index", c.Index)
	put(&o, "message", c.Message)
	put(&o, "delta", c.Delta)
	o.name("finish_reason", c.FinishReason)
	o.raw("logprobs", c.Logprobs)
	return o.finish(c.Extra)
}

// ChatUsage is the usage object of a response or final chunk.
type ChatUsage struct {
	PromptTokens            json.Number
	CompletionTokens        json.Number
	TotalTokens             json.Number
	PromptTokensDetails     json.RawMessage
	CompletionTokensDetails json.RawMessage
	Extra                   map[string]json.RawMessage
}

func (u *ChatUsage) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*u = ChatUsage{}
	m.number("prompt_tokens", &u.PromptTokens)
	m.number("completion_tokens", &u.CompletionTokens)
	m.number("total_tokens", &u.TotalTokens)
	m.raw("prompt_tokens_details", &u.PromptTokensDetails)
	m.raw("completion_tokens_details", &u.CompletionTokensDetails)
	u.Extra = m.extra()
	return nil
}

func (u ChatUsage) MarshalJSON() ([]byte, error) {
	var o object
	o.number("prompt_tokens", u.PromptTokens)
	o.number("completion_tokens", u.CompletionTokens)
	o.number("total_tokens", u.TotalTokens)
	o.raw("prompt_tokens_details", u.PromptTokensDetails)
	o.raw("completion_tokens_details", u.CompletionTokensDetails)
	return o.finish(u.Extra)
}
