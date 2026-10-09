package wire

import "encoding/json"

// AnthropicEvent is the data of one Messages stream event, without the SSE
// "event:" and "data:" framing. Type repeats the SSE event name.
type AnthropicEvent struct {
	Type string
	// Message is set on message_start.
	Message *AnthropicResponse
	// Index identifies the content block of content_block_* events.
	Index        json.Number
	ContentBlock *AnthropicBlock
	// Delta is set on content_block_delta and message_delta.
	Delta *AnthropicDelta
	// Usage is set on message_delta.
	Usage *AnthropicUsage
	Error json.RawMessage
	Extra map[string]json.RawMessage
}

// DecodeAnthropicEvent decodes the data of one Messages stream event.
func DecodeAnthropicEvent(data []byte) (AnthropicEvent, error) {
	var e AnthropicEvent
	err := json.Unmarshal(data, &e)
	return e, err
}

func (e *AnthropicEvent) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*e = AnthropicEvent{}
	m.name("type", &e.Type)
	take(m, "message", &e.Message)
	m.number("index", &e.Index)
	take(m, "content_block", &e.ContentBlock)
	take(m, "delta", &e.Delta)
	take(m, "usage", &e.Usage)
	m.raw("error", &e.Error)
	e.Extra = m.extra()
	return nil
}

func (e AnthropicEvent) MarshalJSON() ([]byte, error) {
	var o object
	o.name("type", e.Type)
	put(&o, "message", e.Message)
	o.number("index", e.Index)
	put(&o, "content_block", e.ContentBlock)
	put(&o, "delta", e.Delta)
	put(&o, "usage", e.Usage)
	o.raw("error", e.Error)
	return o.finish(e.Extra)
}

// AnthropicDelta is the delta of a content_block_delta event (text_delta,
// input_json_delta, thinking_delta, signature_delta, citations_delta) or of
// a message_delta event (stop_reason, stop_sequence).
type AnthropicDelta struct {
	Type         string
	Text         *string
	PartialJSON  *string
	Thinking     *string
	Signature    *string
	Citation     json.RawMessage
	StopReason   string
	StopSequence *string
	Extra        map[string]json.RawMessage
}

func (d *AnthropicDelta) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*d = AnthropicDelta{}
	m.name("type", &d.Type)
	m.text("text", &d.Text)
	m.text("partial_json", &d.PartialJSON)
	m.text("thinking", &d.Thinking)
	m.text("signature", &d.Signature)
	m.raw("citation", &d.Citation)
	m.name("stop_reason", &d.StopReason)
	m.text("stop_sequence", &d.StopSequence)
	d.Extra = m.extra()
	return nil
}

func (d AnthropicDelta) MarshalJSON() ([]byte, error) {
	var o object
	o.name("type", d.Type)
	o.text("text", d.Text)
	o.text("partial_json", d.PartialJSON)
	o.text("thinking", d.Thinking)
	o.text("signature", d.Signature)
	o.raw("citation", d.Citation)
	o.name("stop_reason", d.StopReason)
	o.text("stop_sequence", d.StopSequence)
	return o.finish(d.Extra)
}
