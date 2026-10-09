package wire

import "encoding/json"

// ResponsesResponse is a Responses response object. It is also the
// response of response.created, response.completed and similar events.
type ResponsesResponse struct {
	ID                 string
	Object             string
	CreatedAt          json.Number
	Status             string
	Model              string
	Output             []ResponsesItem
	OutputText         *string
	Instructions       *string
	PreviousResponseID string
	Reasoning          *ResponsesReasoning
	Usage              *ResponsesUsage
	Error              json.RawMessage
	IncompleteDetails  json.RawMessage
	Extra              map[string]json.RawMessage
}

// DecodeResponsesResponse decodes a Responses response object.
func DecodeResponsesResponse(data []byte) (ResponsesResponse, error) {
	var r ResponsesResponse
	err := json.Unmarshal(data, &r)
	return r, err
}

func (r *ResponsesResponse) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*r = ResponsesResponse{}
	m.name("id", &r.ID)
	m.name("object", &r.Object)
	m.number("created_at", &r.CreatedAt)
	m.name("status", &r.Status)
	m.name("model", &r.Model)
	take(m, "output", &r.Output)
	m.text("output_text", &r.OutputText)
	m.text("instructions", &r.Instructions)
	m.name("previous_response_id", &r.PreviousResponseID)
	take(m, "reasoning", &r.Reasoning)
	take(m, "usage", &r.Usage)
	m.raw("error", &r.Error)
	m.raw("incomplete_details", &r.IncompleteDetails)
	r.Extra = m.extra()
	return nil
}

func (r ResponsesResponse) MarshalJSON() ([]byte, error) {
	var o object
	o.name("id", r.ID)
	o.name("object", r.Object)
	o.number("created_at", r.CreatedAt)
	o.name("status", r.Status)
	o.name("model", r.Model)
	putSlice(&o, "output", r.Output)
	o.text("output_text", r.OutputText)
	o.text("instructions", r.Instructions)
	o.name("previous_response_id", r.PreviousResponseID)
	put(&o, "reasoning", r.Reasoning)
	put(&o, "usage", r.Usage)
	o.raw("error", r.Error)
	o.raw("incomplete_details", r.IncompleteDetails)
	return o.finish(r.Extra)
}

// ResponsesUsage is the usage object of a response.
type ResponsesUsage struct {
	InputTokens         json.Number
	OutputTokens        json.Number
	TotalTokens         json.Number
	InputTokensDetails  json.RawMessage
	OutputTokensDetails json.RawMessage
	Extra               map[string]json.RawMessage
}

func (u *ResponsesUsage) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*u = ResponsesUsage{}
	m.number("input_tokens", &u.InputTokens)
	m.number("output_tokens", &u.OutputTokens)
	m.number("total_tokens", &u.TotalTokens)
	m.raw("input_tokens_details", &u.InputTokensDetails)
	m.raw("output_tokens_details", &u.OutputTokensDetails)
	u.Extra = m.extra()
	return nil
}

func (u ResponsesUsage) MarshalJSON() ([]byte, error) {
	var o object
	o.number("input_tokens", u.InputTokens)
	o.number("output_tokens", u.OutputTokens)
	o.number("total_tokens", u.TotalTokens)
	o.raw("input_tokens_details", u.InputTokensDetails)
	o.raw("output_tokens_details", u.OutputTokensDetails)
	return o.finish(u.Extra)
}

// ResponsesEvent is the data of one Responses stream event, without the SSE
// framing. Type repeats the SSE event name.
type ResponsesEvent struct {
	Type           string
	SequenceNumber json.Number
	// Response is set on response.created, response.completed and the
	// other whole-response events.
	Response     *ResponsesResponse
	OutputIndex  json.Number
	ContentIndex json.Number
	SummaryIndex json.Number
	ItemID       string
	Item         *ResponsesItem
	Part         *ResponsesContentPart
	// Delta is the text or arguments fragment of a *.delta event; Text and
	// Arguments carry the complete value of the matching *.done event.
	Delta     *string
	Text      *string
	Arguments *string
	Extra     map[string]json.RawMessage
}

// DecodeResponsesEvent decodes the data of one Responses stream event.
func DecodeResponsesEvent(data []byte) (ResponsesEvent, error) {
	var e ResponsesEvent
	err := json.Unmarshal(data, &e)
	return e, err
}

func (e *ResponsesEvent) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*e = ResponsesEvent{}
	m.name("type", &e.Type)
	m.number("sequence_number", &e.SequenceNumber)
	take(m, "response", &e.Response)
	m.number("output_index", &e.OutputIndex)
	m.number("content_index", &e.ContentIndex)
	m.number("summary_index", &e.SummaryIndex)
	m.name("item_id", &e.ItemID)
	take(m, "item", &e.Item)
	take(m, "part", &e.Part)
	m.text("delta", &e.Delta)
	m.text("text", &e.Text)
	m.text("arguments", &e.Arguments)
	e.Extra = m.extra()
	return nil
}

func (e ResponsesEvent) MarshalJSON() ([]byte, error) {
	var o object
	o.name("type", e.Type)
	o.number("sequence_number", e.SequenceNumber)
	put(&o, "response", e.Response)
	o.number("output_index", e.OutputIndex)
	o.number("content_index", e.ContentIndex)
	o.number("summary_index", e.SummaryIndex)
	o.name("item_id", e.ItemID)
	put(&o, "item", e.Item)
	put(&o, "part", e.Part)
	o.text("delta", e.Delta)
	o.text("text", e.Text)
	o.text("arguments", e.Arguments)
	return o.finish(e.Extra)
}
