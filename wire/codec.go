// Package wire holds typed request, response and stream-chunk structs for the
// OpenAI Chat Completions, OpenAI Responses and Anthropic Messages surfaces.
//
// The structs are codecs, not validators. Decoding followed by encoding
// reproduces the input as JSON: the output equals the input once both are
// parsed, with numbers compared by their literal text. To make that hold:
//
//   - Every modeled object keeps the members it does not model in Extra, and
//     emits them again on encode.
//   - A member whose value does not have the modeled shape, such as null or a
//     number where a string is expected, is kept verbatim in Extra instead of
//     failing the decode. A modeled field that is set takes precedence over an
//     Extra entry with the same name.
//   - Polymorphic members keep their form: content as a string or as parts,
//     tool_choice as a string or as an object, stop as a string or an array.
//   - Numbers are json.Number, so their literal text survives.
//
// Optional identifiers are plain strings and are omitted when empty. Free
// text that may legitimately be empty, such as streamed arguments, is a
// *string, and booleans are *bool, so an explicit "" or false survives. A nil
// slice is omitted, while a non-nil empty slice encodes as [].
//
// Vendor fields that products depend on are modeled explicitly: the Gemini
// thought signature on Chat tool calls (ChatToolCall.ExtraContent, and
// ChatToolCall.ThoughtSignature for the other locations providers use), the
// reasoning_details array on Chat messages (ChatMessage.ReasoningDetails),
// and cache_control on Chat content parts and Anthropic content blocks,
// system blocks and tools (CacheControl).
package wire

import (
	"bytes"
	"encoding/json"
	"errors"
	"sort"
)

var errNotObject = errors.New("wire: value is not a JSON object")

// members is the member set of one JSON object while it is decoded. Decoders
// remove the members they model; whatever remains becomes Extra.
type members map[string]json.RawMessage

func decodeMembers(data []byte) (members, error) {
	var m members
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	if m == nil {
		return nil, errNotObject
	}
	return m, nil
}

func isNull(raw json.RawMessage) bool {
	return len(raw) == 4 && string(raw) == "null"
}

// name takes a non-empty string. An empty string is kept in Extra because
// the field cannot tell it apart from an absent member.
func (m members) name(key string, dst *string) {
	var s string
	if raw, ok := m[key]; ok && len(raw) > 0 && raw[0] == '"' && json.Unmarshal(raw, &s) == nil && s != "" {
		*dst = s
		delete(m, key)
	}
}

// text takes any string, including an empty one.
func (m members) text(key string, dst **string) {
	var s string
	if raw, ok := m[key]; ok && len(raw) > 0 && raw[0] == '"' && json.Unmarshal(raw, &s) == nil {
		*dst = &s
		delete(m, key)
	}
}

func (m members) flag(key string, dst **bool) {
	if raw, ok := m[key]; ok {
		switch string(raw) {
		case "true", "false":
			b := string(raw) == "true"
			*dst = &b
			delete(m, key)
		}
	}
}

// number keeps the literal text of a JSON number.
func (m members) number(key string, dst *json.Number) {
	if raw, ok := m[key]; ok && len(raw) > 0 && (raw[0] == '-' || (raw[0] >= '0' && raw[0] <= '9')) {
		*dst = json.Number(raw)
		delete(m, key)
	}
}

// raw takes any value verbatim, null included.
func (m members) raw(key string, dst *json.RawMessage) {
	if raw, ok := m[key]; ok {
		*dst = raw
		delete(m, key)
	}
}

// extra returns the members nobody modeled, or nil.
func (m members) extra() map[string]json.RawMessage {
	if len(m) == 0 {
		return nil
	}
	return m
}

// take decodes a modeled member into dst. Null and values of another shape
// stay in the member set, so they survive in Extra.
func take[T any](m members, key string, dst *T) {
	raw, ok := m[key]
	if !ok || isNull(raw) {
		return
	}
	var value T
	if json.Unmarshal(raw, &value) != nil {
		return
	}
	*dst = value
	delete(m, key)
}

// object writes one JSON object: modeled members in declaration order, then
// the Extra members that no modeled member already wrote, sorted by name.
type object struct {
	buf     bytes.Buffer
	written map[string]bool
	err     error
}

func (o *object) member(key string, value any) {
	if o.err != nil {
		return
	}
	raw, err := json.Marshal(value)
	if err != nil {
		o.err = err
		return
	}
	o.rawMember(key, raw)
}

func (o *object) rawMember(key string, raw []byte) {
	if o.err != nil {
		return
	}
	name, err := json.Marshal(key)
	if err != nil {
		o.err = err
		return
	}
	if o.written == nil {
		o.written = map[string]bool{}
		o.buf.WriteByte('{')
	} else {
		o.buf.WriteByte(',')
	}
	o.written[key] = true
	o.buf.Write(name)
	o.buf.WriteByte(':')
	o.buf.Write(raw)
}

func (o *object) name(key, value string) {
	if value != "" {
		o.member(key, value)
	}
}

func (o *object) text(key string, value *string) {
	if value != nil {
		o.member(key, *value)
	}
}

func (o *object) flag(key string, value *bool) {
	if value != nil {
		o.member(key, *value)
	}
}

func (o *object) number(key string, value json.Number) {
	if value != "" {
		o.member(key, value)
	}
}

func (o *object) raw(key string, value json.RawMessage) {
	if value != nil {
		o.member(key, value)
	}
}

// put writes a modeled pointer, slice or union member when it is set.
func put[T any](o *object, key string, value *T) {
	if value != nil {
		o.member(key, value)
	}
}

func putSlice[T any](o *object, key string, value []T) {
	if value != nil {
		o.member(key, value)
	}
}

func (o *object) finish(extra map[string]json.RawMessage) ([]byte, error) {
	keys := make([]string, 0, len(extra))
	for key := range extra {
		if !o.written[key] {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	for _, key := range keys {
		o.member(key, extra[key])
	}
	if o.err != nil {
		return nil, o.err
	}
	if o.written == nil {
		return []byte("{}"), nil
	}
	o.buf.WriteByte('}')
	return o.buf.Bytes(), nil
}

// decodeTextOr decodes a member that is either a string or an array.
func decodeTextOr[T any](data []byte, text **string, list *[]T) error {
	switch {
	case len(data) > 0 && data[0] == '"':
		var s string
		if err := json.Unmarshal(data, &s); err != nil {
			return err
		}
		*text, *list = &s, nil
		return nil
	case len(data) > 0 && data[0] == '[':
		items := []T{}
		if err := json.Unmarshal(data, &items); err != nil {
			return err
		}
		*text, *list = nil, items
		return nil
	}
	return errors.New("wire: value is neither a string nor an array")
}

// encodeTextOr encodes the string form when text is set and the array form,
// possibly empty, otherwise.
func encodeTextOr[T any](text *string, list []T) ([]byte, error) {
	if text != nil {
		return json.Marshal(*text)
	}
	if list == nil {
		list = []T{}
	}
	return json.Marshal(list)
}
