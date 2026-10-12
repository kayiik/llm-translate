package translate

import "fmt"

// A Chat request asks for structured output in response_format, and a
// Responses request in text.format. The formats are the same, except that a
// json_schema format's name, description, schema and strict sit in a
// json_schema object in Chat and directly in the format in Responses.

// responsesTextFormat returns the Responses text.format that carries a Chat
// response_format, and whether the request needs one: a format of type text
// asks for what Responses answers with anyway. ok is false for a value it
// cannot carry: not an object, another type, or a member neither API
// defines, which the conversion reports instead of dropping it silently.
func responsesTextFormat(value any) (format map[string]any, needed, ok bool) {
	chat, isObject := asMap(value)
	if !isObject {
		return nil, false, false
	}
	switch chat["type"] {
	case "text", "json_object":
		if !onlyMembers(chat, "type") {
			return nil, false, false
		}
		return map[string]any{"type": chat["type"]}, chat["type"] == "json_object", true
	case "json_schema":
		schema, isSchema := asMap(chat["json_schema"])
		if !isSchema || !onlyMembers(chat, "type", "json_schema") || !onlyMembers(schema, jsonSchemaMembers...) {
			return nil, false, false
		}
		format = map[string]any{"type": "json_schema"}
		for key, member := range schema {
			if member != nil {
				format[key] = member
			}
		}
		return format, true, true
	}
	return nil, false, false
}

// chatResponseFormat returns the Chat response_format that carries a
// Responses request's text, or nil when it asks for nothing beyond text
// output. text may hold only format: another member, such as verbosity,
// needs a native Responses model.
func chatResponseFormat(value any) (map[string]any, error) {
	text, ok := asMap(value)
	if !ok {
		return nil, fmt.Errorf("Responses text must be an object")
	}
	for key, member := range text {
		if key != "format" && member != nil {
			return nil, fmt.Errorf("Responses text field %q requires a native Responses model", key)
		}
	}
	if text["format"] == nil {
		return nil, nil
	}
	format, ok := asMap(text["format"])
	if !ok {
		return nil, fmt.Errorf("Responses text.format must be an object")
	}
	switch format["type"] {
	case "text", "json_object":
		if !onlyMembers(format, "type") {
			return nil, fmt.Errorf("Responses text.format of type %q takes no other member", format["type"])
		}
		if format["type"] == "text" {
			return nil, nil
		}
		return map[string]any{"type": "json_object"}, nil
	case "json_schema":
		if !onlyMembers(format, append([]string{"type"}, jsonSchemaMembers...)...) {
			return nil, fmt.Errorf("Responses text.format has a json_schema member Chat does not define")
		}
		schema := map[string]any{}
		for key, member := range format {
			if key != "type" && member != nil {
				schema[key] = member
			}
		}
		return map[string]any{"type": "json_schema", "json_schema": schema}, nil
	}
	return nil, fmt.Errorf("Responses text.format type %q requires a native Responses model", format["type"])
}

// jsonSchemaMembers are the members of a json_schema format that both APIs
// define.
var jsonSchemaMembers = []string{"name", "description", "schema", "strict"}

// onlyMembers reports an object whose set members are all among allowed.
func onlyMembers(object map[string]any, allowed ...string) bool {
	for key, member := range object {
		if member == nil {
			continue
		}
		known := false
		for _, name := range allowed {
			known = known || key == name
		}
		if !known {
			return false
		}
	}
	return true
}
