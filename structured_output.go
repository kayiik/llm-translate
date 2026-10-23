package translate

import (
	"encoding/json"
	"fmt"
)

// structuredOutputFindings checks only the finite adapting envelopes. Schema
// objects are opaque JSON data: no keyword rewriting or model-subset verdict.
func structuredOutputFindings(edge ConversionID, payload map[string]any) ([]Loss, []string) {
	f := structuredFindings{}
	if edge == ConversionChatToResponses {
		if raw := payload["response_format"]; raw != nil {
			f.format(raw, "response_format", true)
		}
	} else if edge == ConversionResponsesToChat {
		if raw := payload["text"]; raw != nil {
			text, ok := asMap(raw)
			if !ok || text == nil {
				f.invalid("text", "text configuration must be an object")
			} else {
				f.members(text, "text", "format")
				if format := text["format"]; format != nil {
					f.format(format, "text.format", false)
				}
			}
		}
	} else {
		return nil, nil
	}
	tools := mustList(payload["tools"])
	if raw := payload["tools"]; raw != nil {
		if _, ok := asList(raw); !ok {
			f.invalid("tools", "tools must be an evaluated list")
			return f.losses, f.unknown
		}
	}
	for i, raw := range tools {
		tool, ok := asMap(raw)
		path := fmt.Sprintf("tools.%d", i)
		if !ok || tool == nil {
			f.invalid(path, "tool envelope must be an object")
			continue
		}
		if tool["type"] != "function" {
			f.invalid(path+".type", "only client function tools are in this finite mapping")
			continue
		}
		fn := tool
		if edge == ConversionChatToResponses {
			f.members(tool, path, "type function")
			path += ".function"
			fn, ok = asMap(tool["function"])
			if !ok || fn == nil {
				f.invalid(path, "function definition must be an object")
				continue
			}
			f.members(fn, path, "name description parameters strict")
		} else {
			f.members(tool, path, "type name description parameters strict")
		}
		f.name(fn["name"], path+".name")
		f.optionalString(fn, "description", path)
		f.optionalBool(fn, "strict", path)
		if parameters, exists := fn["parameters"]; exists {
			if parameters == nil {
				if edge == ConversionChatToResponses {
					f.invalid(path+".parameters", "a supplied Chat parameter schema must be an object; omit it for no schema")
				}
			} else {
				f.schema(parameters, path+".parameters")
			}
		} else if edge == ConversionResponsesToChat {
			f.invalid(path+".parameters", "Responses requires an explicit object or null parameters field")
		}
		if fn["strict"] == nil {
			if edge == ConversionChatToResponses {
				f.losses = append(f.losses, advisory(path+".strict", LossRenamed, "Chat non-strict default is made explicit as Responses strict:false"))
			} else {
				f.unknown = append(f.unknown, path+".strict")
			}
		}
	}
	return f.losses, f.unknown
}

type structuredFindings struct {
	losses  []Loss
	unknown []string
}

func (f *structuredFindings) invalid(path, detail string) {
	f.losses = append(f.losses, material(path, LossUnsupported, detail))
}

func (f *structuredFindings) members(object map[string]any, path, known string) {
	for key := range object {
		if !assessmentKnown(known, key) {
			f.unknown = append(f.unknown, path+"."+key)
		}
	}
}

func (f *structuredFindings) format(raw any, path string, chat bool) {
	format, ok := asMap(raw)
	if !ok || format == nil {
		f.invalid(path, "format envelope must be an object")
		return
	}
	kind, ok := format["type"].(string)
	if !ok || kind == "" {
		f.invalid(path+".type", "format type must name a supported mode")
		return
	}
	switch kind {
	case "text", "json_object":
		f.members(format, path, "type")
	case "json_schema":
		schema := format
		if chat {
			f.members(format, path, "type json_schema")
			path += ".json_schema"
			schema, ok = asMap(format["json_schema"])
			if !ok || schema == nil {
				f.invalid(path, "json_schema envelope must be an object")
				return
			}
			f.members(schema, path, "name description schema strict")
		} else {
			f.members(schema, path, "type name description schema strict")
		}
		f.name(schema["name"], path+".name")
		f.schema(schema["schema"], path+".schema")
		f.optionalString(schema, "description", path)
		f.optionalBool(schema, "strict", path)
	default:
		f.invalid(path+".type", "format mode is outside this finite conversion")
	}
}

func (f *structuredFindings) name(raw any, path string) {
	name, ok := raw.(string)
	if !ok || name == "" || len(name) > 64 {
		f.invalid(path, "name must be a nonempty string of at most 64 supported characters")
		return
	}
	for _, c := range name {
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' {
			continue
		}
		f.invalid(path, "name contains a character outside the shared envelope alphabet")
		return
	}
}

func (f *structuredFindings) schema(raw any, path string) {
	object, ok := asMap(raw)
	if !ok || object == nil {
		f.invalid(path, "schema must be a JSON object for this conversion")
		return
	}
	if _, err := json.Marshal(object); err != nil {
		f.invalid(path, "schema object must contain JSON values")
	}
}

func (f *structuredFindings) optionalString(object map[string]any, key, path string) {
	if raw := object[key]; raw != nil {
		if _, ok := raw.(string); !ok {
			f.invalid(path+"."+key, "description must be a string when supplied")
		}
	}
}
func (f *structuredFindings) optionalBool(object map[string]any, key, path string) {
	if raw := object[key]; raw != nil {
		if _, ok := raw.(bool); !ok {
			f.invalid(path+"."+key, "strict must be a boolean or null")
		}
	}
}
