package translate

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"
)

// This file translates between the OpenAI Chat Completions schema (the gateway's
// internal pivot) and the OpenAI Responses API. Both are OpenAI-family standards,
// so these transforms are provider-agnostic — any backend that only exposes
// /responses for a model (e.g. Copilot's gpt-5.5) reuses them. They are pure
// functions; the decision of WHEN to apply them lives in the transport layer.

// ChatToResponses converts a Chat Completions request body into a Responses API
// request body.
func ChatToResponses(model string, messages []map[string]any, kw map[string]any, stream bool) map[string]any {
	payload := map[string]any{"model": model}
	var instructions []string
	input := make([]any, 0, len(messages))

	for _, m := range messages {
		role, _ := m["role"].(string)
		switch role {
		case "system", "developer":
			if t := contentToText(m["content"]); t != "" {
				instructions = append(instructions, t)
			}
		case "tool":
			callID, _ := m["tool_call_id"].(string)
			input = append(input, map[string]any{
				"type":    "function_call_output",
				"call_id": callID,
				"output":  contentToText(m["content"]),
			})
		case "assistant":
			if tc, ok := asList(m["tool_calls"]); ok && len(tc) > 0 {
				if t := contentToText(m["content"]); t != "" {
					input = append(input, map[string]any{"role": "assistant", "content": t})
				}
				for _, c := range tc {
					cm, ok := asMap(c)
					if !ok {
						continue
					}
					fn, _ := asMap(cm["function"])
					name, _ := fn["name"].(string)
					args, _ := fn["arguments"].(string)
					id, _ := cm["id"].(string)
					input = append(input, map[string]any{
						"type": "function_call", "call_id": id, "name": name, "arguments": args,
					})
				}
			} else {
				input = append(input, map[string]any{"role": "assistant", "content": contentToText(m["content"])})
			}
		default:
			r := role
			if r == "" {
				r = "user"
			}
			input = append(input, map[string]any{"role": r, "content": responsesUserContent(m["content"])})
		}
	}

	// Ensure every function_call item in input has a matching function_call_output.
	// If a client conversation history lost or omitted a tool output, OpenAI /responses
	// rejects the entire request with a 400 error.
	callOutputs := make(map[string]bool)
	for _, item := range input {
		if im, ok := item.(map[string]any); ok {
			if im["type"] == "function_call_output" {
				if cid, ok := im["call_id"].(string); ok && cid != "" {
					callOutputs[cid] = true
				}
			}
		}
	}
	for _, item := range input {
		if im, ok := item.(map[string]any); ok {
			if im["type"] == "function_call" {
				if cid, ok := im["call_id"].(string); ok && cid != "" && !callOutputs[cid] {
					input = append(input, map[string]any{
						"type":    "function_call_output",
						"call_id": cid,
						"output":  "",
					})
					callOutputs[cid] = true
				}
			}
		}
	}

	payload["input"] = input
	if len(instructions) > 0 {
		payload["instructions"] = strings.Join(instructions, "\n\n")
	}
	if v := firstNonNil(kw, "max_completion_tokens", "max_tokens"); v != nil {
		payload["max_output_tokens"] = v
	}
	if v := kw["reasoning_effort"]; v != nil {
		payload["reasoning"] = map[string]any{"effort": v}
	}
	if v := kw["temperature"]; v != nil {
		payload["temperature"] = v
	}
	if v := kw["top_p"]; v != nil {
		payload["top_p"] = v
	}
	if tools, ok := asList(kw["tools"]); ok && len(tools) > 0 {
		payload["tools"] = responsesTools(tools)
	}
	if tc := kw["tool_choice"]; tc != nil {
		payload["tool_choice"] = responsesToolChoice(tc)
	}
	if metadata := kw["metadata"]; metadata != nil {
		payload["metadata"] = metadata
	}
	if format, needed, ok := responsesTextFormat(kw["response_format"]); ok && needed {
		payload["text"] = map[string]any{"format": format}
	}
	if stream {
		payload["stream"] = true
	}
	return payload
}

// ResponsesRequestToChat converts supported Responses input/history/tools into
// the gateway's internal Chat representation. Built-in Responses-only tools
// fail explicitly instead of being silently flattened. A text.format becomes
// the response_format that asks Chat for the same structure.
func ResponsesRequestToChat(
	payload map[string]any,
) ([]map[string]any, map[string]any, error) {
	allowed := map[string]bool{
		"model": true, "input": true, "instructions": true, "stream": true,
		"max_output_tokens": true, "reasoning": true, "temperature": true,
		"top_p": true, "tools": true, "tool_choice": true,
		"metadata": true, "store": true, "background": true,
		"force_api_support": true, "previous_response_id": true, "conversation": true,
		"client_metadata": true, "include": true, "parallel_tool_calls": true,
		"prompt_cache_key": true, "text": true,
	}
	for key, value := range payload {
		if !allowed[key] && value != nil {
			return nil, nil, protocolErrorf("Responses",
				"field %q requires a native Responses model", key,
			)
		}
	}
	if value := payload["previous_response_id"]; value != nil && value != "" {
		return nil, nil, fmt.Errorf(
			"previous_response_id requires a native Responses model",
		)
	}
	if value := payload["conversation"]; value != nil {
		return nil, nil, fmt.Errorf(
			"conversation state requires a native Responses model",
		)
	}
	if value := payload["store"]; value != nil {
		enabled, ok := value.(bool)
		if !ok || enabled {
			return nil, nil, fmt.Errorf("store requires a native Responses model unless false")
		}
	}
	if value := payload["background"]; value != nil {
		enabled, ok := value.(bool)
		if !ok || enabled {
			return nil, nil, fmt.Errorf("background requires a native Responses model unless false")
		}
	}
	if value := payload["client_metadata"]; value != nil {
		if _, ok := value.(map[string]any); !ok {
			return nil, nil, fmt.Errorf("client_metadata must be an object")
		}
	}
	if value := payload["prompt_cache_key"]; value != nil {
		if _, ok := value.(string); !ok {
			return nil, nil, fmt.Errorf("prompt_cache_key must be a string")
		}
	}
	if value := payload["metadata"]; value != nil {
		if _, ok := value.(map[string]any); !ok {
			return nil, nil, fmt.Errorf("metadata must be an object")
		}
	}
	if value := payload["include"]; value != nil {
		items, ok := value.([]any)
		if !ok {
			return nil, nil, fmt.Errorf("include must be an array")
		}
		if len(items) > 0 {
			return nil, nil, protocolErrorf("Responses", "include values require a native Responses model")
		}
	}
	if instructions := payload["instructions"]; instructions != nil {
		if _, ok := instructions.(string); !ok {
			return nil, nil, fmt.Errorf(
				"structured instructions require a native Responses model",
			)
		}
	}
	messages := []map[string]any{}
	if instructions := contentToText(payload["instructions"]); instructions != "" {
		messages = append(messages, map[string]any{
			"role": "system", "content": instructions,
		})
	}
	appendMessage := func(role string, content any) {
		if role == "" {
			role = "user"
		}
		messages = append(messages, map[string]any{"role": role, "content": content})
	}
	switch input := payload["input"].(type) {
	case string:
		appendMessage("user", input)
	case []any:
		for i, raw := range input {
			item, ok := asMap(raw)
			if !ok {
				return nil, nil, protocolErrorf("Responses", "input item must be an object")
			}
			itemType, _ := item["type"].(string)
			switch itemType {
			case "", "message":
				role, _ := item["role"].(string)
				content, err := responsesInputToChatContent(item["content"])
				if err != nil {
					return nil, nil, responsesContentAt(fmt.Sprintf("input.%d.content", i), err)
				}
				appendMessage(role, content)
			case "function_call":
				call := map[string]any{
					"id": item["call_id"], "type": "function",
					"function": map[string]any{
						"name": item["name"], "arguments": item["arguments"],
					},
				}
				if len(messages) > 0 && messages[len(messages)-1]["role"] == "assistant" &&
					messages[len(messages)-1]["content"] == nil {
					calls, _ := messages[len(messages)-1]["tool_calls"].([]any)
					messages[len(messages)-1]["tool_calls"] = append(calls, call)
				} else {
					appendMessage("assistant", nil)
					messages[len(messages)-1]["tool_calls"] = []any{call}
				}
			case "function_call_output":
				output, err := responsesToolOutputToText(item["output"])
				if err != nil {
					return nil, nil, err
				}
				messages = append(messages, map[string]any{
					"role": "tool", "tool_call_id": item["call_id"],
					"content": output,
				})
			default:
				return nil, nil, protocolErrorf("Responses",
					"input type %q is not supported by Chat fallback", itemType,
				)
			}
		}
	case map[string]any:
		role, _ := input["role"].(string)
		content, err := responsesInputToChatContent(input["content"])
		if err != nil {
			return nil, nil, responsesContentAt("input.content", err)
		}
		appendMessage(role, content)
	case nil:
		return nil, nil, protocolErrorf("Responses", "input is required")
	default:
		return nil, nil, protocolErrorf("Responses", "input has an unsupported shape")
	}
	kw := map[string]any{}
	if value := payload["max_output_tokens"]; value != nil {
		kw["_max_output_tokens"] = value
	}
	if reasoning, ok := asMap(payload["reasoning"]); ok {
		for key, value := range reasoning {
			if key != "effort" && value != nil {
				return nil, nil, protocolErrorf("Responses",
					"reasoning field %q requires a native Responses model",
					key,
				)
			}
		}
		if effort := reasoning["effort"]; effort != nil {
			kw["reasoning_effort"] = effort
		}
	}
	for _, key := range []string{"temperature", "top_p"} {
		if value := payload[key]; value != nil {
			kw[key] = value
		}
	}
	if metadata, ok := payload["metadata"].(map[string]any); ok && len(metadata) > 0 {
		kw["metadata"] = metadata
	}
	if tools, ok := asList(payload["tools"]); ok {
		converted := make([]any, 0, len(tools))
		for _, raw := range tools {
			tool, ok := asMap(raw)
			if !ok || tool["type"] != "function" {
				return nil, nil, protocolErrorf("Responses",
					"built-in tools require a native Responses model",
				)
			}
			allowedToolFields := map[string]bool{
				"type": true, "name": true, "description": true,
				"parameters": true, "strict": true,
			}
			for key, value := range tool {
				if !allowedToolFields[key] && value != nil {
					return nil, nil, protocolErrorf("Responses",
						"function tool field %q requires a native Responses model",
						key,
					)
				}
			}
			function := map[string]any{"name": tool["name"]}
			if description := tool["description"]; description != nil {
				function["description"] = description
			}
			if parameters := tool["parameters"]; parameters != nil {
				function["parameters"] = parameters
			}
			if strict := tool["strict"]; strict != nil {
				function["strict"] = strict
			}
			converted = append(converted, map[string]any{
				"type": "function", "function": function,
			})
		}
		kw["tools"] = converted
	}
	if value := payload["parallel_tool_calls"]; value != nil {
		if _, ok := value.(bool); !ok {
			return nil, nil, fmt.Errorf("parallel_tool_calls must be a boolean")
		}
		kw["parallel_tool_calls"] = value
	}
	if choice := payload["tool_choice"]; choice != nil {
		if choiceMap, ok := asMap(choice); ok {
			if choiceMap["type"] != "function" {
				return nil, nil, protocolErrorf("Responses",
					"tool choice type %q requires a native Responses model",
					choiceMap["type"],
				)
			}
			kw["tool_choice"] = map[string]any{
				"type": "function", "function": map[string]any{"name": choiceMap["name"]},
			}
		} else {
			kw["tool_choice"] = choice
		}
	}
	if text := payload["text"]; text != nil {
		format, err := chatResponseFormat(text)
		if err != nil {
			return nil, nil, err
		}
		if format != nil {
			kw["response_format"] = format
		}
	}
	if payload["stream"] == true {
		kw["stream_options"] = map[string]any{"include_usage": true}
	}
	return messages, kw, nil
}

func responsesToolOutputToText(value any) (string, error) {
	if text, ok := value.(string); ok {
		return text, nil
	}
	parts, ok := asList(value)
	if !ok {
		return contentToText(value), nil
	}
	var text strings.Builder
	for _, raw := range parts {
		part, ok := asMap(raw)
		if !ok {
			return "", protocolErrorf("Responses", "function output part must be an object")
		}
		switch part["type"] {
		case "input_text", "text", "output_text":
			if value, _ := part["text"].(string); value != "" {
				text.WriteString(value)
			}
		default:
			return "", protocolErrorf("Responses",
				"function output type %q requires a native Responses model",
				part["type"],
			)
		}
	}
	return text.String(), nil
}

// responsesContentError retains a source-part path without changing existing
// native-required error messages exposed by ResponsesRequestToChat.
type responsesContentError struct {
	path string
	err  error
}

func (e *responsesContentError) Error() string { return e.err.Error() }

func responsesContentAt(prefix string, err error) error {
	if nested, ok := err.(*responsesContentError); ok {
		return &responsesContentError{path: prefix + "." + nested.path, err: nested.err}
	}
	return &responsesContentError{path: prefix, err: err}
}

func responsesInputToChatContent(value any) (any, error) {
	if text, ok := value.(string); ok {
		return text, nil
	}
	parts, ok := asList(value)
	if !ok {
		return contentToText(value), nil
	}
	out := make([]any, 0, len(parts))
	var textOnly strings.Builder
	allText := true
	for i, raw := range parts {
		part, ok := asMap(raw)
		if !ok {
			continue
		}
		switch part["type"] {
		case "input_text", "text", "output_text":
			if value, _ := part["text"].(string); value != "" {
				textOnly.WriteString(value)
			}
			out = append(out, map[string]any{
				"type": "text", "text": part["text"],
			})
		case "input_image", "image_url":
			allText = false
			if part["file_id"] != nil {
				return nil, responsesContentAt(fmt.Sprintf("%d.file_id", i), protocolErrorf("Responses",
					"file-backed images require a native Responses model",
				))
			}
			imageURL := part["image_url"]
			if imageURL == nil {
				imageURL = part["url"]
			}
			if imageURL == nil || imageURL == "" {
				return nil, responsesContentAt(fmt.Sprintf("%d.image_url", i), protocolErrorf("Responses", "input image requires image_url"))
			}
			image := map[string]any{"url": imageURL}
			if detail := part["detail"]; detail != nil {
				image["detail"] = detail
			}
			out = append(out, map[string]any{
				"type":      "image_url",
				"image_url": image,
			})
		case "input_file":
			if part["file_url"] != nil {
				return nil, responsesContentAt(fmt.Sprintf("%d.file_url", i), protocolErrorf("Responses", "remote file URLs require a native Responses model"))
			}
			file := responseFileFields(part)
			if losses := fileInputLosses(file, ""); len(losses) > 0 {
				return nil, responsesContentAt(fmt.Sprint(i)+losses[0].Path, protocolErrorf("Responses", "file input requires a valid file_data or file_id carrier"))
			}
			allText = false
			out = append(out, map[string]any{"type": "file", "file": file})
		case "refusal", "file":
			return nil, responsesContentAt(fmt.Sprint(i), protocolErrorf("Responses",
				"content type %q requires a native Responses model",
				part["type"],
			))
		default:
			return nil, responsesContentAt(fmt.Sprint(i), protocolErrorf("Responses",
				"content type %q is not supported by Chat fallback",
				part["type"],
			))
		}
	}
	if allText {
		return textOnly.String(), nil
	}
	return out, nil
}

// ResponsesToChat converts a Responses API object into a chat.completion.
func ResponsesToChat(model string, resp map[string]any) map[string]any {
	id, _ := resp["id"].(string)
	if id == "" {
		id = chatCmplID()
	}
	var text strings.Builder
	var toolCalls []any
	tcIndex := 0
	if output, ok := asList(resp["output"]); ok {
		for _, item := range output {
			im, ok := asMap(item)
			if !ok {
				continue
			}
			switch im["type"] {
			case "message":
				if content, ok := asList(im["content"]); ok {
					for _, part := range content {
						pm, ok := asMap(part)
						if !ok {
							continue
						}
						if pm["type"] == "output_text" {
							if t, ok := pm["text"].(string); ok {
								text.WriteString(t)
							}
						}
					}
				}
			case "function_call":
				callID, _ := im["call_id"].(string)
				if callID == "" {
					callID, _ = im["id"].(string)
				}
				name, _ := im["name"].(string)
				args, _ := im["arguments"].(string)
				toolCalls = append(toolCalls, map[string]any{
					"index": tcIndex, "id": callID, "type": "function",
					"function": map[string]any{"name": name, "arguments": args},
				})
				tcIndex++
			}
		}
	}
	if text.Len() == 0 && len(toolCalls) == 0 {
		if ot, ok := resp["output_text"].(string); ok {
			text.WriteString(ot)
		}
	}

	msg := map[string]any{"role": "assistant"}
	finish := "stop"
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
		msg["content"] = nil
		if text.Len() > 0 {
			msg["content"] = text.String()
		}
		finish = "tool_calls"
	} else {
		msg["content"] = text.String()
	}
	if responsesIncompleteMaxTokens(resp) {
		finish = "length"
	}

	out := map[string]any{
		"id": id, "object": "chat.completion", "model": model,
		"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": finish}},
	}
	if usage := responsesUsage(resp["usage"]); usage != nil {
		out["usage"] = usage
	}
	return out
}

// ChatResponseToResponses converts the gateway's internal Chat Completions
// result back into the public Responses API envelope.
func ChatResponseToResponses(model string, chat map[string]any) map[string]any {
	return ChatResponseToResponsesWithRequest(model, chat, nil)
}

func ChatResponseToResponsesWithRequest(
	model string, chat map[string]any, request map[string]any,
) map[string]any {
	id, _ := chat["id"].(string)
	if strings.HasPrefix(id, "chatcmpl-") {
		id = "resp_" + strings.TrimPrefix(id, "chatcmpl-")
	}
	if id == "" {
		id = "resp_" + strings.TrimPrefix(chatCmplID(), "chatcmpl-")
	}
	output := []any{}
	outputText := ""
	status := "completed"
	var incomplete any
	if choices, ok := asList(chat["choices"]); ok && len(choices) > 0 {
		if choice, ok := asMap(choices[0]); ok {
			if reason, _ := choice["finish_reason"].(string); reason == "length" ||
				reason == "content_filter" {
				status = "incomplete"
				incompleteReason := "max_output_tokens"
				if reason == "content_filter" {
					incompleteReason = "content_filter"
				}
				incomplete = map[string]any{"reason": incompleteReason}
			}
			if message, ok := asMap(choice["message"]); ok {
				messageContent := []any{}
				if text, ok := message["content"].(string); ok && text != "" {
					outputText = text
					messageContent = append(messageContent, map[string]any{
						"type": "output_text", "text": text, "annotations": []any{},
					})
				}
				if refusal, _ := message["refusal"].(string); refusal != "" {
					messageContent = append(messageContent, map[string]any{
						"type": "refusal", "refusal": refusal,
					})
				}
				if len(messageContent) > 0 {
					output = append(output, map[string]any{
						"id":   "msg_" + strings.TrimPrefix(id, "resp_"),
						"type": "message", "status": status, "role": "assistant",
						"content": messageContent,
					})
				}
				if calls, ok := asList(message["tool_calls"]); ok {
					for _, value := range calls {
						call, ok := asMap(value)
						if !ok {
							continue
						}
						function, _ := asMap(call["function"])
						output = append(output, map[string]any{
							"type": "function_call", "status": status,
							"id": call["id"], "call_id": call["id"],
							"name": function["name"], "arguments": function["arguments"],
						})
					}
				}
			}
		}
	}
	response := NewResponseEnvelope(model, id, status, output, request)
	response["output_text"] = outputText
	if incomplete != nil {
		response["incomplete_details"] = incomplete
	}
	if mapped, _ := chatResponseUsage(chat["usage"]); mapped != nil {
		response["usage"] = mapped
	}

	return response
}

// copyTokenDetails carries supplied counters without inventing measurements.
// Required target-wire placeholders, if any, are added separately by the caller.
func copyTokenDetails(target map[string]any, targetKey string, source map[string]any, sourceKey string, fields ...string) {
	details, _ := asMap(source[sourceKey])
	mapped := map[string]any{}
	for _, field := range fields {
		if value := details[field]; value != nil {
			mapped[field] = value
		}
	}
	if len(mapped) > 0 {
		target[targetKey] = mapped
	}
}

// NewResponseEnvelope creates the schema-complete Response shape used whenever
// the gateway adapts a Chat-only backend or synthesizes a terminal failure.
func NewResponseEnvelope(
	model, id, status string, output []any, request map[string]any,
) map[string]any {
	if id == "" {
		id = "resp_" + strings.TrimPrefix(chatCmplID(), "chatcmpl-")
	}
	value := func(key string, fallback any) any {
		if request != nil {
			if current, ok := request[key]; ok {
				return current
			}
		}
		return fallback
	}
	reasoning := value("reasoning", map[string]any{
		"effort": nil, "summary": nil,
	})
	text := value("text", map[string]any{
		"format": map[string]any{"type": "text"},
	})
	metadata := value("metadata", map[string]any{})
	return map[string]any{
		"id": id, "object": "response", "created_at": time.Now().Unix(),
		"status": status, "background": false, "error": nil,
		"incomplete_details": nil, "instructions": value("instructions", nil),
		"max_output_tokens": value("max_output_tokens", nil),
		"max_tool_calls":    value("max_tool_calls", nil), "model": model,
		"output": output, "parallel_tool_calls": value("parallel_tool_calls", true),
		"previous_response_id": value("previous_response_id", nil),
		"reasoning":            reasoning, "service_tier": value("service_tier", "default"),
		"store": value("store", false), "temperature": value("temperature", nil),
		"text": text, "tool_choice": value("tool_choice", "auto"),
		"tools": value("tools", []any{}), "top_logprobs": value("top_logprobs", 0),
		"top_p": value("top_p", nil), "truncation": value("truncation", "disabled"),
		"usage": nil, "user": value("user", nil), "metadata": metadata,
	}
}

func FailedResponseEnvelope(
	model, id, code, message string, request map[string]any,
) map[string]any {
	response := NewResponseEnvelope(model, id, "failed", []any{}, request)
	response["error"] = map[string]any{
		"code": code, "message": message,
	}
	return response
}

// firstNumber retains absence/invalidity and validates exact source integers.
// A bad preferred field cannot silently fall through to an alias or become 0.
func firstNumber(values map[string]any, keys ...string) CountObservation {
	for _, key := range keys {
		count := observedCount(values, key, UsageReported)
		if count.Presence != CountAbsent {
			return count
		}
	}
	return CountObservation{Presence: CountAbsent, Provenance: UsageReported, Path: keys[0]}
}

func chatResponseUsage(raw any) (map[string]any, Report) {
	if raw == nil {
		return nil, Report{}
	}
	usage, ok := asMap(raw)
	if !ok {
		return nil, NewReport(material("usage", LossUnsupported, "source usage must be an object"))
	}
	if len(usage) == 0 {
		return nil, Report{}
	}
	input := firstNumber(usage, "prompt_tokens", "input_tokens")
	output := firstNumber(usage, "completion_tokens", "output_tokens")
	total := firstNumber(usage, "total_tokens")
	var losses []Loss
	for _, count := range []CountObservation{input, output, total} {
		if count.Presence == CountInvalid {
			losses = append(losses, material("usage."+count.Path, LossUnsupported, "source aggregate count is invalid or outside exact int64 range; no zero is synthesized"))
		}
	}
	if len(losses) > 0 {
		return nil, NewReport(losses...)
	}
	// Preserve the legacy absent/zero-total fallback, but bound arithmetic.
	// Missing aggregate fields are target placeholders, never source facts.
	if total.Value == 0 {
		if output.Value > math.MaxInt64-input.Value {
			return nil, NewReport(material("usage.total_tokens", LossUnsupported, "derived aggregate exceeds int64 range"))
		}
		total.Value = input.Value + output.Value
	}
	mapped := map[string]any{"input_tokens": usageWireInteger(input.Value), "output_tokens": usageWireInteger(output.Value), "total_tokens": usageWireInteger(total.Value)}
	copyTokenDetails(mapped, "input_tokens_details", usage, "prompt_tokens_details", "cached_tokens", "cache_write_tokens")
	copyTokenDetails(mapped, "output_tokens_details", usage, "completion_tokens_details", "reasoning_tokens")
	// Retain required legacy target shape without labelling placeholders as
	// provider measurements; source observations use the original usage.
	for key, field := range map[string]string{"input_tokens_details": "cached_tokens", "output_tokens_details": "reasoning_tokens"} {
		details, ok := asMap(mapped[key])
		if !ok {
			details = map[string]any{}
			mapped[key] = details
		}
		if details[field] == nil {
			details[field] = 0
		}
	}
	return mapped, Report{}
}

func usageWireInteger(value int64) any {
	if int64(int(value)) == value {
		return int(value)
	}
	return value
}

// ResponsesToChatChunks renders a (buffered) Responses reply as a small sequence
// of chat.completion.chunk payloads for the streaming code path.
func ResponsesToChatChunks(model string, resp map[string]any) []string {
	chat := ResponsesToChat(model, resp)
	choice := chat["choices"].([]any)[0].(map[string]any)
	msg := choice["message"].(map[string]any)
	finish := choice["finish_reason"]
	id, _ := chat["id"].(string)

	mk := func(delta map[string]any, fin any, usage any) string {
		c := map[string]any{
			"id": id, "object": "chat.completion.chunk", "model": model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": fin}},
		}
		if usage != nil {
			c["usage"] = usage
		}
		b, _ := json.Marshal(c)
		return string(b)
	}

	chunks := []string{mk(map[string]any{"role": "assistant"}, nil, nil)}
	if content, ok := msg["content"].(string); ok && content != "" {
		chunks = append(chunks, mk(map[string]any{"content": content}, nil, nil))
	}
	if tc, ok := msg["tool_calls"]; ok {
		chunks = append(chunks, mk(map[string]any{"tool_calls": tc}, nil, nil))
	}
	chunks = append(chunks, mk(map[string]any{}, finish, chat["usage"]))
	return chunks
}

// ---- helpers ------------------------------------------------------------ //

func firstNonNil(kw map[string]any, keys ...string) any {
	for _, k := range keys {
		if v, ok := kw[k]; ok && v != nil {
			return v
		}
	}
	return nil
}

func contentToText(v any) string {
	switch c := v.(type) {
	case string:
		return c
	case []any:
		var b strings.Builder
		for _, part := range c {
			if pm, ok := asMap(part); ok {
				if t, ok := pm["text"].(string); ok {
					b.WriteString(t)
				}
			}
		}
		return b.String()
	default:
		return ""
	}
}

func responsesUserContent(v any) any {
	switch c := v.(type) {
	case string:
		return c
	case []any:
		out := make([]any, 0, len(c))
		for _, part := range c {
			pm, ok := asMap(part)
			if !ok {
				continue
			}
			switch pm["type"] {
			case "text", "input_text":
				out = append(out, map[string]any{"type": "input_text", "text": pm["text"]})
			case "image_url", "input_image":
				img := pm["image_url"]
				detail := pm["detail"]
				if m, ok := asMap(img); ok {
					img = m["url"]
					detail = m["detail"]
				}
				image := map[string]any{"type": "input_image", "image_url": img}
				if detail != nil {
					image["detail"] = detail
				}
				out = append(out, image)
			case "file":
				file, _ := asMap(pm["file"])
				mapped := responseFileFields(file)
				mapped["type"] = "input_file"
				out = append(out, mapped)
			default:
				out = append(out, part)
			}
		}
		if len(out) == 0 {
			return ""
		}
		return out
	default:
		return contentToText(v)
	}
}

func responsesTools(tools []any) []any {
	out := make([]any, 0, len(tools))
	for _, t := range tools {
		tm, ok := asMap(t)
		if !ok {
			out = append(out, t)
			continue
		}
		if tm["type"] == "function" {
			if fn, ok := asMap(tm["function"]); ok {
				// Responses requires nullable parameters/strict fields and has a
				// different implicit strict policy. Preserve Chat's non-strict
				// default explicitly; do not normalize or rewrite its schema.
				rt := map[string]any{"type": "function", "name": fn["name"], "parameters": nil, "strict": false}
				if v, ok := fn["description"]; ok {
					rt["description"] = v
				}
				if v, ok := fn["parameters"]; ok {
					rt["parameters"] = v
				}
				if v, ok := fn["strict"]; ok && v != nil {
					rt["strict"] = v
				}
				out = append(out, rt)
				continue
			}
		}
		out = append(out, tm)
	}
	return out
}

func responsesToolChoice(tc any) any {
	m, ok := asMap(tc)
	if !ok {
		return tc // "auto" | "none" | "required"
	}
	if m["type"] == "function" {
		if fn, ok := asMap(m["function"]); ok {
			return map[string]any{"type": "function", "name": fn["name"]}
		}
	}
	return tc
}

func responsesUsage(v any) map[string]any {
	mapped, _ := responsesUsageWithReport(v)
	return mapped
}

func responsesUsageWithReport(v any) (map[string]any, Report) {
	if v == nil {
		return nil, Report{}
	}
	u, ok := asMap(v)
	if !ok {
		return nil, NewReport(material("usage", LossUnsupported, "source usage must be an object"))
	}
	in := firstNumber(u, "input_tokens")
	out := firstNumber(u, "output_tokens")
	total := firstNumber(u, "total_tokens")
	var losses []Loss
	for _, count := range []CountObservation{in, out, total} {
		if count.Presence == CountInvalid {
			losses = append(losses, material("usage."+count.Path, LossUnsupported, "source aggregate count is invalid or outside exact int64 range; no zero is synthesized"))
		}
	}
	if len(losses) > 0 {
		return nil, NewReport(losses...)
	}
	// Derive only an absent total. A reported zero is source evidence, not
	// a missing value; absent child counters retain legacy wire placeholders.
	if total.Presence == CountAbsent {
		if out.Value > math.MaxInt64-in.Value {
			return nil, NewReport(material("usage.total_tokens", LossUnsupported, "derived aggregate exceeds int64 range"))
		}
		total.Value = in.Value + out.Value
	}
	mapped := map[string]any{"prompt_tokens": usageWireInteger(in.Value), "completion_tokens": usageWireInteger(out.Value), "total_tokens": usageWireInteger(total.Value)}
	copyTokenDetails(mapped, "prompt_tokens_details", u, "input_tokens_details", "cached_tokens", "cache_write_tokens")
	copyTokenDetails(mapped, "completion_tokens_details", u, "output_tokens_details", "reasoning_tokens")
	return mapped, Report{}
}

func responsesIncompleteMaxTokens(resp map[string]any) bool {
	if resp["status"] != "incomplete" {
		return false
	}
	d, ok := asMap(resp["incomplete_details"])
	if !ok {
		return false
	}
	return d["reason"] == "max_output_tokens"
}

// PreferredEndpoint decides which OpenAI-family endpoint a model should use
// given its published supported_surfaces. Returns "responses" only when the
// model advertises /responses but NOT /chat/completions; otherwise "chat"
// (the safe default, including when the list is empty/unknown).
func PreferredEndpoint(supportedEndpoints []string) string {
	if len(supportedEndpoints) == 0 {
		return "chat"
	}
	hasChat, hasResponses := false, false
	for _, e := range supportedEndpoints {
		switch strings.TrimSpace(strings.ToLower(e)) {
		case "/chat/completions":
			hasChat = true
		case "/responses":
			hasResponses = true
		}
	}
	if hasResponses && !hasChat {
		return "responses"
	}
	return "chat"
}
