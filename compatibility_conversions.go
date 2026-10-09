package translate

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// OpenAIChatRequest is the two-part request value returned by Chat conversions.
type OpenAIChatRequest struct {
	Messages []map[string]any
	Keywords map[string]any
}

// AnthropicMessages is the two-part request value returned by Anthropic conversions.
type AnthropicMessages struct {
	System   string
	Messages []map[string]any
	// SystemBlocks is set when a system or developer text part carries
	// cache_control. It holds the System text as Anthropic text blocks split
	// at the breakpoints, each carrying its cache_control. Send it as the
	// request's system value instead of System, or the breakpoints are lost.
	SystemBlocks []any `json:",omitempty"`
}

// AnthropicRequestToOpenAIWithReport delegates to AnthropicRequestToOpenAI and
// gives its incompatible paths structured material-loss semantics.
func AnthropicRequestToOpenAIWithReport(payload map[string]any) ConversionResult[OpenAIChatRequest] {
	messages, keywords, incompatible := AnthropicRequestToOpenAI(payload)
	losses := make([]Loss, 0, len(incompatible))
	for _, path := range incompatible {
		losses = append(losses, material(path, LossUnsupported, "value is not representable by OpenAI Chat"))
	}
	// cache_control is already reported above at its exact path. Thinking
	// blocks are too, at the block path; the .reasoning loss lets a policy
	// address reasoning the same way on every surface.
	rawMessages, _ := asList(payload["messages"])
	for i, raw := range rawMessages {
		message, _ := asMap(raw)
		blocks, _ := asList(message["content"])
		for j, rawBlock := range blocks {
			if block, _ := asMap(rawBlock); isThinkingBlock(block) {
				losses = append(losses, droppedReasoning(fmt.Sprintf("messages.%d.content.%d", i, j)))
			}
		}
	}
	return ConversionResult[OpenAIChatRequest]{
		Value:  OpenAIChatRequest{Messages: messages, Keywords: keywords},
		Report: NewReport(losses...),
	}
}

// OpenAIMessagesToAnthropicWithReport converts messages and reports known
// normalizations performed by the existing converter. cache_control on text
// parts is carried, including on system parts through SystemBlocks; vendor
// fields Anthropic cannot represent are reported at their paths.
func OpenAIMessagesToAnthropicWithReport(messages []map[string]any) ConversionResult[AnthropicMessages] {
	conversion := openAIMessagesToAnthropic(messages)
	var losses []Loss
	for _, path := range conversion.droppedCacheControl {
		losses = append(losses, droppedCacheControl(path, "Anthropic Messages"))
	}
	for i, message := range messages {
		path := "messages." + strconv.Itoa(i)
		losses = append(losses, chatMessageVendorLosses(path, message, "Anthropic Messages")...)
		role, _ := message["role"].(string)
		if role == "developer" {
			losses = append(losses, advisory(path+".role", LossApproximated, "developer instruction is merged into Anthropic system text"))
		}
		if role != "system" && role != "developer" && role != "assistant" && role != "tool" && role != "user" && role != "" {
			losses = append(losses, material(path+".role", LossApproximated, "role is converted to user"))
		}
		if role == "assistant" {
			if calls, ok := asList(message["tool_calls"]); ok {
				for j, raw := range calls {
					call, _ := asMap(raw)
					fn, _ := asMap(call["function"])
					args, _ := fn["arguments"].(string)
					var object map[string]any
					if args != "" && json.Unmarshal([]byte(args), &object) != nil {
						losses = append(losses, material(fmt.Sprintf("%s.tool_calls.%d.function.arguments", path, j), LossApproximated, "invalid or non-object JSON arguments become an empty object"))
					}
				}
			}
		}
	}
	value := AnthropicMessages{System: conversion.system, Messages: conversion.messages, SystemBlocks: conversion.systemBlocks}
	return ConversionResult[AnthropicMessages]{Value: value, Report: NewReport(losses...)}
}

// OpenAIResponseToAnthropicWithReport converts a Chat response with fidelity findings.
func OpenAIResponseToAnthropicWithReport(response map[string]any, model string) ConversionResult[map[string]any] {
	var losses []Loss
	if choices, ok := asList(response["choices"]); ok && len(choices) > 1 {
		losses = append(losses, material("choices.1", LossDropped, "only the first Chat choice is converted"))
	}
	if choices, ok := asList(response["choices"]); ok && len(choices) > 0 {
		choice, _ := asMap(choices[0])
		if choice["finish_reason"] == "content_filter" {
			losses = append(losses, advisory("choices.0.finish_reason", LossApproximated, "content_filter maps to Anthropic end_turn"))
		}
		message, _ := asMap(choice["message"])
		losses = append(losses, chatMessageVendorLosses("choices.0.message", message, "Anthropic Messages")...)
	}
	return ConversionResult[map[string]any]{Value: OpenAIResponseToAnthropic(response, model), Report: NewReport(losses...)}
}

// AnthropicResponseToOpenAIWithReport converts an Anthropic response with fidelity findings.
func AnthropicResponseToOpenAIWithReport(response map[string]any, model string) ConversionResult[map[string]any] {
	var losses []Loss
	if blocks, ok := asList(response["content"]); ok {
		for i, raw := range blocks {
			block, _ := asMap(raw)
			if kind, _ := block["type"].(string); kind != "text" && kind != "tool_use" {
				losses = append(losses, material(fmt.Sprintf("content.%d", i), LossDropped, "Anthropic content block is not representable by Chat"))
			}
			if isThinkingBlock(block) {
				losses = append(losses, droppedReasoning(fmt.Sprintf("content.%d", i)))
			}
		}
	}
	return ConversionResult[map[string]any]{Value: AnthropicResponseToOpenAI(response, model), Report: NewReport(losses...)}
}

// ChatToResponsesWithReport converts a Chat request and reports known fallback synthesis.
func ChatToResponsesWithReport(model string, messages []map[string]any, kw map[string]any, stream bool) ConversionResult[map[string]any] {
	var losses []Loss
	if kw["max_completion_tokens"] != nil {
		losses = append(losses, advisory("max_completion_tokens", LossRenamed, "converted to max_output_tokens"))
	} else if kw["max_tokens"] != nil {
		losses = append(losses, advisory("max_tokens", LossRenamed, "converted to max_output_tokens"))
	}
	supported := map[string]bool{"max_completion_tokens": true, "max_tokens": true, "reasoning_effort": true, "temperature": true, "top_p": true, "tools": true, "tool_choice": true, "metadata": true}
	for key, value := range kw {
		if value != nil && !supported[key] {
			losses = append(losses, material(key, LossDropped, "Chat option is not emitted in the Responses request"))
		}
	}
	outputs := map[string]bool{}
	for _, message := range messages {
		if message["role"] == "tool" {
			if id, _ := message["tool_call_id"].(string); id != "" {
				outputs[id] = true
			}
		}
	}
	for i, message := range messages {
		if calls, ok := asList(message["tool_calls"]); ok {
			for j, raw := range calls {
				call, _ := asMap(raw)
				id, _ := call["id"].(string)
				if id != "" && !outputs[id] {
					losses = append(losses, advisory(fmt.Sprintf("messages.%d.tool_calls.%d", i, j), LossApproximated, "missing tool output is synthesized as an empty string"))
				}
			}
		}
		prefix := "messages." + strconv.Itoa(i)
		losses = append(losses, chatMessageVendorLosses(prefix, message, "OpenAI Responses")...)
		losses = append(losses, responsesDroppedCacheControl(prefix, message)...)
	}
	return ConversionResult[map[string]any]{Value: ChatToResponses(model, messages, kw, stream), Report: NewReport(losses...)}
}

// responsesDroppedCacheControl reports the content-part cache_control values
// that ChatToResponses drops. Responses has no cache breakpoints: it rebuilds
// known user parts and flattens every other role to text. Unknown user part
// types pass through unchanged, so their members are not dropped.
func responsesDroppedCacheControl(prefix string, message map[string]any) []Loss {
	role, _ := message["role"].(string)
	passthrough := role != "system" && role != "developer" && role != "assistant" && role != "tool"
	parts, _ := asList(message["content"])
	var losses []Loss
	for k, raw := range parts {
		part, ok := asMap(raw)
		if !ok || !present(part["cache_control"]) {
			continue
		}
		switch part["type"] {
		case "text", "input_text", "image_url", "input_image":
		default:
			if passthrough {
				continue
			}
		}
		losses = append(losses, droppedCacheControl(fmt.Sprintf("%s.content.%d.cache_control", prefix, k), "OpenAI Responses"))
	}
	return losses
}

// ResponsesRequestToChatWithReport preserves existing validation errors and
// reports unsupported top-level fields as material compatibility loss.
func ResponsesRequestToChatWithReport(payload map[string]any) (ConversionResult[OpenAIChatRequest], error) {
	known := map[string]bool{"model": true, "input": true, "instructions": true, "stream": true, "max_output_tokens": true, "reasoning": true, "temperature": true, "top_p": true, "tools": true, "tool_choice": true, "metadata": true, "store": true, "background": true, "force_api_support": true, "previous_response_id": true, "conversation": true, "client_metadata": true, "include": true, "parallel_tool_calls": true, "prompt_cache_key": true}
	var losses []Loss
	for key, value := range payload {
		if value != nil && !known[key] {
			losses = append(losses, material(key, LossUnsupported, "Responses field requires a native Responses model"))
		}
	}
	if payload["previous_response_id"] != nil && payload["previous_response_id"] != "" {
		losses = append(losses, material("previous_response_id", LossUnsupported, "server-side response state is not representable by Chat"))
	}
	if payload["conversation"] != nil {
		losses = append(losses, material("conversation", LossUnsupported, "conversation state is not representable by Chat"))
	}
	if values, ok := asList(payload["include"]); ok && len(values) > 0 {
		losses = append(losses, material("include", LossUnsupported, "included Responses-only data is not representable by Chat"))
	}
	messages, keywords, err := ResponsesRequestToChat(payload)
	if err != nil && len(losses) == 0 {
		losses = append(losses, material("$", LossUnsupported, err.Error()))
	}
	return ConversionResult[OpenAIChatRequest]{Value: OpenAIChatRequest{Messages: messages, Keywords: keywords}, Report: NewReport(losses...)}, err
}

// ResponsesToChatWithReport converts a Responses object with fidelity findings.
func ResponsesToChatWithReport(model string, response map[string]any) ConversionResult[map[string]any] {
	var losses []Loss
	if output, ok := asList(response["output"]); ok {
		for i, raw := range output {
			item, _ := asMap(raw)
			kind, _ := item["type"].(string)
			if kind == "reasoning" {
				losses = append(losses, advisory(fmt.Sprintf("output.%d", i), LossDropped, "Responses reasoning content is not emitted by Chat"))
				losses = append(losses, droppedReasoning(fmt.Sprintf("output.%d", i)))
			} else if kind != "message" && kind != "function_call" {
				losses = append(losses, material(fmt.Sprintf("output.%d", i), LossDropped, "Responses output item is not representable by Chat"))
			}
			if kind == "message" {
				if content, ok := asList(item["content"]); ok {
					for j, rawPart := range content {
						part, _ := asMap(rawPart)
						if part["type"] != "output_text" {
							losses = append(losses, material(fmt.Sprintf("output.%d.content.%d", i, j), LossDropped, "Responses content part is not representable by Chat"))
						} else if annotations, ok := asList(part["annotations"]); ok && len(annotations) > 0 {
							losses = append(losses, material(fmt.Sprintf("output.%d.content.%d.annotations", i, j), LossDropped, "text annotations are not emitted by Chat"))
						}
					}
				}
			}
		}
	}
	if response["status"] == "incomplete" && !responsesIncompleteMaxTokens(response) {
		losses = append(losses, material("incomplete_details.reason", LossApproximated, "Chat cannot preserve this incomplete reason"))
	}
	return ConversionResult[map[string]any]{Value: ResponsesToChat(model, response), Report: NewReport(losses...)}
}

// ChatResponseToResponsesWithReport converts a Chat response with fidelity findings.
func ChatResponseToResponsesWithReport(model string, chat map[string]any) ConversionResult[map[string]any] {
	return ChatResponseToResponsesWithRequestAndReport(model, chat, nil)
}

// ChatResponseToResponsesWithRequestAndReport also uses request fields when
// constructing the Responses envelope.
func ChatResponseToResponsesWithRequestAndReport(model string, chat map[string]any, request map[string]any) ConversionResult[map[string]any] {
	var losses []Loss
	if choices, ok := asList(chat["choices"]); ok && len(choices) > 1 {
		losses = append(losses, material("choices.1", LossDropped, "only the first Chat choice is converted"))
	}
	if choices, ok := asList(chat["choices"]); ok && len(choices) > 0 {
		choice, _ := asMap(choices[0])
		message, _ := asMap(choice["message"])
		losses = append(losses, chatMessageVendorLosses("choices.0.message", message, "OpenAI Responses")...)
	}
	return ConversionResult[map[string]any]{Value: ChatResponseToResponsesWithRequest(model, chat, request), Report: NewReport(losses...)}
}

// ResponsesToChatChunksWithReport reports that a complete response is rendered
// as buffered chunks rather than preserving original event boundaries.
func ResponsesToChatChunksWithReport(model string, response map[string]any) ConversionResult[[]string] {
	base := ResponsesToChatWithReport(model, response)
	losses := append([]Loss(nil), base.Report.Losses...)
	losses = append(losses, advisory("stream", LossApproximated, "buffered response is rendered as synthetic Chat chunks"))
	return ConversionResult[[]string]{Value: ResponsesToChatChunks(model, response), Report: NewReport(losses...)}
}

// OpenAIStreamToAnthropicSSEWithReport delegates stream conversion and reports
// the token estimate used when chunks do not carry usage, and the vendor
// fields of deltas that Anthropic events cannot carry.
func OpenAIStreamToAnthropicSSEWithReport(chunks func() (string, bool), model string, emit func(string)) Report {
	sawText := false
	sawOutputUsage := false
	var losses []Loss
	observed := func() (string, bool) {
		raw, ok := chunks()
		if !ok {
			return raw, false
		}
		var chunk map[string]any
		if json.Unmarshal([]byte(raw), &chunk) == nil {
			if usage, ok := asMap(chunk["usage"]); ok && usage["completion_tokens"] != nil {
				sawOutputUsage = true
			}
			if choices, ok := asList(chunk["choices"]); ok {
				for _, rawChoice := range choices {
					choice, _ := asMap(rawChoice)
					delta, _ := asMap(choice["delta"])
					if text, _ := delta["content"].(string); text != "" {
						sawText = true
					}
					losses = append(losses, deltaVendorLosses(choice, delta)...)
				}
			}
		}
		return raw, true
	}
	OpenAIStreamToAnthropicSSE(observed, model, emit)
	if sawText && !sawOutputUsage {
		losses = append(losses, advisory("usage.output_tokens", LossApproximated, "output tokens are estimated from streamed text because usage is absent"))
	}
	return NewReport(losses...)
}

// deltaVendorLosses reports the vendor fields of one stream delta. Paths use
// the choice index and the tool call index, which identify a call across the
// chunks of a stream; repeats collapse in the report.
func deltaVendorLosses(choice, delta map[string]any) []Loss {
	prefix := "choices." + strconv.Itoa(toInt(choice["index"])) + ".delta"
	var losses []Loss
	if calls, ok := asList(delta["tool_calls"]); ok {
		for _, raw := range calls {
			if call, ok := asMap(raw); ok {
				losses = append(losses, thoughtSignatureLosses(prefix+".tool_calls."+strconv.Itoa(toInt(call["index"])), call, "Anthropic Messages")...)
			}
		}
	}
	return append(losses, reasoningFieldLosses(prefix, delta, "Anthropic Messages")...)
}

// AnthropicSSEToOpenAIChunksWithReport delegates stream conversion and reports
// usage fields that the existing Chat chunk adapter does not emit, and the
// thinking blocks it drops.
func AnthropicSSEToOpenAIChunksWithReport(lines func() (string, bool), model string, emit func(string)) Report {
	sawUsage := false
	var losses []Loss
	observed := func() (string, bool) {
		raw, ok := lines()
		if !ok {
			return raw, false
		}
		line := raw
		if len(line) >= len("data:") && line[:len("data:")] == "data:" {
			var event map[string]any
			if json.Unmarshal([]byte(line[len("data:"):]), &event) == nil && event["usage"] != nil {
				sawUsage = true
			}
		}
		// Parse the way the converter does, so the thinking blocks reported
		// are exactly the ones it skips.
		if data, ok := strings.CutPrefix(strings.TrimSpace(raw), "data:"); ok {
			var event map[string]any
			if json.Unmarshal([]byte(strings.TrimSpace(data)), &event) == nil && event["type"] == "content_block_start" {
				if block, _ := asMap(event["content_block"]); isThinkingBlock(block) {
					losses = append(losses, droppedReasoning("content."+strconv.Itoa(toInt(event["index"]))))
				}
			}
		}
		return raw, true
	}
	AnthropicSSEToOpenAIChunks(observed, model, emit)
	if sawUsage {
		losses = append(losses, material("usage", LossDropped, "Anthropic streaming usage is not emitted in Chat chunks"))
	}
	return NewReport(losses...)
}
