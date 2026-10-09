package translate

import "strconv"

// Vendor fields travel inside messages, where the conversions historically
// dropped them without a trace. These helpers report each one at its exact
// source path so a LossPolicy can target it:
//
//   - A dropped Gemini thought signature is material: without it the next
//     turn of a multi-turn tool call fails or loses the model's reasoning.
//     Providers store it at tool_calls.N.extra_content.google.thought_signature,
//     tool_calls.N.function.thought_signature or tool_calls.N.thought_signature;
//     each location that carries one is reported.
//   - Dropped reasoning_details, reasoning_content and cache_control are
//     advisory: the answer is intact, and a product that depends on them can
//     still reject the loss by path.
//   - Dropped reasoning items and blocks are reported at <item path>.reasoning
//     so that a policy can match every reasoning loss with **.reasoning.

// present reports whether a vendor value carries anything. Null, "" and
// empty arrays or objects carry nothing, so dropping them loses nothing.
func present(value any) bool {
	switch v := value.(type) {
	case nil:
		return false
	case string:
		return v != ""
	case []any:
		return len(v) > 0
	case map[string]any:
		return len(v) > 0
	}
	return true
}

// thoughtSignatureLosses reports every thought signature one dropped tool
// call carries. callPath is the path of the call. The locations are listed
// in the order wire.ChatToolCall.ThoughtSignature reads them.
func thoughtSignatureLosses(callPath string, call map[string]any, target string) []Loss {
	extra, _ := asMap(call["extra_content"])
	google, _ := asMap(extra["google"])
	function, _ := asMap(call["function"])
	var losses []Loss
	for _, location := range []struct {
		path  string
		value any
	}{
		{"extra_content.google.thought_signature", google["thought_signature"]},
		{"function.thought_signature", function["thought_signature"]},
		{"thought_signature", call["thought_signature"]},
	} {
		if present(location.value) {
			losses = append(losses, material(callPath+"."+location.path, LossDropped, "Gemini thought signature is not representable by "+target+", so the next turn cannot replay it"))
		}
	}
	return losses
}

func droppedReasoningDetails(path, target string) Loss {
	return advisory(path, LossDropped, "reasoning_details are not representable by "+target)
}

func droppedReasoningContent(path, target string) Loss {
	return advisory(path, LossDropped, "reasoning_content is not representable by "+target)
}

func droppedCacheControl(path, target string) Loss {
	return advisory(path, LossDropped, "cache_control is not representable by "+target)
}

func droppedReasoning(path string) Loss {
	return advisory(path+".reasoning", LossDropped, "reasoning is not emitted by Chat")
}

func isThinkingBlock(block map[string]any) bool {
	return block["type"] == "thinking" || block["type"] == "redacted_thinking"
}

// chatMessageVendorLosses reports the thought signatures, reasoning_details
// and reasoning_content that a conversion of one Chat message to target
// drops. prefix is the message path, such as messages.3 or choices.0.message.
func chatMessageVendorLosses(prefix string, message map[string]any, target string) []Loss {
	var losses []Loss
	if calls, ok := asList(message["tool_calls"]); ok {
		for j, raw := range calls {
			if call, ok := asMap(raw); ok {
				losses = append(losses, thoughtSignatureLosses(prefix+".tool_calls."+strconv.Itoa(j), call, target)...)
			}
		}
	}
	return append(losses, reasoningFieldLosses(prefix, message, target)...)
}

// reasoningFieldLosses reports the reasoning fields of one Chat message or
// delta that a conversion to target drops.
func reasoningFieldLosses(prefix string, message map[string]any, target string) []Loss {
	var losses []Loss
	if present(message["reasoning_details"]) {
		losses = append(losses, droppedReasoningDetails(prefix+".reasoning_details", target))
	}
	if present(message["reasoning_content"]) {
		losses = append(losses, droppedReasoningContent(prefix+".reasoning_content", target))
	}
	return losses
}
