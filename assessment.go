package translate

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// ConversionID identifies an existing request edge throughout its response.
type ConversionID string

const (
	ConversionChatToResponses ConversionID = "chat-to-responses"
	ConversionResponsesToChat ConversionID = "responses-to-chat"
	ConversionChatToMessages  ConversionID = "chat-to-messages"
	ConversionMessagesToChat  ConversionID = "messages-to-chat"
)

type ObservationPhase string

const (
	PhaseRequest     ObservationPhase = "request"
	PhaseResponse    ObservationPhase = "response"
	PhaseStreamEvent ObservationPhase = "stream-event"
	PhaseFinal       ObservationPhase = "final"
)

type Feasibility string

const (
	FeasibilitySupported   Feasibility = "supported"
	FeasibilityUnsupported Feasibility = "unsupported"
	FeasibilityUnknown     Feasibility = "unknown"
)

type FidelityFeature string

const (
	FidelityContent FidelityFeature = "content"
	FidelityTools   FidelityFeature = "tools"
	FidelitySchema  FidelityFeature = "schema"
	FidelityBilling FidelityFeature = "billing"
	FidelityReplay  FidelityFeature = "replay"
)

type FidelityRequirements struct {
	Content, Tools, Schema, Billing, Replay bool
}

// Assessment is conversion evidence, not admission or target-model capability.
// UnknownPaths are unevaluated source semantics; a loss-free legacy conversion
// alone is not evidence that those semantics were preserved.
type Assessment struct {
	Conversion   ConversionID
	Phase        ObservationPhase
	Feasibility  Feasibility
	Requirements FidelityRequirements
	Evaluated    []FidelityFeature
	UnknownPaths []string
	Report       Report
}

func InferFidelityRequirements(payload map[string]any) FidelityRequirements {
	r := FidelityRequirements{Content: true}
	if payload["response_format"] != nil {
		r.Schema = true
	}
	if text, ok := asMap(payload["text"]); ok && text["format"] != nil {
		r.Schema = true
	}
	for _, raw := range mustList(payload["tools"]) {
		tool, _ := asMap(raw)
		fn := tool
		if nested, ok := asMap(tool["function"]); ok {
			fn = nested
		}
		if fn["parameters"] != nil || fn["strict"] != nil {
			r.Schema = true
		}
	}
	walkFidelity(payload, "", func(path, key string, value any, object map[string]any) {
		if !present(value) {
			return
		}
		switch key {
		case "type":
			if value == "tool_use" || value == "tool_result" || value == "function_call" || value == "function_call_output" {
				r.Tools = true
			}
		case "tools", "tool_calls", "tool_choice", "function_call":
			r.Tools = true
		case "response_format", "text", "input_schema", "parameters", "strict":
			if key != "text" || path == "text" {
				r.Schema = true
			}
		case "service_tier", "store", "prompt_cache_key", "prompt_cache_options", "cache_control":
			r.Billing = true
		}
		if replayField(key, object) {
			r.Replay = true
		}
	})
	return r
}

func AssessConversion(id ConversionID, phase ObservationPhase, payload map[string]any, requirements FidelityRequirements) Assessment {
	inferred := InferFidelityRequirements(payload)
	r := FidelityRequirements{Content: requirements.Content || inferred.Content, Tools: requirements.Tools || inferred.Tools, Schema: requirements.Schema || inferred.Schema, Billing: requirements.Billing || inferred.Billing, Replay: requirements.Replay || inferred.Replay}
	a := Assessment{Conversion: id, Phase: phase, Feasibility: FeasibilityUnknown, Requirements: r}
	if payload == nil || (phase != PhaseRequest && phase != PhaseResponse) {
		a.UnknownPaths = []string{"$"}
		return a
	}
	var losses []Loss
	var known string
	if phase == PhaseRequest {
		switch id {
		case ConversionChatToResponses, ConversionChatToMessages:
			messages, err := assessmentMessages(payload["messages"])
			if err != nil {
				losses = append(losses, material("messages", LossUnsupported, err.Error()))
			}
			if id == ConversionChatToResponses {
				kw := map[string]any{}
				for k, v := range payload {
					if k != "model" && k != "messages" && k != "stream" {
						kw[k] = v
					}
				}
				losses = append(losses, ChatToResponsesWithReport("", messages, kw, false).Report.Losses...)
				known = "model messages stream max_tokens max_completion_tokens reasoning_effort temperature top_p tools tool_choice metadata response_format n stop seed stream_options parallel_tool_calls"
			} else {
				losses = append(losses, OpenAIMessagesToAnthropicWithReport(messages).Report.Losses...)
				known = "model messages stream max_tokens max_completion_tokens temperature top_p top_k tools tool_choice stop user stream_options parallel_tool_calls response_format"
				// This edge's message-list helper has no top-level schema/billing
				// mapping. Admission must not mistake that omission for support.
				for _, k := range []string{"response_format", "service_tier", "store", "prompt_cache_key", "prompt_cache_options", "reasoning_effort"} {
					if present(payload[k]) {
						losses = append(losses, material(k, LossDropped, "required control is not emitted by the Messages conversion"))
					}
				}
				if n, ok := payload["n"]; ok && n != nil && toInt(n) != 1 {
					losses = append(losses, material("n", LossUnsupported, "Messages supports one generation"))
				}
			}
			losses = append(losses, assessmentChatTools(id, payload)...)
		case ConversionResponsesToChat:
			result, _ := ResponsesRequestToChatWithReport(payload)
			losses = append(losses, result.Report.Losses...)
			known = "model input instructions stream max_output_tokens reasoning temperature top_p tools tool_choice metadata store background force_api_support previous_response_id conversation client_metadata include parallel_tool_calls prompt_cache_key text"
			for _, k := range []string{"prompt_cache_key", "client_metadata"} {
				if present(payload[k]) {
					losses = append(losses, material(k, LossDropped, "accepted source control is not emitted by the Chat conversion"))
				}
			}
		case ConversionMessagesToChat:
			losses = append(losses, AnthropicRequestToOpenAIWithReport(payload).Report.Losses...)
			known = "model messages max_tokens system tools tool_choice stream temperature top_p stop_sequences metadata thinking output_config _llmgw_preamble"
		default:
			a.UnknownPaths = []string{"$"}
			return a
		}
	} else {
		switch id {
		case ConversionChatToResponses:
			losses = append(losses, ResponsesToChatWithReport("", payload).Report.Losses...)
			known = "id object created_at model output output_text status error incomplete_details usage background instructions max_output_tokens max_tool_calls parallel_tool_calls previous_response_id reasoning service_tier store temperature text tool_choice tools top_logprobs top_p truncation user metadata"
			if _, ok := asList(payload["output"]); !ok {
				if _, text := payload["output_text"].(string); !text {
					a.UnknownPaths = append(a.UnknownPaths, "output")
				}
			}
		case ConversionResponsesToChat:
			losses = append(losses, ChatResponseToResponsesWithReport("", payload).Report.Losses...)
			known = "id object created model choices usage service_tier system_fingerprint"
			if _, ok := asList(payload["choices"]); !ok {
				a.UnknownPaths = append(a.UnknownPaths, "choices")
			}
		case ConversionChatToMessages:
			losses = append(losses, AnthropicResponseToOpenAIWithReport(payload, "").Report.Losses...)
			known = "id type role model content stop_reason stop_sequence usage"
			if _, ok := asList(payload["content"]); !ok {
				a.UnknownPaths = append(a.UnknownPaths, "content")
			}
		case ConversionMessagesToChat:
			losses = append(losses, OpenAIResponseToAnthropicWithReport(payload, "").Report.Losses...)
			known = "id object created model choices usage service_tier system_fingerprint"
			if _, ok := asList(payload["choices"]); !ok {
				a.UnknownPaths = append(a.UnknownPaths, "choices")
			}
		default:
			a.UnknownPaths = []string{"$"}
			return a
		}
	}
	// The observed Chat routing annotation is informational, not destination
	// response content. Other shapes remain unassessed.
	if phase == PhaseResponse && (id == ConversionResponsesToChat || id == ConversionMessagesToChat) {
		if _, ok := payload["provider"].(string); ok {
			known += " provider"
		}
	}
	for k, v := range payload {
		if v != nil && !assessmentKnown(known, k) {
			a.UnknownPaths = append(a.UnknownPaths, k)
		}
	}
	a.UnknownPaths = append(a.UnknownPaths, assessmentNestedUnknowns(payload)...)
	if phase == PhaseRequest && (id == ConversionChatToResponses || id == ConversionResponsesToChat) {
		_, unknown := structuredOutputFindings(id, payload)
		a.UnknownPaths = append(a.UnknownPaths, unknown...)
	}
	if phase == PhaseRequest {
		for _, f := range []struct {
			required, inferred bool
			feature            FidelityFeature
		}{{requirements.Replay, inferred.Replay, FidelityReplay}, {requirements.Schema, inferred.Schema, FidelitySchema}, {requirements.Billing, inferred.Billing, FidelityBilling}} {
			if f.required && !f.inferred {
				a.UnknownPaths = append(a.UnknownPaths, "$requirements."+string(f.feature))
			}
		}
	}
	if raw, exists := payload["tools"]; exists && raw != nil && (phase != PhaseRequest || (id != ConversionChatToResponses && id != ConversionResponsesToChat)) {
		if _, ok := asList(raw); !ok {
			losses = append(losses, material("tools", LossUnsupported, "tools must be an evaluated list"))
		}
	}
	if phase == PhaseResponse && (id == ConversionResponsesToChat || id == ConversionMessagesToChat) {
		for i, raw := range mustList(payload["choices"]) {
			choice, _ := asMap(raw)
			message, _ := asMap(choice["message"])
			if value := message["content"]; value != nil {
				if _, ok := value.(string); !ok {
					losses = append(losses, material(fmt.Sprintf("choices.%d.message.content", i), LossDropped, "this response conversion emits string content only"))
				}
			}
		}
	}
	if id == ConversionChatToResponses && phase == PhaseRequest {
		for i, raw := range mustList(payload["messages"]) {
			message, _ := asMap(raw)
			for j, rawPart := range mustList(message["content"]) {
				part, _ := asMap(rawPart)
				if part["type"] != "image_url" && part["type"] != "input_image" {
					continue
				}
				path := fmt.Sprintf("messages.%d.content.%d", i, j)
				for _, key := range []string{"file_id", "source"} {
					if present(part[key]) {
						losses = append(losses, material(path+"."+key, LossDropped, "image source form is not mapped by this conversion"))
					}
				}
				url := part["image_url"]
				if nested, ok := asMap(url); ok {
					url = nested["url"]
				}
				if text, ok := url.(string); !ok || text == "" {
					losses = append(losses, material(path+".image_url", LossUnsupported, "mapped image requires a nonempty URL/data string"))
				}
			}
		}
	}
	for i := range losses {
		if r.Billing && (strings.Contains(losses[i].Path, "cache_control") || losses[i].Path == "service_tier" || losses[i].Path == "store" || strings.HasPrefix(losses[i].Path, "prompt_cache")) {
			losses[i].Severity = LossMaterial
		}
		if phase == PhaseRequest && r.Tools && id == ConversionChatToResponses && losses[i].Class == LossApproximated && strings.Contains(losses[i].Path, ".tool_calls.") {
			losses[i].Severity = LossMaterial
		}
	}
	// A display-only thinking block may be omitted, but signed/opaque state
	// and continuation metadata require exact paths even with an empty summary.
	if r.Replay {
		walkFidelity(payload, "", func(path, key string, value any, object map[string]any) {
			if !present(value) {
				return
			}
			if replayField(key, object) || (key == "id" && object["type"] == "reasoning") {
				losses = append(losses, material(path, LossDropped, "opaque continuation state is not preserved by this conversion"))
			}
		})
	}
	if !r.Replay && phase == PhaseResponse && id == ConversionChatToMessages {
		// Preserve the existing terminal-summary interpretation without
		// suppressing signed/redacted state when replay is required.
		for i := range losses {
			for j, raw := range mustList(payload["content"]) {
				block, _ := asMap(raw)
				if isThinkingBlock(block) && losses[i].Path == fmt.Sprintf("content.%d", j) {
					losses[i].Severity = LossAdvisory
				}
			}
		}
	}
	a.Report = NewReport(losses...)
	a.Evaluated = []FidelityFeature{FidelityContent}
	for _, f := range []struct {
		required bool
		feature  FidelityFeature
	}{{r.Tools, FidelityTools}, {r.Schema, FidelitySchema}, {r.Billing, FidelityBilling}, {r.Replay, FidelityReplay}} {
		if f.required {
			a.Evaluated = append(a.Evaluated, f.feature)
		}
	}
	sort.Strings(a.UnknownPaths)
	a.UnknownPaths = uniquePaths(a.UnknownPaths)
	a.Feasibility = FeasibilitySupported
	if len(a.UnknownPaths) > 0 {
		a.Feasibility = FeasibilityUnknown
	}
	for _, l := range a.Report.Losses {
		if l.Severity == LossMaterial || l.Class == LossUnsupported {
			a.Feasibility = FeasibilityUnsupported
		}
	}
	return a
}

func assessmentMessages(value any) ([]map[string]any, error) {
	if messages, ok := value.([]map[string]any); ok {
		return messages, nil
	}
	list, ok := asList(value)
	if !ok {
		return nil, fmt.Errorf("messages must be a list")
	}
	messages := make([]map[string]any, 0, len(list))
	for _, raw := range list {
		m, ok := asMap(raw)
		if !ok {
			return nil, fmt.Errorf("each message must be an object")
		}
		messages = append(messages, m)
	}
	return messages, nil
}

func mustList(v any) []any { list, _ := asList(v); return list }

func assessmentKnown(known, key string) bool {
	// Keys are fixed protocol field names, never caller-supplied patterns.
	for _, field := range splitFields(known) {
		if field == key {
			return true
		}
	}
	return false
}

func splitFields(s string) []string { return strings.Fields(s) }

func replayField(key string, object map[string]any) bool {
	switch key {
	case "encrypted_content", "signature", "thought_signature", "phase", "previous_response_id", "conversation", "continuation_token":
		return true
	case "data":
		kind, _ := object["type"].(string)
		return kind == "redacted_thinking" || kind == "reasoning.encrypted"
	}
	return false
}

func walkFidelity(value any, prefix string, visit func(path, key string, value any, object map[string]any)) {
	switch v := value.(type) {
	case map[string]any:
		for key, value := range v {
			path := key
			if prefix != "" {
				path = prefix + "." + key
			}
			visit(path, key, value, v)
			// Schema property names and caller metadata are data, not wire
			// controls or continuation-state fields.
			if key != "schema" && key != "parameters" && key != "input_schema" && key != "metadata" && key != "input" {
				walkFidelity(value, path, visit)
			} else if key == "input" && prefix == "" {
				walkFidelity(value, path, visit)
			}
		}
	case []any:
		for i, value := range v {
			walkFidelity(value, prefix+"."+strconv.Itoa(i), visit)
		}
	case []map[string]any:
		for i, value := range v {
			walkFidelity(value, prefix+"."+strconv.Itoa(i), visit)
		}
	}
}

func assessmentChatTools(id ConversionID, payload map[string]any) []Loss {
	var losses []Loss
	for i, raw := range mustList(payload["messages"]) {
		message, _ := asMap(raw)
		if rawCalls, exists := message["tool_calls"]; exists && rawCalls != nil {
			if _, ok := asList(rawCalls); !ok {
				losses = append(losses, material(fmt.Sprintf("messages.%d.tool_calls", i), LossUnsupported, "tool calls must be a list"))
			}
		}
		for j, rawCall := range mustList(message["tool_calls"]) {
			call, _ := asMap(rawCall)
			if value, ok := call["id"].(string); !ok || value == "" {
				losses = append(losses, material(fmt.Sprintf("messages.%d.tool_calls.%d.id", i, j), LossUnsupported, "missing tool call identity cannot be repaired by compatibility permission"))
			}
		}
	}
	for i, raw := range mustList(payload["tools"]) {
		tool, ok := asMap(raw)
		path := fmt.Sprintf("tools.%d", i)
		if !ok || tool["type"] != "function" {
			losses = append(losses, material(path, LossUnsupported, "only function tools are evaluated for this conversion"))
			continue
		}
		fn, _ := asMap(tool["function"])
		if id == ConversionChatToMessages && present(fn["strict"]) {
			losses = append(losses, material(path+".function.strict", LossDropped, "strict tool enforcement is not mapped by this conversion"))
		}
	}
	return losses
}

func assessmentNestedUnknowns(payload map[string]any) []string {
	var unknown []string
	check := func(object map[string]any, path, known string) {
		for key, value := range object {
			if value != nil && !assessmentKnown(known, key) {
				unknown = append(unknown, path+"."+key)
			}
		}
	}
	for _, root := range []string{"messages", "input", "output"} {
		items := mustList(payload[root])
		objectInput := false
		if object, ok := asMap(payload[root]); ok {
			items = []any{object}
			objectInput = true
		}
		for i, raw := range items {
			item, _ := asMap(raw)
			path := fmt.Sprintf("%s.%d", root, i)
			if objectInput {
				path = root
			}
			check(item, path, "type id role content tool_calls tool_call_id call_id arguments output status summary encrypted_content phase reasoning_content reasoning_details name")
			if root == "messages" && item["name"] != nil {
				unknown = append(unknown, path+".name")
			}
			for j, rawCall := range mustList(item["tool_calls"]) {
				call, _ := asMap(rawCall)
				callPath := fmt.Sprintf("%s.tool_calls.%d", path, j)
				check(call, callPath, "id index type function extra_content thought_signature")
				if fn, ok := asMap(call["function"]); ok {
					check(fn, callPath+".function", "name arguments thought_signature")
				}
			}
			for j, rawPart := range mustList(item["content"]) {
				part, _ := asMap(rawPart)
				known := partFields(part)
				if root == "output" && part["type"] == "output_text" {
					known += " annotations"
				}
				check(part, fmt.Sprintf("%s.content.%d", path, j), known)
				if nested, ok := asMap(part["image_url"]); ok {
					check(nested, fmt.Sprintf("%s.content.%d.image_url", path, j), "url detail")
				}
				if nested, ok := asMap(part["source"]); ok {
					check(nested, fmt.Sprintf("%s.content.%d.source", path, j), "type media_type data url")
				}
				if nested, ok := asMap(part["file"]); ok {
					check(nested, fmt.Sprintf("%s.content.%d.file", path, j), "filename file_data file_id")
				}
			}
		}
	}
	for i, raw := range mustList(payload["tools"]) {
		tool, _ := asMap(raw)
		path := fmt.Sprintf("tools.%d", i)
		check(tool, path, "type function name description parameters strict input_schema cache_control")
		if fn, ok := asMap(tool["function"]); ok {
			check(fn, path+".function", "name description parameters strict")
		}
	}
	for i, raw := range mustList(payload["content"]) {
		part, _ := asMap(raw)
		check(part, fmt.Sprintf("content.%d", i), partFields(part))
	}
	for i, raw := range mustList(payload["choices"]) {
		choice, _ := asMap(raw)
		path := fmt.Sprintf("choices.%d", i)
		known := "index message finish_reason"
		// An opaque native reason cannot supply missing/unknown standard
		// completion semantics. Only an annotation beside a known reason is safe.
		if _, ok := choice["native_finish_reason"].(string); ok {
			switch choice["finish_reason"] {
			case "stop", "length", "tool_calls":
				known += " native_finish_reason"
			}
		}
		check(choice, path, known)
		if message, ok := asMap(choice["message"]); ok {
			known := "role content refusal tool_calls reasoning_content reasoning_details"
			if _, ok := message["reasoning"].(string); ok {
				known += " reasoning"
			}
			check(message, path+".message", known)
		}
	}
	return unknown
}

func partFields(part map[string]any) string {
	switch part["type"] {
	case "text", "input_text", "output_text":
		return "type text cache_control"
	case "image_url", "input_image":
		return "type image_url detail cache_control file_id source"
	case "image", "document":
		return "type source cache_control"
	case "file":
		return "type file cache_control"
	case "input_file":
		return "type filename file_data file_id file_url cache_control"
	case "thinking":
		return "type thinking signature"
	case "redacted_thinking":
		return "type data"
	case "tool_use":
		return "type id name input cache_control"
	case "tool_result":
		return "type tool_use_id content is_error cache_control"
	case "refusal":
		return "type refusal"
	}
	return "type"
}

func uniquePaths(paths []string) []string {
	if len(paths) < 2 {
		return paths
	}
	out := paths[:1]
	for _, path := range paths[1:] {
		if path != out[len(out)-1] {
			out = append(out, path)
		}
	}
	return out
}
