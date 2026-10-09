package translate

import (
	"encoding/json"
	"strings"
	"testing"
)

// lossAt returns the loss reported at path, if any.
func lossAt(report Report, path string) (Loss, bool) {
	for _, loss := range report.Losses {
		if loss.Path == path {
			return loss, true
		}
	}
	return Loss{}, false
}

func requireLoss(t *testing.T, report Report, path string, severity LossSeverity) {
	t.Helper()
	loss, ok := lossAt(report, path)
	if !ok || loss.Severity != severity || loss.Class != LossDropped {
		t.Fatalf("want %s dropped loss at %s, got %+v in %+v", severity, path, loss, report.Losses)
	}
}

func TestVendorFieldsAreCarriedOrReported(t *testing.T) {
	messages := decodeJSON[[]map[string]any](t, vendorChatMessages)
	signature := "messages.2.tool_calls.0.extra_content.google.thought_signature"

	t.Run("Chat to Responses reports every field", func(t *testing.T) {
		report := ChatToResponsesWithReport("m", messages, nil, false).Report
		requireLoss(t, report, signature, LossMaterial)
		requireLoss(t, report, "messages.2.reasoning_details", LossAdvisory)
		requireLoss(t, report, "messages.0.content.0.cache_control", LossAdvisory)
		requireLoss(t, report, "messages.3.content.0.cache_control", LossAdvisory)
		if _, ok := lossAt(report, "messages.2.tool_calls.1.extra_content.google.thought_signature"); ok {
			t.Fatal("an empty thought signature carries nothing and must not be reported")
		}
	})

	t.Run("Chat to Anthropic carries text cache_control", func(t *testing.T) {
		result := OpenAIMessagesToAnthropicWithReport(messages)
		requireLoss(t, result.Report, signature, LossMaterial)
		requireLoss(t, result.Report, "messages.2.reasoning_details", LossAdvisory)
		requireLoss(t, result.Report, "messages.1.content.1.cache_control", LossAdvisory)
		for _, carried := range []string{"messages.0.content.0.cache_control", "messages.1.content.0.cache_control", "messages.3.content.0.cache_control"} {
			if _, ok := lossAt(result.Report, carried); ok {
				t.Fatalf("carried cache_control reported as lost: %s", carried)
			}
		}
		system, _ := json.Marshal(result.Value.SystemBlocks)
		if !strings.Contains(string(system), `"cache_control":{"ttl":"1h","type":"ephemeral"}`) || result.Value.System != "cached policy\ntail" {
			t.Fatalf("system breakpoint not carried: %s / %q", system, result.Value.System)
		}
		body, _ := json.Marshal(result.Value.Messages)
		if strings.Count(string(body), `"cache_control"`) != 2 {
			t.Fatalf("message breakpoints not carried: %s", body)
		}
	})

	t.Run("Anthropic to Chat reports cache_control and reasoning", func(t *testing.T) {
		report := AnthropicRequestToOpenAIWithReport(decodeJSON[map[string]any](t, `{"model":"m","system":[{"type":"text","text":"s","cache_control":{"type":"ephemeral"}}],
			"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"t","signature":"s"},{"type":"text","text":"a","cache_control":{"type":"ephemeral"}}]}]}`)).Report
		for _, path := range []string{"system.0.cache_control", "messages.0.content.1.cache_control"} {
			if _, ok := lossAt(report, path); !ok {
				t.Fatalf("cache_control not reported at %s: %+v", path, report.Losses)
			}
		}
		requireLoss(t, report, "messages.0.content.0.reasoning", LossAdvisory)
	})

	t.Run("responses report reasoning at a reasoning path", func(t *testing.T) {
		report := ResponsesToChatWithReport("m", decodeJSON[map[string]any](t, `{"output":[{"type":"message","content":[]},{"type":"reasoning"}]}`)).Report
		requireLoss(t, report, "output.1.reasoning", LossAdvisory)
		report = AnthropicResponseToOpenAIWithReport(decodeJSON[map[string]any](t, `{"content":[{"type":"redacted_thinking","data":"x"}]}`), "m").Report
		requireLoss(t, report, "content.0.reasoning", LossAdvisory)
		chat := decodeJSON[map[string]any](t, vendorChatResponse)
		for name, report := range map[string]Report{
			"to Anthropic": OpenAIResponseToAnthropicWithReport(chat, "m").Report,
			"to Responses": ChatResponseToResponsesWithReport("m", chat).Report,
		} {
			t.Run(name, func(t *testing.T) {
				requireLoss(t, report, "choices.0.message.tool_calls.0.extra_content.google.thought_signature", LossMaterial)
				requireLoss(t, report, "choices.0.message.reasoning_details", LossAdvisory)
			})
		}
	})

	t.Run("streams report per delta and per block", func(t *testing.T) {
		report := OpenAIStreamToAnthropicSSEWithReport(pullItems([]string{
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":2,"extra_content":{"google":{"thought_signature":"sig"}}}],"reasoning_details":[{"type":"x"}]}}]}`,
		}), "m", func(string) {})
		requireLoss(t, report, "choices.0.delta.tool_calls.2.extra_content.google.thought_signature", LossMaterial)
		requireLoss(t, report, "choices.0.delta.reasoning_details", LossAdvisory)
		report = AnthropicSSEToOpenAIChunksWithReport(pullItems([]string{
			` data: {"type":"content_block_start","index":3,"content_block":{"type":"thinking","thinking":""}}`,
		}), "m", func(string) {})
		requireLoss(t, report, "content.3.reasoning", LossAdvisory)
	})
}
