package translate

import "testing"

func TestAlternateSignaturesAndReasoningContentAreReported(t *testing.T) {
	messages := decodeJSON[[]map[string]any](t, altChatMessages)
	chat := decodeJSON[map[string]any](t, altChatResponse)
	requests := map[string]Report{
		"Chat to Responses": ChatToResponsesWithReport("m", messages, nil, false).Report,
		"Chat to Anthropic": OpenAIMessagesToAnthropicWithReport(messages).Report,
	}
	for name, report := range requests {
		t.Run(name, func(t *testing.T) {
			requireLoss(t, report, "messages.1.tool_calls.0.function.thought_signature", LossMaterial)
			requireLoss(t, report, "messages.1.tool_calls.1.thought_signature", LossMaterial)
			requireLoss(t, report, "messages.1.tool_calls.2.extra_content.google.thought_signature", LossMaterial)
			requireLoss(t, report, "messages.1.reasoning_content", LossAdvisory)
			for _, empty := range []string{"messages.1.tool_calls.2.function.thought_signature", "messages.1.tool_calls.2.thought_signature", "messages.5.reasoning_content"} {
				if _, ok := lossAt(report, empty); ok {
					t.Fatalf("an empty value carries nothing and must not be reported: %s", empty)
				}
			}
		})
	}
	responses := map[string]Report{
		"response to Anthropic": OpenAIResponseToAnthropicWithReport(chat, "m").Report,
		"response to Responses": ChatResponseToResponsesWithReport("m", chat).Report,
	}
	for name, report := range responses {
		t.Run(name, func(t *testing.T) {
			requireLoss(t, report, "choices.0.message.tool_calls.0.function.thought_signature", LossMaterial)
			requireLoss(t, report, "choices.0.message.tool_calls.0.thought_signature", LossMaterial)
			requireLoss(t, report, "choices.0.message.reasoning_content", LossAdvisory)
		})
	}
	t.Run("stream deltas", func(t *testing.T) {
		report := OpenAIStreamToAnthropicSSEWithReport(pullItems([]string{
			`{"choices":[{"index":1,"delta":{"reasoning_content":"hmm","tool_calls":[{"index":3,"function":{"thought_signature":"f"},"thought_signature":"c"}]}}]}`,
		}), "m", func(string) {})
		requireLoss(t, report, "choices.1.delta.tool_calls.3.function.thought_signature", LossMaterial)
		requireLoss(t, report, "choices.1.delta.tool_calls.3.thought_signature", LossMaterial)
		requireLoss(t, report, "choices.1.delta.reasoning_content", LossAdvisory)
	})
}
