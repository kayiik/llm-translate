package realtime

func (s *Session) Server(raw []byte) (Output, error) {
	var out Output
	if s.unusable {
		return out, &Error{Code: "session_unusable", Param: "new session required", RequiresClose: true}
	}
	v, err := object(raw)
	if err != nil {
		return out, s.fatal("invalid upstream event")
	}
	if _, ok := v["setupComplete"]; ok {
		if !s.setupSent || s.ready {
			return out, s.fatal("unexpected setupComplete")
		}
		s.ready = true
		out.Downstream = append(out.Downstream, s.event("session.updated", map[string]any{"session": s.session()}))
	}
	if !s.ready {
		return out, s.fatal("upstream event before setupComplete")
	}
	if err := keys(v, "server", "setupComplete", "serverContent", "toolCall", "toolCallCancellation", "usageMetadata"); err != nil {
		return out, s.fatal(err.Error())
	}
	if _, ok := v["toolCallCancellation"]; ok {
		return out, s.fatal("toolCallCancellation: external tool/history reconciliation unavailable")
	}
	if x, ok := v["serverContent"]; ok {
		m := dict(x)
		if m == nil {
			return out, s.fatal("serverContent")
		}
		interrupted := m["interrupted"] == true
		if interrupted {
			if !s.cancelPending || !s.active {
				return out, s.fatal("unsolicited serverContent.interrupted")
			}
			s.interruptionEnding = true
		}
		if err := keys(m, "serverContent", "modelTurn", "turnComplete", "generationComplete", "interrupted", "inputTranscription", "outputTranscription"); err != nil {
			return out, s.fatal(err.Error())
		}
		if x, ok := m["inputTranscription"]; ok {
			_ = x
			return out, s.fatal("inputTranscription: unordered source item attribution unsupported; not requested in setup")
		}
		if x, ok := m["modelTurn"]; ok {
			turn := dict(x)
			if err := keys(turn, "modelTurn", "parts", "role"); err != nil {
				return out, s.fatal(err.Error())
			}
			a, yes := turn["parts"].([]any)
			if !yes {
				return out, s.fatal("modelTurn.parts")
			}
			for _, part := range a {
				p := dict(part)
				if err := keys(p, "modelTurn.part", "inlineData"); err != nil {
					return out, s.fatal(err.Error())
				}
				b := dict(p["inlineData"])
				if err := keys(b, "inlineData", "mimeType", "data"); err != nil {
					return out, s.fatal(err.Error())
				}
				if b["mimeType"] != "audio/pcm;rate=24000" {
					return out, s.fatal("output audio must be PCM16 mono 24kHz")
				}
				audio, err := pcm(b["data"])
				if err != nil {
					return out, s.fatal(err.Error())
				}
				if s.cancelPending || interrupted {
					continue
				} // fence late cancelled-generation audio
				if !s.active {
					return out, s.fatal("unsolicited output without response.create")
				}
				s.openAudio(&out)
				s.audioSeen = true
				out.Downstream = append(out.Downstream, s.event("response.output_audio.delta", s.audioFields(map[string]any{"delta": audio})))
			}
		}
		if x, ok := m["outputTranscription"]; ok {
			if !s.cancelPending && !interrupted {
				t := dict(x)
				value, yes := t["text"].(string)
				if !yes || !s.active {
					return out, s.fatal("outputTranscription without response/text")
				}
				s.openAudio(&out)
				s.transcriptSeen = true
				s.transcript += value
				out.Downstream = append(out.Downstream, s.event("response.output_audio_transcript.delta", s.audioFields(map[string]any{"delta": value})))
			}
		}
		if m["turnComplete"] == true {
			if s.interruptionEnding {
				s.finishStatus(&out, "cancelled")
				s.cancelPending = false
				s.interruptionEnding = false
			} else if s.active {
				s.cancelPending = false
				s.finish(&out)
			} else if len(s.calls) == 0 {
				return out, s.fatal("turnComplete without active response")
			}
		}
	}
	if x, ok := v["toolCall"]; ok {
		if s.cancelPending {
			return out, s.fatal("toolCall during pending cancellation; execution not authorized")
		}
		m := dict(x)
		if err := keys(m, "toolCall", "functionCalls"); err != nil {
			return out, s.fatal(err.Error())
		}
		a, yes := m["functionCalls"].([]any)
		if !yes || len(a) == 0 || !s.active || len(s.calls) > 0 {
			return out, s.fatal("unexpected toolCall")
		}
		// Validate the whole call batch before publishing any invocation IDs.
		calls := make([]function, 0, len(a))
		seen := map[string]bool{}
		for _, x := range a {
			c := dict(x)
			if err := keys(c, "functionCall", "id", "name", "args"); err != nil {
				return out, s.fatal(err.Error())
			}
			id, name := text(c["id"]), text(c["name"])
			if id == "" || !s.tools[name] || seen[id] || s.seenCalls[id] || dict(c["args"]) == nil {
				return out, s.fatal("toolCall id/name/args cannot be correlated")
			}
			seen[id] = true
			calls = append(calls, function{id: id, name: name, item: s.next("item")})
		}
		for i, c := range calls {
			s.seenCalls[c.id] = true
			args := string(frame(dict(a[i])["args"]))
			item := map[string]any{"id": c.item, "object": "realtime.item", "type": "function_call", "call_id": c.id, "name": c.name, "arguments": "", "status": "in_progress"}
			index := len(s.items)
			out.Downstream = append(out.Downstream, s.event("response.output_item.added", map[string]any{"response_id": s.responseID, "output_index": index, "item": item}))
			fields := map[string]any{"response_id": s.responseID, "item_id": c.item, "output_index": index, "call_id": c.id, "delta": args}
			out.Downstream = append(out.Downstream, s.event("response.function_call_arguments.delta", fields))
			out.Downstream = append(out.Downstream, s.event("response.function_call_arguments.done", map[string]any{"response_id": s.responseID, "item_id": c.item, "output_index": index, "call_id": c.id, "name": c.name, "arguments": args}))
			item["arguments"] = args
			item["status"] = "completed"
			s.items = append(s.items, item)
			s.lastItem = c.item
			out.Downstream = append(out.Downstream, s.event("response.output_item.done", map[string]any{"response_id": s.responseID, "output_index": index, "item": item}))
		}
		s.calls = calls
		// Live waits for toolResponse; expose completed function-call output so
		// an OpenAI client can submit results and explicitly create its next response.
		s.finish(&out)
	}
	// usageMetadata is deliberately NOT translated to token counts: its native
	// per-turn/cumulative scope and attribution have not been established here.
	return out, nil
}

func (s *Session) audioFields(fields map[string]any) map[string]any {
	fields["response_id"] = s.responseID
	fields["item_id"] = s.audioID
	fields["output_index"] = 0
	fields["content_index"] = 0
	return fields
}
func (s *Session) audioItem(status string) map[string]any {
	return map[string]any{"id": s.audioID, "object": "realtime.item", "type": "message", "role": "assistant", "status": status, "content": []any{map[string]any{"type": "audio", "transcript": s.transcript}}}
}
func (s *Session) openAudio(out *Output) {
	if s.audioID != "" {
		return
	}
	s.audioID = s.next("item")
	s.items = append(s.items, s.audioItem("in_progress"))
	s.lastItem = s.audioID
	out.Downstream = append(out.Downstream, s.event("response.output_item.added", map[string]any{"response_id": s.responseID, "output_index": 0, "item": map[string]any{"id": s.audioID, "object": "realtime.item", "type": "message", "role": "assistant", "status": "in_progress", "content": []any{}}}), s.event("response.content_part.added", s.audioFields(map[string]any{"part": map[string]any{"type": "audio", "transcript": ""}})))
}
func (s *Session) finish(out *Output) {
	s.finishStatus(out, "completed")
}
func (s *Session) finishStatus(out *Output, status string) {
	if s.audioID != "" {
		if s.audioSeen {
			out.Downstream = append(out.Downstream, s.event("response.output_audio.done", s.audioFields(map[string]any{})))
		}
		if s.transcriptSeen {
			out.Downstream = append(out.Downstream, s.event("response.output_audio_transcript.done", s.audioFields(map[string]any{"transcript": s.transcript})))
		}
		out.Downstream = append(out.Downstream, s.event("response.content_part.done", s.audioFields(map[string]any{"part": map[string]any{"type": "audio", "transcript": s.transcript}})))
		itemStatus := "completed"
		if status == "cancelled" {
			itemStatus = "incomplete"
		}
		item := s.audioItem(itemStatus)
		// Audio can precede function calls in the same response. It occupies
		// index zero; tool indexes are assigned after reserving that slot.
		if len(s.items) == 0 {
			s.items = append(s.items, item)
		} else {
			s.items[0] = item
		}
		out.Downstream = append(out.Downstream, s.event("response.output_item.done", map[string]any{"response_id": s.responseID, "output_index": 0, "item": item}))
	}
	items := s.items
	if items == nil {
		items = []any{}
	}
	response := map[string]any{"id": s.responseID, "object": "realtime.response", "status": status, "output": items}
	if status == "cancelled" {
		response["status_details"] = map[string]any{"type": "cancelled", "reason": "client_cancelled"}
	}
	out.Downstream = append(out.Downstream, s.event("response.done", map[string]any{"response": response}))
	s.active = false
}
