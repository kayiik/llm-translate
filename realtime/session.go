// Package realtime maps a bounded OpenAI GA audio session to Gemini Live
// v1beta. It owns no transport, audio playback, tool execution or billing.
// Callers must serialize all calls, forward each output in order, and close
// both sides on an Error with RequiresClose. Continued conversation after an
// acknowledged interruption uses server-delivered context, NOT accurate heard
// history. The transport must stop/flush playback on the cancelled response.
package realtime

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync/atomic"
)

type Output struct {
	Upstream   [][]byte
	Downstream [][]byte
}

// Error describes a refused event, not an invented upstream acknowledgement.
// RequiresClose means continuing this conversation would be unsafe.
type Error struct {
	Code          string
	Param         string
	RequiresClose bool
}

func (e *Error) Error() string { return e.Code + ": " + e.Param }

type function struct {
	name, id, item string
	result         any
	returned       bool
}

type Session struct {
	model, id                                        string
	seq                                              uint64
	started, setupSent, ready, unusable              bool
	config                                           map[string]any
	tools                                            map[string]bool
	inputOpen, committed                             bool
	inputHasAudio, cancelPending, interruptionEnding bool
	inputID, lastItem                                string
	responseID, audioID                              string
	active, audioSeen, transcriptSeen                bool
	transcript                                       string
	items                                            []any
	calls                                            []function
	seenCalls                                        map[string]bool
}

var sessionSequence atomic.Uint64

func New(model string) *Session {
	return &Session{model: model, id: fmt.Sprintf("sess_bridge_%d", sessionSequence.Add(1)), tools: map[string]bool{}, seenCalls: map[string]bool{}}
}

func (s *Session) next(prefix string) string {
	s.seq++
	return fmt.Sprintf("%s_%s_%d", prefix, s.id, s.seq)
}
func frame(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}
func (s *Session) event(kind string, fields map[string]any) []byte {
	if fields == nil {
		fields = map[string]any{}
	}
	fields["type"] = kind
	fields["event_id"] = s.next("event")
	return frame(fields)
}
func (s *Session) session() map[string]any {
	v := map[string]any{"id": s.id, "object": "realtime.session", "type": "realtime", "model": s.model}
	for k, x := range s.config {
		v[k] = x
	}
	return v
}
func (s *Session) Start() [][]byte {
	if s.started {
		return nil
	}
	s.started = true
	return [][]byte{s.event("session.created", map[string]any{"session": s.session()})}
}
func object(raw []byte) (map[string]any, error) {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	var v map[string]any
	if err := d.Decode(&v); err != nil || v == nil {
		return nil, &Error{Code: "invalid_json", Param: "event"}
	}
	var extra any
	if d.Decode(&extra) != io.EOF {
		return nil, &Error{Code: "invalid_json", Param: "trailing data"}
	}
	return v, nil
}
func dict(v any) map[string]any { m, _ := v.(map[string]any); return m }
func text(v any) string         { s, _ := v.(string); return s }
func reject(path string) error  { return &Error{Code: "unsupported_semantic", Param: path} }
func keys(m map[string]any, path string, allowed ...string) error {
	for k := range m {
		ok := false
		for _, a := range allowed {
			if k == a {
				ok = true
				break
			}
		}
		if !ok {
			return reject(path + "." + k)
		}
	}
	return nil
}
func pcm(v any) (string, error) {
	s, ok := v.(string)
	if !ok || s == "" || len(s) > 6*1024*1024 {
		return "", &Error{Code: "invalid_audio", Param: "audio"}
	}
	b, err := base64.StdEncoding.Strict().DecodeString(s)
	if err != nil || len(b) == 0 || len(b)%2 != 0 {
		return "", &Error{Code: "invalid_audio", Param: "PCM16 requires complete samples"}
	}
	return s, nil
}
func (s *Session) fatal(path string) error {
	s.unusable = true
	return &Error{Code: "unsupported_semantic", Param: path, RequiresClose: true}
}

func (s *Session) Client(raw []byte) (Output, error) {
	var out Output
	if s.unusable {
		return out, &Error{Code: "session_unusable", Param: "new session required", RequiresClose: true}
	}
	v, err := object(raw)
	if err != nil {
		return out, err
	}
	kind := text(v["type"])
	if kind == "session.update" {
		if err := keys(v, "event", "type", "event_id", "session"); err != nil {
			return out, err
		}
		if s.setupSent {
			return out, reject("session.update after initial setup")
		}
		setup, config, tools, err := s.setup(dict(v["session"]))
		if err != nil {
			return out, err
		}
		s.setupSent = true
		s.config = config
		s.tools = tools
		out.Upstream = append(out.Upstream, frame(map[string]any{"setup": setup}))
		return out, nil
	}
	if !s.ready {
		return out, &Error{Code: "session_not_ready", Param: "await setupComplete"}
	}
	switch kind {
	case "input_audio_buffer.append":
		if err := keys(v, "event", "type", "event_id", "audio"); err != nil {
			return out, err
		}
		if s.active || len(s.calls) > 0 {
			return out, reject("audio during response/tool cycle; await acknowledged cancellation")
		}
		if s.committed {
			return out, reject("append after commit before response.create")
		}
		audio, err := pcm(v["audio"])
		if err != nil {
			return out, err
		}
		if !s.inputOpen {
			s.inputOpen = true
			out.Upstream = append(out.Upstream, frame(map[string]any{"realtimeInput": map[string]any{"activityStart": map[string]any{}}}))
		}
		// GA PCM is 24 kHz. Preserve its samples and declare the actual rate:
		// Live documents resampling non-native input rates on the provider side.
		out.Upstream = append(out.Upstream, frame(map[string]any{"realtimeInput": map[string]any{"audio": map[string]any{"data": audio, "mimeType": "audio/pcm;rate=24000"}}}))
		s.inputHasAudio = true
	case "input_audio_buffer.commit":
		if err := keys(v, "event", "type", "event_id"); err != nil {
			return out, err
		}
		if !s.inputOpen || !s.inputHasAudio || s.committed {
			return out, &Error{Code: "invalid_state", Param: "no uncommitted audio"}
		}
		s.committed = true
		s.inputID = s.next("item")
		var previous any
		if s.lastItem != "" {
			previous = s.lastItem
		}
		out.Downstream = append(out.Downstream, s.event("input_audio_buffer.committed", map[string]any{"item_id": s.inputID, "previous_item_id": previous}))
		item := map[string]any{"id": s.inputID, "object": "realtime.item", "type": "message", "role": "user", "status": "completed", "content": []any{map[string]any{"type": "input_audio"}}}
		out.Downstream = append(out.Downstream, s.event("conversation.item.added", map[string]any{"item": item, "previous_item_id": previous}), s.event("conversation.item.done", map[string]any{"item": item, "previous_item_id": previous}))
		s.lastItem = s.inputID
	case "response.create":
		if err := keys(v, "event", "type", "event_id", "response"); err != nil {
			return out, err
		}
		if r, exists := v["response"]; exists {
			m := dict(r)
			if m == nil || len(m) > 0 {
				return out, reject("response.create overrides")
			}
		}
		if s.active {
			return out, &Error{Code: "response_in_progress", Param: "response.create"}
		}
		if s.interruptionEnding {
			return out, &Error{Code: "invalid_state", Param: "await interrupted turnComplete before response.create"}
		}
		if len(s.calls) > 0 {
			results := make([]any, 0, len(s.calls))
			for _, c := range s.calls {
				if !c.returned {
					return out, &Error{Code: "invalid_state", Param: "await all tool results"}
				}
				results = append(results, map[string]any{"id": c.id, "name": c.name, "response": c.result})
			}
			out.Upstream = append(out.Upstream, frame(map[string]any{"toolResponse": map[string]any{"functionResponses": results}}))
			s.calls = nil
		} else {
			if !s.committed {
				return out, &Error{Code: "invalid_state", Param: "response.create requires committed audio or tool results"}
			}
			// Sending activityEnd, not commit AND response.create independently,
			// is the single owner of starting this manual generation.
			out.Upstream = append(out.Upstream, frame(map[string]any{"realtimeInput": map[string]any{"activityEnd": map[string]any{}}}))
			s.inputOpen = false
			s.inputHasAudio = false
			s.committed = false
		}
		s.begin(&out)
	case "conversation.item.create":
		if err := keys(v, "event", "type", "event_id", "item"); err != nil {
			return out, err
		}
		m := dict(v["item"])
		if m == nil {
			return out, reject("item")
		}
		if err := keys(m, "item", "type", "call_id", "output"); err != nil {
			return out, err
		}
		if m["type"] != "function_call_output" {
			return out, reject("item.type")
		}
		id := text(m["call_id"])
		index := -1
		for i, c := range s.calls {
			if c.id == id {
				index = i
				break
			}
		}
		if index < 0 || s.calls[index].returned {
			return out, &Error{Code: "invalid_tool_result", Param: "unknown or duplicate call_id"}
		}
		value, ok := m["output"].(string)
		if !ok {
			return out, reject("item.output must be a string")
		}
		// OpenAI's output is an opaque string. Do not reinterpret JSON-looking
		// strings as a different value or round argument numbers through float64.
		s.calls[index].result = map[string]any{"output": value}
		s.calls[index].returned = true
		item := map[string]any{"id": s.next("item"), "object": "realtime.item", "type": "function_call_output", "call_id": id, "output": value, "status": "completed"}
		var previous any
		if s.lastItem != "" {
			previous = s.lastItem
		}
		out.Downstream = append(out.Downstream, s.event("conversation.item.added", map[string]any{"item": item, "previous_item_id": previous}), s.event("conversation.item.done", map[string]any{"item": item, "previous_item_id": previous}))
		s.lastItem = text(item["id"])
	case "response.cancel":
		if err := keys(v, "event", "type", "event_id", "response_id"); err != nil {
			return out, err
		}
		if !s.active || s.cancelPending {
			return out, &Error{Code: "invalid_state", Param: "no cancellable response"}
		}
		if id, ok := v["response_id"]; ok && id != s.responseID {
			return out, &Error{Code: "invalid_state", Param: "response_id"}
		}
		// START_OF_ACTIVITY_INTERRUPTS requests cancellation. No empty
		// activityEnd is sent: the activity remains open for the next audio.
		s.cancelPending = true
		s.inputOpen = true
		s.inputHasAudio = false
		out.Upstream = append(out.Upstream, frame(map[string]any{"realtimeInput": map[string]any{"activityStart": map[string]any{}}}))
	case "conversation.item.truncate":
		return out, s.fatal(kind + ": heard-history reconciliation unavailable")
	case "input_audio_buffer.clear":
		// Audio already sent to Live cannot be recalled by clearing a local
		// OpenAI input buffer. Do not pretend a clear acknowledgement did so.
		if s.inputOpen || s.committed {
			return out, s.fatal("input_audio_buffer.clear after upstream audio")
		}
		if err := keys(v, "event", "type", "event_id"); err != nil {
			return out, err
		}
		out.Downstream = append(out.Downstream, s.event("input_audio_buffer.cleared", nil))
	default:
		return out, reject("client event " + kind)
	}
	return out, nil
}

func (s *Session) begin(out *Output) {
	s.active = true
	s.responseID = s.next("resp")
	s.audioID = ""
	s.audioSeen = false
	s.transcriptSeen = false
	s.transcript = ""
	s.items = nil
	out.Downstream = append(out.Downstream, s.event("response.created", map[string]any{"response": map[string]any{"id": s.responseID, "object": "realtime.response", "status": "in_progress", "output": []any{}}}))
}

func (s *Session) setup(v map[string]any) (map[string]any, map[string]any, map[string]bool, error) {
	fail := func(path string) (map[string]any, map[string]any, map[string]bool, error) {
		return nil, nil, nil, reject(path)
	}
	if v == nil || s.model == "" {
		return fail("session/model")
	}
	if err := keys(v, "session", "type", "model", "audio", "output_modalities", "instructions", "tools", "tool_choice"); err != nil {
		return nil, nil, nil, err
	}
	if x, ok := v["type"]; ok && x != "realtime" {
		return fail("session.type")
	}
	if x, ok := v["model"]; ok && x != s.model {
		return fail("session.model")
	}
	if x, ok := v["output_modalities"]; ok {
		a, yes := x.([]any)
		if !yes || len(a) != 1 || a[0] != "audio" {
			return fail("session.output_modalities")
		}
	}
	audio := dict(v["audio"])
	if err := keys(audio, "session.audio", "input", "output"); err != nil {
		return nil, nil, nil, err
	}
	in, out := dict(audio["input"]), dict(audio["output"])
	if err := keys(in, "session.audio.input", "format", "turn_detection"); err != nil {
		return nil, nil, nil, err
	}
	if err := keys(out, "session.audio.output", "format", "voice"); err != nil {
		return nil, nil, nil, err
	}
	manual, exists := in["turn_detection"]
	if !exists || manual != nil {
		return fail("session.audio.input.turn_detection must be null")
	}
	for _, spec := range []struct {
		v          any
		rate, path string
	}{{in["format"], "24000", "session.audio.input.format"}, {out["format"], "24000", "session.audio.output.format"}} {
		m := dict(spec.v)
		if err := keys(m, spec.path, "type", "rate"); err != nil {
			return nil, nil, nil, err
		}
		// GA's PCM rate is optional, but its only supported value is 24000.
		rate, specified := m["rate"]
		if m["type"] != "audio/pcm" || (specified && rate != json.Number(spec.rate)) {
			return fail(spec.path)
		}
	}
	if out["voice"] != "Puck" {
		return fail("session.audio.output.voice must be Puck")
	}
	if x, ok := v["tool_choice"]; ok && x != "auto" {
		return fail("session.tool_choice")
	}
	model := s.model
	if !strings.HasPrefix(model, "models/") {
		model = "models/" + model
	}
	setup := map[string]any{"model": model, "generationConfig": map[string]any{"responseModalities": []string{"AUDIO"}, "speechConfig": map[string]any{"voiceConfig": map[string]any{"prebuiltVoiceConfig": map[string]any{"voiceName": "Puck"}}}}, "realtimeInputConfig": map[string]any{"automaticActivityDetection": map[string]any{"disabled": true}, "activityHandling": "START_OF_ACTIVITY_INTERRUPTS"}, "outputAudioTranscription": map[string]any{}}
	if x, ok := v["instructions"]; ok {
		t, yes := x.(string)
		if !yes {
			return fail("session.instructions")
		}
		setup["systemInstruction"] = map[string]any{"parts": []any{map[string]any{"text": t}}}
	}
	names := map[string]bool{}
	if x, ok := v["tools"]; ok {
		a, yes := x.([]any)
		if !yes {
			return fail("session.tools")
		}
		decl := make([]any, 0, len(a))
		for _, t := range a {
			m := dict(t)
			if err := keys(m, "session.tools", "type", "name", "description", "parameters"); err != nil {
				return nil, nil, nil, err
			}
			name := text(m["name"])
			if m["type"] != "function" || name == "" || names[name] {
				return fail("session.tools function name/type")
			}
			names[name] = true
			d := map[string]any{"name": name}
			if x, ok := m["description"]; ok {
				if _, yes := x.(string); !yes {
					return fail("session.tools.description")
				}
				d["description"] = x
			}
			if x, ok := m["parameters"]; ok {
				schema, err := parameters(dict(x))
				if err != nil {
					return nil, nil, nil, err
				}
				if schema["type"] != "OBJECT" {
					return fail("session.tools.parameters.type")
				}
				d["parameters"] = schema
			}
			decl = append(decl, d)
		}
		if len(decl) > 0 {
			setup["tools"] = []any{map[string]any{"functionDeclarations": decl}}
		}
	}
	return setup, v, names, nil
}

// parameters admits a finite common JSON Schema subset; unsupported constraints
// are not silently dropped when translating to Gemini's Schema representation.
func parameters(v map[string]any) (map[string]any, error) {
	if v == nil {
		return nil, reject("tool parameters")
	}
	if err := keys(v, "tool parameters", "type", "properties", "required", "items", "description", "enum"); err != nil {
		return nil, err
	}
	r := map[string]any{}
	t := text(v["type"])
	switch t {
	case "object", "array", "string", "number", "integer", "boolean":
		r["type"] = strings.ToUpper(t)
	default:
		return nil, reject("tool parameters.type")
	}
	for _, k := range []string{"description", "enum", "required"} {
		if x, ok := v[k]; ok {
			if k == "description" {
				if _, yes := x.(string); !yes {
					return nil, reject("tool parameters.description")
				}
			} else {
				a, yes := x.([]any)
				if !yes {
					return nil, reject("tool parameters." + k)
				}
				for _, s := range a {
					if _, yes := s.(string); !yes {
						return nil, reject("tool parameters." + k)
					}
				}
			}
			r[k] = x
		}
	}
	if x, ok := v["properties"]; ok {
		m := dict(x)
		if m == nil || t != "object" {
			return nil, reject("tool parameters.properties")
		}
		p := map[string]any{}
		for k, x := range m {
			v, err := parameters(dict(x))
			if err != nil {
				return nil, err
			}
			p[k] = v
		}
		r["properties"] = p
	}
	if x, ok := v["items"]; ok {
		if t != "array" {
			return nil, reject("tool parameters.items")
		}
		p, err := parameters(dict(x))
		if err != nil {
			return nil, err
		}
		r["items"] = p
	}
	return r, nil
}
