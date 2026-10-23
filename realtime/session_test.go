package realtime

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

const profile = `{"type":"session.update","session":{"type":"realtime","output_modalities":["audio"],"instructions":"Speak briefly.","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"turn_detection":null},"output":{"format":{"type":"audio/pcm","rate":24000},"voice":"Puck"}},"tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}}],"tool_choice":"auto"}}`

func decode(t *testing.T, b []byte) map[string]any {
	t.Helper()
	v, err := object(b)
	if err != nil {
		t.Fatal(err)
	}
	return v
}
func client(t *testing.T, s *Session, b string) Output {
	t.Helper()
	v, err := s.Client([]byte(b))
	if err != nil {
		t.Fatal(err)
	}
	return v
}
func server(t *testing.T, s *Session, b string) Output {
	t.Helper()
	v, err := s.Server([]byte(b))
	if err != nil {
		t.Fatal(err)
	}
	return v
}
func types(t *testing.T, o Output) string {
	t.Helper()
	var a []string
	for _, b := range o.Downstream {
		a = append(a, text(decode(t, b)["type"]))
	}
	return strings.Join(a, ",")
}
func ready(t *testing.T) *Session {
	t.Helper()
	s := New("fixture-live")
	if len(s.Start()) != 1 || len(s.Start()) != 0 {
		t.Fatal("Start not once")
	}
	o := client(t, s, profile)
	if len(o.Downstream) != 0 || len(o.Upstream) != 1 {
		t.Fatal("setup acknowledged before upstream")
	}
	setup := dict(decode(t, o.Upstream[0])["setup"])
	if setup["model"] != "models/fixture-live" || dict(dict(setup["realtimeInputConfig"])["automaticActivityDetection"])["disabled"] != true {
		t.Fatal(setup)
	}
	if strings.Contains(string(o.Upstream[0]), "marin") || !strings.Contains(string(o.Upstream[0]), `"voiceName":"Puck"`) {
		t.Fatal("voice substituted")
	}
	if types(t, server(t, s, `{"setupComplete":{}}`)) != "session.updated" {
		t.Fatal("missing setup acknowledgement")
	}
	return s
}
func spoken(t *testing.T, s *Session) string {
	t.Helper()
	o := client(t, s, `{"type":"input_audio_buffer.append","audio":"AQACAA=="}`)
	if len(o.Upstream) != 2 || !strings.Contains(string(o.Upstream[0]), "activityStart") || !strings.Contains(string(o.Upstream[1]), `"data":"AQACAA=="`) {
		t.Fatalf("input PCM/activity: %s", o.Upstream)
	}
	o = client(t, s, `{"type":"input_audio_buffer.commit"}`)
	if len(o.Upstream) != 0 || !strings.HasPrefix(types(t, o), "input_audio_buffer.committed,conversation.item.added,conversation.item.done") {
		t.Fatal("commit generated prematurely", types(t, o))
	}
	input := text(decode(t, o.Downstream[0])["item_id"])
	o = client(t, s, `{"type":"response.create"}`)
	if len(o.Upstream) != 1 || !strings.Contains(string(o.Upstream[0]), "activityEnd") || types(t, o) != "response.created" {
		t.Fatal("single generation owner", o)
	}
	if _, err := s.Client([]byte(`{"type":"response.create"}`)); err == nil {
		t.Fatal("double generation accepted")
	}
	return input
}

func TestSpokenTurnAudioTranscriptAndContinuation(t *testing.T) {
	s := ready(t)
	firstInput := spoken(t, s)
	o := server(t, s, `{"serverContent":{"modelTurn":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AwAEAA=="}}]},"outputTranscription":{"text":"Hello"}}}`)
	if types(t, o) != "response.output_item.added,response.content_part.added,response.output_audio.delta,response.output_audio_transcript.delta" {
		t.Fatal(types(t, o))
	}
	delta := decode(t, o.Downstream[2])
	if delta["delta"] != "AwAEAA==" || delta["item_id"] == firstInput {
		t.Fatal("audio corrupt or IDs aliased", delta)
	}
	item, response := delta["item_id"], delta["response_id"]
	o = server(t, s, `{"serverContent":{"generationComplete":true}}`)
	if len(o.Downstream) != 0 {
		t.Fatal("generation falsely acknowledged playback/turn completion")
	}
	o = server(t, s, `{"serverContent":{"turnComplete":true},"usageMetadata":{"totalTokenCount":9007199254740993}}`)
	if types(t, o) != "response.output_audio.done,response.output_audio_transcript.done,response.content_part.done,response.output_item.done,response.done" {
		t.Fatal(types(t, o))
	}
	for _, b := range o.Downstream {
		v := decode(t, b)
		if v["response_id"] != nil && v["response_id"] != response {
			t.Fatal("response ID drift")
		}
	}
	done := dict(decode(t, o.Downstream[len(o.Downstream)-1])["response"])
	if _, ok := done["usage"]; ok {
		t.Fatal("fabricated usage scope")
	}
	if dict(done["output"].([]any)[0])["id"] != item {
		t.Fatal("item ID drift")
	}
	secondInput := spoken(t, s)
	if secondInput == firstInput {
		t.Fatal("item reused across turns")
	}
	server(t, s, `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"BQAGAA=="}}]},"turnComplete":true}}`)
}

func TestFunctionCycleCorrelatesIDsAndExactArguments(t *testing.T) {
	s := ready(t)
	spoken(t, s)
	o := server(t, s, `{"toolCall":{"functionCalls":[{"id":"native-call-1","name":"lookup","args":{"n":9007199254740993}},{"id":"native-call-2","name":"lookup","args":{"n":2}}]}}`)
	if strings.Count(types(t, o), "response.function_call_arguments.done") != 2 || !strings.HasSuffix(types(t, o), "response.done") {
		t.Fatal(types(t, o))
	}
	if !strings.Contains(string(o.Downstream[1]), "9007199254740993") {
		t.Fatal("argument precision lost", string(o.Downstream[1]))
	}
	for _, id := range []string{"native-call-2", "native-call-1"} {
		o = client(t, s, `{"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"`+id+`","output":"{\"answer\":true}"}}`)
		if len(o.Upstream) != 0 {
			t.Fatal("tool result generated before response.create")
		}
		if _, err := s.Client([]byte(`{"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"` + id + `","output":"duplicate"}}`)); err == nil {
			t.Fatal("duplicate result accepted")
		}
	}
	o = client(t, s, `{"type":"response.create"}`)
	if len(o.Upstream) != 1 || !strings.Contains(string(o.Upstream[0]), "toolResponse") || strings.Contains(string(o.Upstream[0]), "activityEnd") {
		t.Fatal("wrong tool continuation", o)
	}
	results := dict(decode(t, o.Upstream[0])["toolResponse"])["functionResponses"].([]any)
	if dict(results[0])["id"] != "native-call-1" || dict(results[1])["id"] != "native-call-2" || dict(dict(results[0])["response"])["output"] != `{"answer":true}` {
		t.Fatal("result identity/value drift", results)
	}
	server(t, s, `{"serverContent":{"outputTranscription":{"text":"Found it"},"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AQACAA=="}}]},"turnComplete":true}}`)
	spoken(t, s)
}

func TestAudioThenToolKeepsDistinctOutputIndexes(t *testing.T) {
	s := ready(t)
	spoken(t, s)
	server(t, s, `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AQACAA=="}}]}}}`)
	o := server(t, s, `{"toolCall":{"functionCalls":[{"id":"call","name":"lookup","args":{"n":1}}]}}`)
	added := decode(t, o.Downstream[0])
	if added["output_index"] != json.Number("1") {
		t.Fatal("audio/tool index collision", added)
	}
	done := dict(decode(t, o.Downstream[len(o.Downstream)-1])["response"])
	items := done["output"].([]any)
	if len(items) != 2 || dict(items[0])["type"] != "message" || dict(items[1])["type"] != "function_call" {
		t.Fatal("audio overwrote tool", items)
	}
}

func TestUnsupportedSettingsAndInputDoNotDispatch(t *testing.T) {
	for _, p := range []string{
		strings.Replace(profile, `"Puck"`, `"marin"`, 1),
		strings.Replace(profile, `"turn_detection":null`, `"turn_detection":{"type":"server_vad"}`, 1),
		strings.Replace(profile, `"rate":24000`, `"rate":16000`, 1),
		strings.Replace(profile, `"type":"integer"`, `"type":"integer","minimum":1`, 1),
		strings.Replace(profile, `"tool_choice":"auto"`, `"tool_choice":"required"`, 1),
	} {
		s := New("fixture-live")
		o, err := s.Client([]byte(p))
		if err == nil || len(o.Upstream) != 0 || len(o.Downstream) != 0 {
			t.Fatalf("unsupported profile dispatched: %v %s", err, p)
		}
	}
	s := ready(t)
	for _, event := range []string{`{"type":"input_audio_buffer.append","audio":"AA=="}`, `{"type":"input_audio_buffer.append","audio":"not base64"}`, `{"type":"response.create"}`, `{"type":"session.update","session":{}}`, `{"type":"conversation.item.create","item":{"type":"message"}}`} {
		o, err := s.Client([]byte(event))
		if err == nil || len(o.Upstream) != 0 {
			t.Fatal("invalid input dispatched", event)
		}
	}
}

func TestTruncateRequiresCloseAndNeverClaimsRollback(t *testing.T) {
	for _, event := range []string{`{"type":"conversation.item.truncate","item_id":"unknown","audio_end_ms":10}`} {
		s := ready(t)
		spoken(t, s)
		o, err := s.Client([]byte(event))
		var semantic *Error
		if !errors.As(err, &semantic) || !semantic.RequiresClose || len(o.Upstream) != 0 || len(o.Downstream) != 0 {
			t.Fatal("fabricated cancel/rollback", o, err)
		}
		if _, err = s.Client([]byte(`{"type":"input_audio_buffer.append","audio":"AQACAA=="}`)); err == nil {
			t.Fatal("unsafe continuation accepted")
		}
	}
	s := ready(t)
	spoken(t, s)
	if _, err := s.Server([]byte(`{"serverContent":{"interrupted":true}}`)); err == nil {
		t.Fatal("provider interruption ignored")
	}
	if _, err := s.Server([]byte(`{"serverContent":{"turnComplete":true}}`)); err == nil {
		t.Fatal("continued after unsupported history change")
	}
}

func TestManualInterruptAcknowledgementContinuationAndTools(t *testing.T) {
	s := ready(t)
	// A complete spoken response, then another response interrupted mid-audio.
	spoken(t, s)
	server(t, s, `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AQACAA=="}}]},"turnComplete":true}}`)
	spoken(t, s)
	server(t, s, `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AwAEAA=="}}]}}}`)
	o := client(t, s, `{"type":"response.cancel"}`)
	if len(o.Downstream) != 0 || len(o.Upstream) != 1 || !strings.Contains(string(o.Upstream[0]), "activityStart") || strings.Contains(string(o.Upstream[0]), "activityEnd") {
		t.Fatal("cancel ack fabricated or empty turn sent", o)
	}
	// Already-in-flight audio is not appended to client playback after cancel.
	if o := server(t, s, `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"BQAGAA=="}}]}}}`); len(o.Downstream) != 0 {
		t.Fatal("late cancelled audio leaked")
	}
	o = server(t, s, `{"serverContent":{"interrupted":true}}`)
	if len(o.Downstream) != 0 {
		t.Fatal("cancel acknowledged before interrupted turnComplete")
	}
	if _, err := s.Client([]byte(`{"type":"input_audio_buffer.append","audio":"AQACAA=="}`)); err == nil {
		t.Fatal("new input before cancellation settles")
	}
	o = server(t, s, `{"serverContent":{"turnComplete":true}}`)
	done := dict(decode(t, o.Downstream[len(o.Downstream)-1])["response"])
	if done["status"] != "cancelled" || strings.Contains(types(t, o), "response.cancelled") || strings.Contains(types(t, o), "item.truncated") {
		t.Fatal("noncanonical or fabricated cancellation", types(t, o), done)
	}
	// The interruption activity remains open; do not start a second one.
	o = client(t, s, `{"type":"input_audio_buffer.append","audio":"AQACAA=="}`)
	if len(o.Upstream) != 1 || strings.Contains(string(o.Upstream[0]), "activityStart") {
		t.Fatal("double activityStart")
	}
	client(t, s, `{"type":"input_audio_buffer.commit"}`)
	o = client(t, s, `{"type":"response.create"}`)
	if len(o.Upstream) != 1 || !strings.Contains(string(o.Upstream[0]), "activityEnd") {
		t.Fatal("continuation missing manual activityEnd")
	}
	server(t, s, `{"toolCall":{"functionCalls":[{"id":"after-interrupt","name":"lookup","args":{"n":4}}]}}`)
	client(t, s, `{"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"after-interrupt","output":"ok"}}`)
	client(t, s, `{"type":"response.create"}`)
	o = server(t, s, `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"BwAIAA=="}}]},"outputTranscription":{"text":"Continued"},"turnComplete":true}}`)
	if !strings.Contains(types(t, o), "response.output_audio.delta") || !strings.HasSuffix(types(t, o), "response.done") {
		t.Fatal("same-session voice/tool continuation failed", types(t, o))
	}
}

func TestSetupFenceAndUnsolicitedGeneration(t *testing.T) {
	s := New("fixture-live")
	client(t, s, profile)
	if _, err := s.Client([]byte(`{"type":"input_audio_buffer.append","audio":"AQACAA=="}`)); err == nil {
		t.Fatal("sent before setupComplete")
	}
	s = ready(t)
	if _, err := s.Server([]byte(`{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AQACAA=="}}]}}}`)); err == nil {
		t.Fatal("accepted unowned generation")
	}
}

func TestCancelIDAndEmptyActivityCannotStartGeneration(t *testing.T) {
	s := ready(t)
	spoken(t, s)
	if out, err := s.Client([]byte(`{"type":"response.cancel","response_id":"wrong"}`)); err == nil || len(out.Upstream) != 0 {
		t.Fatal("wrong response cancellation dispatched")
	}
	client(t, s, `{"type":"response.cancel"}`)
	if out, err := s.Client([]byte(`{"type":"response.cancel"}`)); err == nil || len(out.Upstream) != 0 {
		t.Fatal("duplicate interruption dispatched")
	}
	o := server(t, s, `{"serverContent":{"interrupted":true,"turnComplete":true}}`)
	if strings.Count(types(t, o), "response.done") != 1 {
		t.Fatal("interruption terminal duplicated", types(t, o))
	}
	if out, err := s.Client([]byte(`{"type":"input_audio_buffer.commit"}`)); err == nil || len(out.Upstream) != 0 {
		t.Fatal("empty interruption activity committed")
	}
	if out, err := s.Client([]byte(`{"type":"response.create"}`)); err == nil || len(out.Upstream) != 0 {
		t.Fatal("dummy activityEnd started generation")
	}
}
