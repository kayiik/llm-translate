package realtime

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"strings"
	"testing"
)

// The bridge preserves signed PCM16LE mono samples. Gemini, not this package,
// performs any resampling; these tests do not measure provider DSP quality.
func rateAudio(t *testing.T, o Output) []byte {
	t.Helper()
	var result []byte
	for _, b := range o.Upstream {
		input := dict(decode(t, b)["realtimeInput"])
		if audio := dict(input["audio"]); audio != nil {
			if audio["mimeType"] != "audio/pcm;rate=24000" {
				t.Fatal("actual input rate not declared", audio["mimeType"])
			}
			data, err := base64.StdEncoding.Strict().DecodeString(text(audio["data"]))
			if err != nil || len(data) == 0 || len(data)%2 != 0 {
				t.Fatal("invalid upstream PCM framing", err)
			}
			result = append(result, data...)
		}
	}
	return result
}

func TestGA24kSignedPCMChunksPreserveSamplesAndDuration(t *testing.T) {
	// Exactly one second, including both signed extrema and asymmetric LE bytes.
	samples := []int16{-32768, -12345, -1, 0, 1, 12345, 32767}
	input := make([]byte, 24000*2)
	for i := 0; i < len(input)/2; i++ {
		binary.LittleEndian.PutUint16(input[i*2:], uint16(samples[i%len(samples)]))
	}
	stream := func(chunks []int) []byte {
		t.Helper()
		s := ready(t)
		var got []byte
		for offset, n := 0, 0; offset < len(input); n++ {
			end := offset + chunks[n%len(chunks)]*2
			if end > len(input) {
				end = len(input)
			}
			o := client(t, s, `{"type":"input_audio_buffer.append","audio":"`+base64.StdEncoding.EncodeToString(input[offset:end])+`"}`)
			wantFrames := 1
			if offset == 0 {
				wantFrames = 2
				if !strings.Contains(string(o.Upstream[0]), "activityStart") {
					t.Fatal("missing first activityStart")
				}
			}
			if len(o.Upstream) != wantFrames {
				t.Fatal("unexpected framing at chunk boundary", len(o.Upstream))
			}
			got = append(got, rateAudio(t, o)...)
			offset = end
		}
		if o := client(t, s, `{"type":"input_audio_buffer.commit"}`); len(o.Upstream) != 0 {
			t.Fatal("passthrough unexpectedly buffered samples at commit")
		}
		if o := client(t, s, `{"type":"response.create"}`); len(o.Upstream) != 1 || !strings.Contains(string(o.Upstream[0]), "activityEnd") {
			t.Fatal("manual generation owner changed")
		}
		// Output is still 24 kHz and must preserve the same signed byte patterns.
		o := server(t, s, `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"`+base64.StdEncoding.EncodeToString(input[:14])+`"}}]},"turnComplete":true}}`)
		var output []byte
		for _, event := range o.Downstream {
			v := decode(t, event)
			if v["type"] == "response.output_audio.delta" {
				var err error
				output, err = base64.StdEncoding.DecodeString(text(v["delta"]))
				if err != nil {
					t.Fatal(err)
				}
			}
		}
		if !bytes.Equal(output, input[:14]) || !strings.HasSuffix(types(t, o), "response.done") {
			t.Fatal("output samples/lifecycle changed")
		}
		// No partial converter history leaks into a subsequent input turn.
		if next := client(t, s, `{"type":"input_audio_buffer.append","audio":"AIAAAP9/"}`); !bytes.Equal(rateAudio(t, next), []byte{0, 128, 0, 0, 255, 127}) {
			t.Fatal("next-turn bytes changed")
		}
		return got
	}
	whole := stream([]int{24000})
	split := stream([]int{1, 2, 17, 509, 3, 1024})
	if !bytes.Equal(whole, input) || !bytes.Equal(split, whole) || len(split)/2 != 24000 {
		t.Fatal("split-chunk sample count/one-second duration changed")
	}
}

func TestGA24kRateDefaultsAndRejections(t *testing.T) {
	for _, p := range []string{
		strings.Replace(profile, `,"rate":24000`, "", 1),
		strings.ReplaceAll(profile, `,"rate":24000`, ""),
	} {
		s := New("fixture-live")
		client(t, s, p)
		server(t, s, `{"setupComplete":{}}`)
		if got := rateAudio(t, client(t, s, `{"type":"input_audio_buffer.append","audio":"AIAAAP9/"}`)); !bytes.Equal(got, []byte{0, 128, 0, 0, 255, 127}) {
			t.Fatal("omitted rate did not default to unchanged 24k PCM")
		}
	}
	for _, rate := range []string{"16000", "48000", "null", `"24000"`} {
		for _, p := range []string{
			strings.Replace(profile, `"rate":24000`, `"rate":`+rate, 1),
			strings.ReplaceAll(profile, `"rate":24000`, `"rate":`+rate),
			strings.Replace(profile, `"output":{"format":{"type":"audio/pcm","rate":24000}`, `"output":{"format":{"type":"audio/pcm","rate":`+rate+`}`, 1),
		} {
			s := New("fixture-live")
			if o, err := s.Client([]byte(p)); err == nil || len(o.Upstream) != 0 || len(o.Downstream) != 0 {
				t.Fatal("invalid client rate dispatched", rate)
			}
		}
	}
}

func TestGA24kFramingBoundsAndClear(t *testing.T) {
	// Retain the existing six-MiB encoded-audio bound; do not loosen it to
	// accommodate a converter. The transport can impose a tighter message cap.
	const limit = 6 * 1024 * 1024
	if _, err := pcm(strings.Repeat("AAAA", limit/4)); err != nil {
		t.Fatal("previous boundary rejected", err)
	}
	if _, err := pcm(strings.Repeat("AAAA", limit/4+1)); err == nil {
		t.Fatal("oversized audio accepted")
	}
	s := ready(t)
	if o := client(t, s, `{"type":"input_audio_buffer.clear"}`); types(t, o) != "input_audio_buffer.cleared" || len(o.Upstream) != 0 {
		t.Fatal("empty clear changed")
	}
	// Each append must contain complete samples, even at arbitrary chunk sizes.
	if o, err := s.Client([]byte(`{"type":"input_audio_buffer.append","audio":"AA=="}`)); err == nil || len(o.Upstream) != 0 || s.inputOpen {
		t.Fatal("partial sample dispatched or changed activity state")
	}
	o := client(t, s, `{"type":"input_audio_buffer.append","audio":"AIAAAP9/"}`)
	if !bytes.Equal(rateAudio(t, o), []byte{0, 128, 0, 0, 255, 127}) {
		t.Fatal("invalid append contaminated subsequent samples")
	}
	if o, err := s.Client([]byte(`{"type":"input_audio_buffer.clear"}`)); err == nil || len(o.Upstream) != 0 || len(o.Downstream) != 0 || !s.unusable {
		t.Fatal("clear falsely recalled provider input")
	}
}
