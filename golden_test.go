package translate

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The golden tests characterize the exported conversions. Each case pins the
// input together with the value and report it produces, so a change in any
// of them shows up as a diff under testdata/golden. Regenerate with
// UPDATE_GOLDEN=1 only when a change is intended; cases are meant to be added,
// not rewritten.

type goldenCase struct {
	// name is the file path below testdata/golden, without extension.
	name string
	run  func(t *testing.T) map[string]any
}

func TestGoldenConversions(t *testing.T) {
	var cases []goldenCase
	for _, group := range [][]goldenCase{
		chatResponsesGoldenCases(),
		responsesChatGoldenCases(),
		anthropicGoldenCases(),
		anthropicResponseGoldenCases(),
		streamGoldenCases(),
		helperGoldenCases(),
		vendorGoldenCases(),
		alternateVendorGoldenCases(),
	} {
		cases = append(cases, group...)
	}
	seen := map[string]bool{}
	for _, c := range cases {
		if seen[c.name] {
			t.Fatalf("duplicate golden case %s", c.name)
		}
		seen[c.name] = true
		t.Run(c.name, func(t *testing.T) {
			checkGolden(t, c.name, c.run(t))
		})
	}
}

func checkGolden(t *testing.T, name string, record map[string]any) {
	t.Helper()
	got := renderGolden(t, record)
	path := filepath.Join("testdata", "golden", filepath.FromSlash(name)+".json")
	if os.Getenv("UPDATE_GOLDEN") == "1" {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create it): %v", err)
	}
	// A Windows checkout may convert the fixture to CRLF. Raw carriage
	// returns never occur inside the JSON, which escapes them in strings.
	want = bytes.ReplaceAll(want, []byte("\r"), nil)
	if !bytes.Equal(got, want) {
		t.Fatalf("golden mismatch for %s\n--- got ---\n%s\n--- want ---\n%s", path, got, want)
	}
}

// renderGolden marshals a record with normalized random identifiers and
// timestamps, so equal behavior renders to equal bytes.
func renderGolden(t *testing.T, record any) []byte {
	t.Helper()
	raw, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	for _, rule := range []struct{ pattern, replacement string }{
		{`\b(msg|toolu|resp)_[0-9a-f]{24}\b`, "${1}_RANDOM"},
		{`\bchatcmpl-[0-9a-f]{24}\b`, "chatcmpl-RANDOM"},
		{`\bcall_[0-9a-f]{12}\b`, "call_RANDOM"},
		{`"created_at":-?[0-9]+`, `"created_at":"NORMALIZED"`},
	} {
		text = regexp.MustCompile(rule.pattern).ReplaceAllString(text, rule.replacement)
	}
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		t.Fatal(err)
	}
	out, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return append(out, '\n')
}

// samePayload asserts that a plain conversion and its WithReport counterpart
// produced the same value once random identifiers are normalized.
func samePayload(t *testing.T, plain, reported any) {
	t.Helper()
	if a, b := renderGolden(t, plain), renderGolden(t, reported); !bytes.Equal(a, b) {
		t.Fatalf("plain and WithReport values differ\nplain:\n%s\nWithReport:\n%s", a, b)
	}
}

func decodeJSON[T any](t *testing.T, text string) T {
	t.Helper()
	var value T
	if err := json.Unmarshal([]byte(text), &value); err != nil {
		t.Fatalf("fixture %s: %v", text, err)
	}
	return value
}

// pullItems turns a fixed list into the pull function the stream converters read.
func pullItems(items []string) func() (string, bool) {
	next := 0
	return func() (string, bool) {
		if next >= len(items) {
			return "", false
		}
		next++
		return items[next-1], true
	}
}
