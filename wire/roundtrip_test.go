package wire

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// canonical parses JSON for comparison, keeping numbers as their literal
// text so 1.0 and 1 stay different.
func canonical(t *testing.T, data []byte) any {
	t.Helper()
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		t.Fatalf("parse %s: %v", data, err)
	}
	if _, err := decoder.Token(); err != io.EOF {
		t.Fatalf("trailing data after JSON value: %s", data)
	}
	return value
}

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// assertRoundTrip decodes data, encodes the result and requires the output
// to equal the input as JSON. It also requires a second pass to be stable.
func assertRoundTrip[T any](t *testing.T, data []byte, decode func([]byte) (T, error)) T {
	t.Helper()
	value, err := decode(data)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if !reflect.DeepEqual(canonical(t, data), canonical(t, encoded)) {
		t.Fatalf("round trip changed the payload\ninput:  %s\noutput: %s", bytes.Join(bytes.Fields(data), nil), encoded)
	}
	again, err := decode(encoded)
	if err != nil {
		t.Fatalf("decode encoded: %v", err)
	}
	if reencoded, err := json.Marshal(again); err != nil || !bytes.Equal(encoded, reencoded) {
		t.Fatalf("second pass is not stable (%v):\n%s\n%s", err, encoded, reencoded)
	}
	return value
}

// assertEachRoundTrips round-trips every element of a JSON array fixture,
// such as the chunks of one stream.
func assertEachRoundTrips[T any](t *testing.T, data []byte, decode func([]byte) (T, error)) []T {
	t.Helper()
	var items []json.RawMessage
	if err := json.Unmarshal(data, &items); err != nil {
		t.Fatal(err)
	}
	values := make([]T, len(items))
	for i, item := range items {
		values[i] = assertRoundTrip(t, item, decode)
	}
	return values
}

func str(s string) *string { return &s }
