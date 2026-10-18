package translate

import "fmt"

// Protocol names are proper nouns in the existing public error text. Keep
// their spelling separate from the lower-case diagnostic, without changing
// messages or suppressing checks for ordinary capitalized error sentences.
func protocolErrorf(protocol, format string, args ...any) error {
	return fmt.Errorf("%s %s", protocol, fmt.Sprintf(format, args...))
}
