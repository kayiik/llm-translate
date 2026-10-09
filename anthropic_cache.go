package translate

import (
	"strconv"
	"strings"
)

// anthropicConversion is the complete result of converting Chat messages to
// Anthropic, including what only the WithReport form can return.
type anthropicConversion struct {
	system string
	// systemBlocks is set when a system or developer text part carries
	// cache_control. It holds the system text split at the breakpoints.
	systemBlocks []any
	messages     []map[string]any
	// droppedCacheControl lists the paths of cache_control values that the
	// conversion could not carry.
	droppedCacheControl []string
}

// textPiece is one text element of Chat content, with the separator that
// joins it to the previous piece.
type textPiece struct {
	sep          string
	text         string
	cacheControl any
}

// contentPieces splits Chat content into the pieces openaiContentToText
// joins with "\n". It returns the paths of cache_control values on text
// parts, which a conversion can carry, and on other parts, which the
// conversion drops together with the part.
func contentPieces(content any, prefix string) (pieces []textPiece, cached, lost []string) {
	parts, ok := content.([]any)
	if !ok {
		return []textPiece{{text: openaiContentToText(content)}}, nil, nil
	}
	for k, raw := range parts {
		sep := "\n"
		if len(pieces) == 0 {
			sep = ""
		}
		path := prefix + ".content." + strconv.Itoa(k) + ".cache_control"
		switch part := raw.(type) {
		case string:
			pieces = append(pieces, textPiece{sep: sep, text: part})
		case map[string]any:
			text, isText := part["text"].(string)
			isText = isText && part["type"] == "text"
			if isText {
				pieces = append(pieces, textPiece{sep: sep, text: text})
			}
			if !present(part["cache_control"]) {
				continue
			}
			if isText {
				pieces[len(pieces)-1].cacheControl = part["cache_control"]
				cached = append(cached, path)
			} else {
				lost = append(lost, path)
			}
		}
	}
	return pieces, cached, lost
}

// cacheBlocks renders pieces as Anthropic text blocks. Each block ends at a
// piece with cache_control and carries it, so the cached prefix is the one
// the Chat request marked. Text inside a block is joined exactly as
// openaiContentToText joins it; only the block boundaries are new.
func cacheBlocks(pieces []textPiece) []any {
	var blocks []any
	var text strings.Builder
	open := false
	for _, piece := range pieces {
		if open {
			text.WriteString(piece.sep)
		}
		text.WriteString(piece.text)
		open = true
		if piece.cacheControl != nil {
			blocks = append(blocks, map[string]any{"type": "text", "text": text.String(), "cache_control": piece.cacheControl})
			text.Reset()
			open = false
		}
	}
	if open && text.Len() > 0 {
		blocks = append(blocks, map[string]any{"type": "text", "text": text.String()})
	}
	return blocks
}
