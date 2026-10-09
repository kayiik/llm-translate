package wire

import "encoding/json"

// CacheControl is a prompt-caching breakpoint, such as
// {"type":"ephemeral","ttl":"1h"}. Anthropic accepts it on content blocks,
// system blocks and tools; several Chat Completions proxies accept it on
// content parts and forward it to Anthropic.
type CacheControl struct {
	Type  string
	TTL   string
	Extra map[string]json.RawMessage
}

func (c *CacheControl) UnmarshalJSON(data []byte) error {
	m, err := decodeMembers(data)
	if err != nil {
		return err
	}
	*c = CacheControl{}
	m.name("type", &c.Type)
	m.name("ttl", &c.TTL)
	c.Extra = m.extra()
	return nil
}

func (c CacheControl) MarshalJSON() ([]byte, error) {
	var o object
	o.name("type", c.Type)
	o.name("ttl", c.TTL)
	return o.finish(c.Extra)
}
