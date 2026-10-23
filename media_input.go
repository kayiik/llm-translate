package translate

import "strings"

func fileInputLosses(raw any, path string) []Loss {
	file, ok := asMap(raw)
	if !ok {
		return []Loss{material(path, LossUnsupported, "file input must be an object")}
	}
	var losses []Loss
	sources := 0
	for _, key := range []string{"file_data", "file_id"} {
		if value, exists := file[key]; exists {
			text, valid := value.(string)
			if !valid || strings.TrimSpace(text) == "" {
				losses = append(losses, material(path+"."+key, LossUnsupported, "file source must be a nonempty string"))
			}
			sources++
		}
	}
	if sources != 1 {
		losses = append(losses, material(path, LossUnsupported, "file input requires exactly one of file_data or file_id"))
	}
	if value, exists := file["filename"]; exists {
		if name, valid := value.(string); !valid || strings.TrimSpace(name) == "" {
			losses = append(losses, material(path+".filename", LossUnsupported, "filename must be a nonempty string"))
		}
	}
	for key, value := range file {
		if value != nil && key != "filename" && key != "file_data" && key != "file_id" {
			losses = append(losses, material(path+"."+key, LossUnsupported, "file member is not representable by this conversion"))
		}
	}
	return losses
}

func responseFileFields(part map[string]any) map[string]any {
	file := map[string]any{}
	for _, key := range []string{"filename", "file_data", "file_id"} {
		if value, exists := part[key]; exists {
			file[key] = value
		}
	}
	return file
}
