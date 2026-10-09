package translate

import (
	"fmt"
	"sort"
	"strings"
)

// LossClass describes how a source value differs after conversion.
type LossClass string

const (
	LossUnsupported  LossClass = "unsupported"
	LossDropped      LossClass = "dropped"
	LossApproximated LossClass = "approximated"
	LossRenamed      LossClass = "renamed"
)

// LossSeverity distinguishes fidelity failures from useful compatibility notes.
type LossSeverity string

const (
	LossMaterial LossSeverity = "material"
	LossAdvisory LossSeverity = "advisory"
)

// Loss is one deterministic, machine-readable compatibility finding.
type Loss struct {
	Path     string       `json:"path"`
	Class    LossClass    `json:"class"`
	Severity LossSeverity `json:"severity"`
	Detail   string       `json:"detail"`
}

// Report describes all known compatibility losses for one conversion.
// Losses are sorted by path, severity, class, then detail.
type Report struct {
	Losses []Loss `json:"losses,omitempty"`
}

// ConversionResult couples an existing conversion value to its compatibility report.
type ConversionResult[T any] struct {
	Value  T      `json:"value"`
	Report Report `json:"report"`
}

// NewReport normalizes ordering and removes duplicate losses.
func NewReport(losses ...Loss) Report {
	ordered := append([]Loss(nil), losses...)
	sort.Slice(ordered, func(i, j int) bool {
		a, b := ordered[i], ordered[j]
		if a.Path != b.Path {
			return a.Path < b.Path
		}
		if a.Severity != b.Severity {
			return a.Severity < b.Severity
		}
		if a.Class != b.Class {
			return a.Class < b.Class
		}
		return a.Detail < b.Detail
	})
	unique := ordered[:0]
	for _, loss := range ordered {
		if len(unique) == 0 || unique[len(unique)-1] != loss {
			unique = append(unique, loss)
		}
	}
	return Report{Losses: unique}
}

// HasMaterialLoss reports whether conversion changed or omitted material semantics.
func (r Report) HasMaterialLoss() bool {
	for _, loss := range r.Losses {
		if loss.Severity == LossMaterial {
			return true
		}
	}
	return false
}

// MaterialLossError contains the material subset of a rejected report.
type MaterialLossError struct {
	Losses []Loss
}

func (e *MaterialLossError) Error() string {
	paths := make([]string, len(e.Losses))
	for i, loss := range e.Losses {
		paths[i] = loss.Path
	}
	return fmt.Sprintf("conversion has material compatibility loss at %s", strings.Join(paths, ", "))
}

// RejectMaterialLoss returns an error containing every material loss in the report.
func RejectMaterialLoss(report Report) error {
	var losses []Loss
	for _, loss := range report.Losses {
		if loss.Severity == LossMaterial {
			losses = append(losses, loss)
		}
	}
	if len(losses) == 0 {
		return nil
	}
	return &MaterialLossError{Losses: losses}
}

// RejectMaterialLoss returns an error when this result contains material loss.
func (r ConversionResult[T]) RejectMaterialLoss() error {
	return RejectMaterialLoss(r.Report)
}

func material(path string, class LossClass, detail string) Loss {
	return Loss{Path: path, Class: class, Severity: LossMaterial, Detail: detail}
}

func advisory(path string, class LossClass, detail string) Loss {
	return Loss{Path: path, Class: class, Severity: LossAdvisory, Detail: detail}
}
