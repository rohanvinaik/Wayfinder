# Wayfinder Context

<!-- LINTGATE:BEGIN compass_state v1 -->
## True North

Project purpose: Navigational proof search through structured semantic networks

## Architecture Philosophy

Uses NumPy for numerical computation

## Machine Rules
# LINTGATE_FORBID: proposed_rules
# LINTGATE_FORBID: existing_rule_count
# LINTGATE_FORBID: directives_analyzed
# LINTGATE_FORBID: already_covered

<!-- LINTGATE:END compass_state -->

## Implementation Notes

Uses unittest

## Thinking register (Rohan's rule — how deep-theory decisions get made)

Rigor and jargon are tools; use them rigorously — in the docs, the ledgers, the commit
messages, the papers. But when we're THINKING THROUGH a design decision together — holding
the whole theory of the project in our heads and deciding what to do — engage
conversationally, in plain thought. That's how Rohan thinks natively.

The calibration example (his own): Wayfinder wasn't conceived as "a principled approach to
proof-solving through data geometry and navigation of a constrained semantic domain." It was
conceived as: "When people solve proofs, they figure out what KIND of proof it is, then think
about how proofs like that are solved. Novel proofs don't need new fields of math invented —
they need novel combinations of existing ideas. Einstein thought about trains and clocks; he
didn't reinvent calculus. So proving even novel theorems is: understand what the theorem is
trying to say, and use the tools math already has."

So: technical register for artifacts, plain-thinking register for decisions. When a design
question goes to Rohan, frame it the second way — what would a person do here, and why —
with the formal anchor as a footnote, not the headline. He answers everything better that way.