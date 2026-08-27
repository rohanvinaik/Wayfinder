import Lean
open Lean Meta Tactic

/-!
Minimal reproduction: `TryThis.getIndentAndColumn` panics when a
`Syntax.Range`'s byte offset falls inside a multibyte UTF-8 character.

`"𝕜"` is 4 bytes (F0 9D 95 9C), so byte offset 2 is not a character
boundary; `map.source.pos!` (the panicking constructor) aborts.
Well-formed parser ranges avoid this, but ranges produced or adjusted
by external drivers (REPLs, replay harnesses) reach this code path —
observed at ~1 panic per theorem-batch driving `exact?`/`apply?` over
Mathlib-derived goal states.
-/

#eval TryThis.getIndentAndColumn (FileMap.ofString "𝕜 : Type := by exact rfl") ⟨⟨2⟩, ⟨6⟩⟩
