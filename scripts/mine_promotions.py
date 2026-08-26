"""Mine promotion candidates from the EXP-RF-007 proof corpus.

The positive promotion loop's stage 1 (the significance corpus's §16.2
ledger, transported): extract head-normalized tactic FORMS from the 40
Aristotle proofs of certified Wayfinder-residual theorems, count
recurrence across DISTINCT theorems (the learning signal), cross-reference
each form against what the current grammar/skeleton stack can utter, and
attach the population warrant (the form's share of the 338K-step traced
ground-truth corpus) per THE_REFOUNDING §10.7: a promotion needs BOTH
needed-by-the-residual AND corpus-population presence, and forms are
parameterized by shape (head + arity + at/using/with flags), never by
lemma identity.

Output: runs/promotion_candidates.json — the ranked ledger the promote
step consumes. This script proposes; promotions are authored, pinned,
and paired-measured by hand.

Usage:
    .venv/bin/python -m scripts.mine_promotions
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Forms the CURRENT deterministic stack can utter (tactic_compiler v2 +
# Dr. Ducky skeletons + structural/solver lanes). Used to mark candidates
# expressible/absent — the diff IS the promotion surface.
EXPRESSIBLE = {
    "rw#1", "rw#2", "simp#0", "simp#1", "simp#2", "simp#3", "simpa#0",
    "simpa+using", "simpa#1", "exact", "apply", "refine+holes", "intro",
    "intros", "constructor", "rfl", "trivial", "assumption", "norm_num#0",
    "ring#0", "omega#0", "decide#0", "linarith#0", "nlinarith#0",
    "positivity#0", "aesop#0", "simp_all#0", "field_simp#0", "push_cast#0",
    "norm_cast#0", "ext#0", "funext#0", "congr#0", "solve_by_elim#0",
    "use", "obtain+anon", "have+assign", "left", "right", "gcongr#0",
    "tauto#0", "push_neg#0", "ring_nf#0", "induction", "simp_rw#1",
    "exact?", "apply?", "infer_instance",
}

_BULLET = re.compile(r"^[·\-\*]\s*|^\.\s+")
_HEAD = re.compile(r"^([A-Za-z_][A-Za-z0-9_'?]*)")


def split_tactic_steps(body: str) -> list[str]:
    """Split a `by`-block body into top-level tactic steps.

    Line-based with bracket/paren/anonymous-constructor depth tracking:
    continuation lines (positive depth, or starting with an operator/calc
    underscore) fold into the current step; bullets are stripped (the
    branching structure is not the form); `<;>` stays inside its step
    (it IS part of the form).
    """
    steps: list[str] = []
    cur = ""
    depth = 0
    for raw in body.splitlines():
        line = _BULLET.sub("", raw.strip())
        if not line or line.startswith("--"):
            continue
        opens = sum(line.count(c) for c in "([{⟨")
        closes = sum(line.count(c) for c in ")]}⟩")
        starts_cont = depth > 0 or line.startswith(("_", "≤", "=", "<", ">", "|", "↦"))
        if cur and starts_cont:
            cur += " " + line
        else:
            if cur:
                steps.append(cur)
            cur = line
        depth += opens - closes
        if depth < 0:
            depth = 0
    if cur:
        steps.append(cur)
    # split top-level `;` chains (but never `<;>`)
    out: list[str] = []
    for s in steps:
        parts = re.split(r"(?<!<);(?!>)", s)
        out.extend(p.strip() for p in parts if p.strip())
    return out


def normalize_step(step: str) -> str:
    """Head-normalize one tactic step to its FORM code.

    The form carries: head · bracket-arity band (#0/#1/#2/#3/#4+) ·
    structural flags (+at hypothesis-side, +using, +with destructuring,
    +anon ⟨⟩ constructor, +assign `:=`, +seq `<;>`). Lemma names never
    survive — §10.7's never-by-identity, enforced at the representation.
    """
    s = step.strip()
    if s.startswith("calc"):
        return "calc"
    m = _HEAD.match(s)
    if not m:
        return "term" if s.startswith(("⟨", "fun", "(")) else "?"
    head = m.group(1)
    rest = s[m.end():]
    br = re.search(r"\[([^\]]*)\]", rest)
    arity = 0
    if br:
        n = len([a for a in br.group(1).split(",") if a.strip()])
        arity = n if n < 4 else 4
    flags = ""
    if re.search(r"\bat\b", rest):
        flags += "+at"
    if " using " in rest:
        flags += "+using"
    if " with " in rest:
        flags += "+with"
    if "⟨" in rest:
        flags += "+anon"
    if ":=" in rest:
        flags += "+assign"
    if "<;>" in rest:
        flags += "+seq"
    arity_tag = f"#{arity}" + ("+" if arity == 4 else "")
    if not br and not flags:
        # bare head with term args (exact foo, apply foo) vs truly bare
        arity_tag = "" if rest.strip() else "#0"
    return f"{head}{arity_tag}{flags}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proofs", type=Path, default=Path("runs/exp_rf007_proofs"))
    parser.add_argument("--out", type=Path, default=Path("runs/promotion_candidates.json"))
    parser.add_argument("--min-theorems", type=int, default=2)
    args = parser.parse_args()

    form_theorems: dict[str, set] = defaultdict(set)
    bigram_theorems: dict[str, set] = defaultdict(set)
    n_proofs = 0
    for lean in sorted(args.proofs.rglob("Residual.lean")):
        text = lean.read_text()
        if ":= by" not in text:
            continue
        body = text.split(":= by", 1)[1]
        thm = lean.parent.name.split("-", 3)[-1]  # theorem identity for recurrence only
        n_proofs += 1
        forms = [normalize_step(s) for s in split_tactic_steps(body)]
        for f in forms:
            form_theorems[f].add(thm)
        for a, b in zip(forms, forms[1:]):
            bigram_theorems[f"{a} → {b}"].add(thm)

    # Population warrant: head-form share over the traced ground-truth corpus.
    pop = Counter()
    pop_total = 0
    for fname in ("data/nav_training.jsonl", "data/nav_eval.jsonl"):
        with open(fname) as f:
            for line in f:
                r = json.loads(line)
                tac = (r.get("ground_truth_tactic") or "").strip()
                if not tac:
                    continue
                pop_total += 1
                n = len(r.get("ground_truth_premises") or [])
                band = 0 if n == 0 else n if n < 4 else 4
                pop[f"{tac}#{band}" + ("+" if band == 4 else "")] += 1
                pop[tac] += 1  # head-only fallback key

    def pop_share(form: str) -> float:
        head = form.split("#")[0].split("+")[0]
        exact = pop.get(form.split("+")[0], 0)
        return round(max(exact, pop.get(head, 0)) / max(pop_total, 1), 4)

    candidates = []
    for form, thms in form_theorems.items():
        if len(thms) < args.min_theorems:
            continue
        candidates.append(
            {
                "form": form,
                "theorems": len(thms),
                "expressible": form in EXPRESSIBLE,
                "population_share": pop_share(form),
            }
        )
    candidates.sort(key=lambda c: (c["expressible"], -c["theorems"]))
    bigrams = sorted(
        ({"bigram": b, "theorems": len(t)} for b, t in bigram_theorems.items()
         if len(t) >= args.min_theorems),
        key=lambda x: -x["theorems"],
    )[:30]
    report = {
        "proofs_mined": n_proofs,
        "distinct_forms": len(form_theorems),
        "candidates": candidates,
        "top_bigrams": bigrams,
    }
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    absent = [c for c in candidates if not c["expressible"]]
    print(f"{n_proofs} proofs → {len(form_theorems)} distinct forms; "
          f"{len(candidates)} recurrent; {len(absent)} ABSENT from the current stack")
    for c in absent[:25]:
        print(f"  {c['theorems']:2} thms  pop {c['population_share']:6.2%}  {c['form']}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
