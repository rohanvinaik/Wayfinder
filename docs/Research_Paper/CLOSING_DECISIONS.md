# The Closing Decisions

**Status:** open ledger — seven decisions to close out the multi-stage thesis, taken one at a
time, each marked when decided and again when done. The plain-language framing is the primary
text; the technical anchor is the footnote. (House rule: deep-theory decisions are thought
through conversationally; rigor lives in the artifacts that come out the other side.)

**How we got here:** the deterministic system proves 1,042 of 2,000 theorems with no neural
network anywhere, every proof checkable, every run reproducible to the byte. The neural
version scored 1,277 — so we kept ~80% of the performance and threw away the black box. What's
left is deciding what "finished" means.

---

## D1 — Teach Dr. Ducky to check that pieces fit before trying them

**The plain question.** When a person tries "apply this lemma to that fact," they glance first:
does this lemma even *take* that kind of thing? Dr. Ducky currently doesn't glance — she tries
every lemma against every fact and lets Lean reject the nonsense, which we measured at ~100%
rejection across ~3,000 attempts. Her own paper says she should check fit first; we never built
the checking.

**Recommendation:** build it. Ask the kernel once per lemma what it takes (cacheable), only
pair things whose shapes match. It's the last place the code does less than the theory says.

*Anchor: Dr. Ducky paper §5.5 "deterministic compatibility rules"; pass-3c/3d telemetry
(2,977 plan executions, 3 progresses); env_inspect signature fetch at binding-space build.*

**Decided:** ☐  **Done:** ☐

---

## D2 — What to do with the fourth layer we never built (the second-order SoM)

**The plain question.** The original plan had a trained model at the top of the stack, learning
from the residuals. Since then, the whole point sharpened: the thinking is the deterministic
part; anything learned is just a hunch about what to try first — allowed to suggest an order,
never allowed to decide. Do we build the hunch-machine now, or close the thesis without it?

**Recommendation:** close without it. "1,042 proofs with zero learned parameters" is the
demonstration; bolting on a trained orderer now blurs the line the whole project exists to
draw. It stays specified in FUTURE_DIRECTIONS — an escape hatch, on Rohan's schedule.

*Anchor: THE_REFOUNDING §4.3 (statistical = ordering over fenced spaces); the packet corpus
(rendered cheat sheets) as its training data if ever built.*

**Decided:** ☐  **Done:** ☐

---

## D3 — Agree on how we count

**The plain question.** Some theorems flicker: change anything about the order of attempts and
three drop out while a different one drops in. But a checked proof is a checked proof forever —
once we've proven something, it's banked. So the honest score is "everything we've ever proven"
(the union), and single-run wobbles of ±3 are noise about ordering, not evidence about the
machinery.

**Recommendation:** write union-accounting into the canon as the loop's official measure, with
the flicker band documented so nobody (including us) mistakes wobble for signal.

*Anchor: §7.3 metric extension; the measured ±3 band across passes 2/3/3b/3d; current union
1,042/2,000.*

**Decided:** ☐  **Done:** ☐

---

## D4 — Make the "stale fences" lesson a rule the code enforces

**The plain question.** We learned the hard way that a list of "things that never work" is only
true for the version of the system that generated it — change the prover, and yesterday's
never-works can become today's winner (that cost us 4 proofs once). The lesson is written down;
lessons written down get forgotten. Make the code check: the fence list remembers which version
mined it, and complains when the prover has changed underneath it.

**Recommendation:** implement — store records its mining commit; gate warns on mismatch; every
re-mine re-runs the safety check.

*Anchor: §10.8; censor store metadata `mined_from` + code commit; the paired delta ≥ 0 check.*

**Decided:** ☐  **Done:** ☐

---

## D5 — Pay the three measurement debts

**The plain question.** Three promised checks were never run: (1) do our emitted proofs
actually compile on the newer Lean, end to end? (2) the one theorem this system proved about
its own math — audit that its proof doesn't smuggle in extra axioms; (3) rerun the 21 novel
proofs from before the refit through everything we've built since — the cleanest before/after
we own. A thesis whose motto is "every claim measured" can't close with unmeasured claims.

**Recommendation:** run all three before declaring closure.

*Anchor: EXP-RF-006 `--verify`; `#print axioms` on `bridge_curvature_bound`; EXP-RF-021.*

**Decided:** ☐  **Done:** ☐

---

## D6 — Don't move the goalposts (the norms question)

**The plain question.** All our "what's typical for a goal" baselines were measured on the full
corpus. The leftover hard theorems look different from typical. Should we re-baseline to the
leftovers? Instinct says careful: re-baselining silently re-keys every fence and coordinate at
once. Better to measure how different the leftovers are (a report), and only re-baseline if we
formally start over on a new corpus.

**Recommendation:** measure, don't re-base.

*Anchor: SurfaceNorms/ContextNorms mined 2026-08-24 from nav_eval; §10.7 population
re-estimation; re-basing = only with a formal re-founding.*

**Decided:** ☐  **Done:** ☐

---

## D7 — Open the next universe (problems, not just proofs)

**The plain question.** The original hunch was never really about proofs. It was: people solve
hard problems by figuring out what *kind* of problem it is, then reaching for the tools that
kind needs — Einstein didn't reinvent calculus, he thought about trains and clocks and combined
what existed. Proofs were the perfect practice arena because Lean tells you instantly whether
you're right. Once D1–D6 close, the arena is complete — and the real move is pointing the same
machinery at open-ended math problems, where "what kind of thing is this and what tools does
that kind take" is the whole game.

**Recommendation:** this IS the closure. Declare the proof instrument finished, re-issue
THE_REFOUNDING with every experiment carrying its number, and begin the transport.

*Anchor: Rohan's March note (problem-solving above proofs); the corpus's transport record
(program identity → text meaning → proofs → …); same reads, same two signs, same loop.*

**Decided:** ☐  **Done:** ☐

---

*Order of attack: chosen one at a time, conversationally, starting wherever Rohan points.*
