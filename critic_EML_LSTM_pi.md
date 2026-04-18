# Critic — Counter-arguments to the LSTM-EML π approach
_A deliberately adversarial read-through. Each section states the objection, then
the rebuttal we will put in the paper / README._

---

## C1. "The LSTM is learning the trivial identity P' ≈ N + guard."
**Objection.** For the surrogate loss proposed in §4.1, the global minimum is
`P' = N + g` (shortcut mode) or some monotonic function of `|π digits|` (hard mode).
A *linear* model `P' = N + 4` achieves zero loss. Hence the LSTM is ceremonial.

**Rebuttal.** True — and *intended* for Phase 1. The demonstration target is
(i) that a **recurrent network** can reliably steer a symbolic operator across
four orders of magnitude, (ii) that in **hard mode** (length hidden) the LSTM
has to *count / parse* the input, which linear regression cannot do. The triviality
in shortcut mode is the upper baseline; hard mode is the actual scientific claim.
The project is explicit that this is a **conceptual proof-of-concept** for LSTM↔symbolic
coupling — simplicity of the target is a feature.

---

## C2. "The EML does not actually derive π — mpmath already knows π."
**Objection.** `mpmath.log(-1)` internally evaluates `mpc(0, mp.pi)`. Since
`mp.pi` is pre-computed using Machin-like series, the "EML = exp(x) − log(y)"
construction is syntactic dressing over mpmath's built-in π constant.

**Rebuttal.** Correct — and unavoidable for *finite* nesting. The Odrzywołek paper
shows π emerges from infinitely deep EML(1,1) nests; any implementation must
truncate and defer to an arbitrary-precision log. We acknowledge this in the README
("EML is a symbolic *interface*, the numerical engine is mpmath") and cite the
paper's theoretical derivation separately from our practical realisation. The
neural-network claim is about *controlling* the precision knob, not replacing
mpmath's series expansion.

---

## C3. "MSE on π values is ill-posed."
**Objection.** `|π@5 − π@9999| ≈ 10⁻⁵` — the low-N samples would dominate training
by many orders of magnitude if MSE on the value were used.

**Rebuttal.** Agreed — we do **not** use value-MSE. §4.1 of the concept uses a
scale-normalised hinge on P' itself. The analytic equivalence to the log-relative
error (§4.3) is exactly what makes the loss homogeneous across N ∈ [1, 9999].

---

## C4. "mpmath is not differentiable — backprop through the EML is impossible."
**Objection.** You cannot take gradients of `mpmath.log` w.r.t. `dps`; `dps` is
an integer anyway.

**Rebuttal.** Correct. We sidestep the issue: the differentiable surrogate acts
on the LSTM's continuous P' directly (no EML in the forward graph). The EML is
only invoked for **monitoring** (digit-agreement reward) and for **optional
REINFORCE fine-tuning**, where the log-prob of the sampled P' provides the
gradient path. At no point do we pretend mpmath is differentiable.

---

## C5. "Integer vs. continuous mismatch."
**Objection.** EML needs an integer `dps`, but the LSTM outputs a continuous `P'`.
Rounding breaks the gradient.

**Rebuttal.** The surrogate operates on the continuous output; rounding only
happens at inference / reward evaluation. Straight-through estimation is not
required because the forward loss is already continuous.

---

## C6. "Why an LSTM? A scalar regressor on `log N` would suffice."
**Objection.** Both input and output are essentially one scalar. Recurrence is wasted.

**Rebuttal.** True *only* in shortcut mode. In hard mode the network sees a
variable-length string "3.14159…" and must infer N. That is a sequence task.
The LSTM is the minimal architecture that handles both modes with one codebase,
and supports seamless extrapolation to longer strings without retraining.

---

## C7. "Generalisation to N > 9999 is not demonstrated by training on N ≤ 9999."
**Objection.** Recurrent networks notoriously fail length-extrapolation.

**Rebuttal.** Accepted risk. We report hard-mode extrapolation explicitly
(train on N ≤ 1000, test on N ∈ [5000, 9999]) as a diagnostic. If extrapolation
fails beyond the training envelope, we will add positional-length encodings and
a log-N conditioning scalar to the LSTM input — standard remedies.

---

## C8. "Inference speed at N = 9999 is bounded by mpmath, not the net."
**Objection.** The LSTM contributes < 1 ms; mpmath dominates. The neural net
offers no performance advantage.

**Rebuttal.** This is not a performance paper. The LSTM's job is to select
precision; speed is a bonus feature of the EML back-end. The research question is
whether a differentiable controller can be coupled to a symbolic operator at all.

---

## C9. "Why not just skip the EML and make the LSTM output π digits directly?"
**Objection.** A sequence-to-sequence model trained on "N → π@N" does the same
job end-to-end.

**Rebuttal.** It would (a) require memorising π inside the weights, which does
not scale beyond the training horizon, and (b) lose interpretability.
The whole point of the hybrid is that the EML provides a **correct, extensible,
arbitrary-precision** computation, and the LSTM only has to choose *how much* of
it to run. This is the decomposition we want to generalise to other symbolic
regressions.

---

## C10. "The shortcut flag makes the claim unfalsifiable."
**Objection.** With shortcut ON it's a trivial reformat; with shortcut OFF the
model will fail. Either way the claim 'LSTM learns π precision' is defensible
after the fact.

**Rebuttal.** We pre-register the evaluation: final report numbers are **hard-mode
on held-out N-ranges**. Shortcut mode numbers are reported only as an *upper bound*
(ceiling analysis). The flag is pedagogical — it lets readers see the problem
decompose cleanly.

---

## C11. "Stochastic noise in REINFORCE will be enormous at high N."
**Objection.** When N = 9999 and P' is off by even 1, the reward is near zero —
credit assignment is noisy.

**Rebuttal.** Yes — that is why REINFORCE is only Phase-3 refinement, not the
primary training signal. The surrogate loss gets P' into the correct neighbourhood
(within ±5 of the optimum); policy gradient polishes it. We use a running-mean
baseline and small σ (= 1–2 digits).

---

## C12. "Not actually symbolic regression."
**Objection.** Symbolic regression searches over expression trees. Here the EML
tree is fixed; only a scalar knob is learned. Calling this "LSTM-EML symbolic
regression" oversells it.

**Rebuttal.** Fair framing critique. We will retitle the public-facing summary as:
> *"Hybrid LSTM–Symbolic controller: a first step toward learning to drive
>  EML-based symbolic operators."*
Full tree-search SR remains future work; this paper is Stage 1 of a larger
programme (see README §Roadmap).

---

## Summary
The concept is **honestly modest**: it is a controllable hybrid, not a replacement
for mpmath, not a new π algorithm, and not full symbolic regression. Its value is
the clean template for coupling an LSTM to a symbolic operator with a
scale-invariant, differentiable loss — something that can be extended to `e`,
`sin`, `cos`, `tanh`, and eventually general EML-expression synthesis.
