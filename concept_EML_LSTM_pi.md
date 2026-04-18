# Concept — LSTM + EML Neural Network for Pi at Arbitrary Precision

## 1. One-sentence goal
Train a tiny **LSTM** that learns to *trigger* the **EML** symbolic operator with the right
internal precision `dps` so that the EML outputs π matching a user-requested number of
significant digits N ∈ [1, 9999] (extendable to 99,999+).

```
  user request N ──►  LSTM  ──►  P' (precision)  ──►  EML(dps=P')  ──►  π̂
                                                                        │
                       true π@N (mpmath reference) ◄───────────── compare
```

The LSTM is the *controller*; the EML is the *symbolic computer*. Neither alone is
enough: the EML needs a precision hint, the LSTM has no numerical engine.


## 2. Components

### 2.1 EML operator (fixed, symbolic)
From the existing file `EML_operator.md`:
```
EML(x, y) = exp(x) − ln(y)          # binary operator
π emerges from 5-level nesting of EML(1,1) constructions plus
the complex-log identity ln(−1) = iπ.
```
In practice we use `mpmath.log(-1)` at `mp.dps = P'` as the concrete realisation.
The EML is frozen — **no learnable parameters**.

### 2.2 LSTM controller (trainable)
* Input tensor = *encoded request*. Two modes, toggled by a flag:
  * **shortcut mode (enabled = True)** : input = `[N_token] + digits_of_π@N`
    (LSTM sees the target length directly).
  * **hard mode (enabled = False)**    : input = `digits_of_π@N` only;
    the LSTM must *infer* N from the string length / content.
* Recurrent body: single-layer LSTM, hidden = 128, embedding size = 32,
  vocab = `{0–9, '.', <PAD>, <EOS>, <LEN_TOK>}` (13 tokens).
* Output head: a scalar precision prediction `P' = softplus(W h_T + b) + 1`.
  Positive and continuous → differentiable.

### 2.3 Pi reference generator
`mpmath.mp.dps = N + guard; str(mpmath.pi)` — truncated to N significant digits.
This is the supervision signal (ground-truth digit string).


## 3. Training sample generator

```
for step in range(num_steps):
    N        = random integer from curriculum range (see §6)
    target_π = true_pi(N)                     # mpmath π to N sig-figs
    input_x  = encode(N, target_π[: some visible prefix])
    yield (N, target_π, input_x)
```

The generator is cheap — a few ms per sample at N ≤ 9999.


## 4. The loss — precision-homogeneous across scales

### Problem statement
A naive MSE between mpf-Pi-values is catastrophic:
* absolute error at N = 5 is ≈ 10⁻⁵, at N = 9999 it is ≈ 10⁻⁹⁹⁹⁹
* the low-N samples dominate the gradient 10 000 orders of magnitude over
* mpmath values are also **not differentiable** — PyTorch cannot backprop through them.

We therefore do **not** differentiate through the EML. We build a
*surrogate analytical loss* over P' that has the same monotone structure as
real digit-agreement and is **scale-invariant in N**.

### 4.1 Surrogate (differentiable) loss
Let `P'` = LSTM output (continuous, ≥ 1), `N` = requested digits, `g` = safety guard (e.g. 4).

```
L_under(P', N) = ReLU( (N + g) − P' ) / (N + g)        # missing precision, normalised
L_over (P', N) = λ · ReLU( P' − (N + 2g) ) / (N + g)    # wasted precision
L_surr  (P', N) = L_under + L_over                       # in [0, 1]
```

Properties
* **Scale-homogeneous**: dividing by (N+g) gives values in [0, 1] for every N, so
  samples at N = 5 and N = 9999 contribute comparable gradients.
* **Asymmetric**: under-shooting is cheap *only* if tolerated (λ ≈ 0.1);
  over-shooting is discouraged to keep inference fast.
* **Piecewise-linear**, differentiable almost everywhere → plain SGD / Adam works.

### 4.2 Ground-truth digit-agreement (non-differentiable, used for monitoring / RL refinement)
```
d(P', N) = length of longest common prefix of EML(dps=⌊P'⌉) and true_pi(N),
           clipped to [0, N]
R(P', N) = d(P', N) / N       ∈ [0, 1]
```
This is the *reward* for an optional REINFORCE fine-tuning stage:
```
Δθ = (R − baseline) · ∇θ log π_θ(P' | input)
```
Stage 1 (surrogate) handles the heavy lifting; Stage 2 (policy gradient) eliminates
residual systematic bias.

### 4.3 Log-error formulation (analytic equivalence)
If `err = |π − EML(P')|` then `log10 err ≈ −P'`. Thus
```
L_norm_log  =  max(0, N − P') / N
```
is the expected value of `L_under` up to a constant — confirming the surrogate is
the *analytic* version of a log-relative-error loss. This justifies using it instead of
MSE on digit tensors.


## 5. Why an LSTM (and not a feed-forward regressor)?
* In **hard mode** the network sees only the π digit string and must count its length
  — a canonical sequential task.
* In **shortcut mode** the LSTM receives a `[LEN_TOK]` marker concatenated with the
  digits; this forms a mini language the recurrent state can parse.
* An LSTM is the smallest architecture that generalises to arbitrary-length inputs —
  this is essential if we later want to extrapolate N → 99 999+.
* We deliberately keep it small (~100 k parameters) so that *any* demonstrated learning
  clearly comes from the symbolic EML side, not from memorising π inside the net.


## 6. Training strategy — curriculum SGD

| Phase | N range           | steps    | notes                                  |
|------:|-------------------|---------:|----------------------------------------|
| 0     | 1 – 10            |   2 000  | warm-start, shortcut ON                |
| 1     | 1 – 100           |  10 000  | shortcut ON                            |
| 2     | 1 – 1 000         |  20 000  | shortcut ON                            |
| 3     | 1 – 9 999         |  50 000  | shortcut ON, add REINFORCE refinement  |
| 4     | 1 – 9 999         |  20 000  | shortcut **OFF** (hard mode)           |

* Optimiser: Adam, lr = 3e-4, cosine decay.
* Batch size: 32 (batches are cheap; mpmath is the bottleneck).
* Mix of N values *within each batch* to keep gradient stationary.
* Because `L_surr ∈ [0,1]` regardless of N, we never need to re-weight loss per phase.


## 7. Inference
```
n = int(input("Precision (digits of π): "))
P_star = round(model(encode(n)).item())
pi_hat = eml_pi(P_star)            # EML computation, not the LSTM
print("LSTM-chosen precision :", P_star)
print("EML π                 :", pi_hat[:n+2])
print("mpmath reference π     :", true_pi(n))
print("matching digits        :", common_prefix_len(pi_hat, true_pi(n)))
```

We explicitly overlay the standard mpmath π so users can visually confirm equality.


## 8. Deliverables
1. `eml_operator.py` — EML tree + `eml_pi(dps)` + `true_pi(n)`.
2. `pi_generator.py` — batched sample generator.
3. `lstm_eml_model.py` — `LSTM_EML` module with `use_length_input` toggle.
4. `train.py` — curriculum SGD loop, checkpointing, TensorBoard-friendly metrics.
5. `inference.py` — CLI: user enters N, sees LSTM precision, EML π, mpmath π, digit agreement.
6. `critic_EML_LSTM_pi.md` — counter-arguments + rebuttals.
7. `README.md` — wide-audience explainer + GitHub plan.


## 9. Success criteria
* ≥ 99 % of N ∈ [1, 9999] produce EML output whose first N digits match mpmath π.
* The LSTM trained with `use_length_input=False` (hard mode) achieves the same on an
  unseen range N ∈ [5 000, 9 999] — proving the LSTM *counts digits* rather than
  memorising a number.
* Inference at N = 9 999 completes in < 1 s on CPU (mpmath-dominated).

This is a *conceptual* experiment: the point is not to beat mpmath at computing π,
but to demonstrate that a recurrent neural net can steer a **symbolic** operator
(EML) at **arbitrary precision**, opening the door to hybrid LSTM-symbolic
regressions for harder constants, identities, and eventually general SR tasks.
