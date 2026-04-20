# LSTM-EML π  —  A Recurrent Network that Drives a Symbolic Operator

> *A small LSTM learns to dial the precision knob of a symbolic **EML** operator,
> so that a hybrid neural / symbolic system outputs **π to any requested number
> of digits** — from 5 up to 9 999 and beyond.*

---

## TL;DR for a wide audience

Think of the EML operator as a tiny hand-cranked calculator that can, in
principle, compute anything — but only if you tell it how many decimal places
to keep. The hand crank is an integer called `dps` (decimal places).
Ask for `dps = 5` and you get π ≈ 3.1416. Ask for `dps = 9999` and you get
π to 9 999 digits.

We train a **small LSTM neural network** to look at your request ("I want
N digits of π") and turn the hand crank to exactly the right number of turns.
The LSTM never computes π itself — the **symbolic EML operator does the math**.
The neural net is the *controller*, the symbolic operator is the *engine*.

That split is the point. The same LSTM controller could later dial symbolic
operators for `e`, `sin`, `cos`, `tanh`, or whole expression trees — we picked
π as the first demonstration because it has a beautiful derivation via
`ln(-1) = iπ` that the EML can express with nothing but `exp`, `log`, and the
constant 1 (see Odrzywołek 2026, included in this repo).

**The scientific question**: can a recurrent network *generalise* the
precision-selection rule across four orders of magnitude (N ∈ [1, 9999]), and
eventually infer the requested precision purely by looking at the length of the
π-string we feed it? This repo answers that experimentally.

---

## What is in this repository

| File                           | Role                                                        |
|--------------------------------|-------------------------------------------------------------|
| `EML_operator.md`              | Background on the EML (exp(x) − log(y)) operator            |
| `concept_EML_LSTM_pi.md`       | **Full technical concept**: architecture, loss, curriculum  |
| `critic_EML_LSTM_pi.md`        | Adversarial review + rebuttals                              |
| `eml_operator.py`              | EML tree + `eml_pi(dps)` + mpmath reference `true_pi(n)`    |
| `pi_generator.py`              | Random training-batch generator, tokenizer, shortcut toggle |
| `lstm_eml_model.py`            | `LSTM_EML` PyTorch module + scale-homogeneous loss          |
| `train.py`                     | Curriculum SGD trainer (5 phases, optional REINFORCE)       |
| `inference.py`                 | Interactive / single-shot CLI, overlays mpmath π            |

---

## Method in one picture

```
   N   (user request, 1…9999)
   │
   ▼
 [tokenize]──►  LSTM  ──►  P' ∈ ℝ₊   (predicted precision)
                                │
                         round│
                                ▼
                     EML(dps=P*) ──► π̂  (digit string)
                                │
        mpmath π@N ─────► compare (digit prefix length)
```

- **LSTM**: 1-layer, hidden = 128, ~100 k parameters. Outputs a single positive
  scalar.
- **EML**: `exp(x) − log(y)` binary node + constants; the concrete π comes via
  `Im(log(-1))` evaluated at `mpmath.mp.dps = P*`.
- **Loss**: scale-homogeneous hinge on P'
  ```
  L = ReLU((N+g) − P') / (N+g)  +  λ · ReLU(P' − (N+2g)) / (N+g)
  ```
  (see `concept_EML_LSTM_pi.md` §4). This avoids the MSE-across-10000-orders-of-magnitude
  pitfall and is differentiable almost everywhere.
- **Optional REINFORCE** stage uses the real digit-agreement reward
  `R = matching_digits / N` to polish the surrogate.

---

## Install & run

```bash
pip install torch mpmath

# smoke-test (a few seconds)
python train.py --quick

# full curriculum (~240 min on 5 ARM CPU M3)
python train.py

# interactive inference — compare LSTM-EML π vs mpmath π
python inference.py --ckpt lstm_eml_pi.pt

# single shot, hard mode (N hidden from LSTM, must be inferred from length)
python inference.py --ckpt lstm_eml_pi.pt --n 9999 --hard
```

Expected terminal output:
```
Requested precision N          : 9999
Mode                           : shortcut (N shown to LSTM)
LSTM precision (raw)           : 10003.124
LSTM precision (rounded P*)    : 10003
Matching digits vs. mpmath π   : 9999/9999  ✔ perfect
```

---

## Key design choices (why these, not others)

- **Scale-homogeneous hinge loss, not MSE.** Digit-level absolute error spans
  10⁻⁵ to 10⁻⁹⁹⁹⁹ — naive MSE would let low-N samples crush the gradient.
  Normalising by `(N + guard)` puts every sample in [0, 1].
- **EML outside the autograd graph.** mpmath is not differentiable; instead of
  hacking straight-through estimators, we make the surrogate loss operate
  directly on the LSTM's continuous output, and use REINFORCE only as an
  optional polish.
- **Shortcut toggle.** Early training shows the LSTM `N` explicitly
  (`use_length_input=True`). Late training hides it so the LSTM must *count the
  Pi digits*. Both modes share one codebase.
- **Curriculum, not uniform sampling.** Starting with N ≤ 10 lets the net
  discover the `P' ≈ N + 4` relationship cheaply before the generator loads
  expensive high-N mpmath strings.
- **LSTM (not feed-forward)**. Hard mode is a sequence-length task; a recurrent
  model handles variable-length inputs without retraining.

---

## Critic / honest limitations

Please read `critic_EML_LSTM_pi.md` — it enumerates every reasonable objection
(MSE ill-posedness, circularity of `log(-1)`, triviality of the shortcut mode,
sequence-length extrapolation) and our rebuttals. Headline caveats:

1. mpmath internally already knows π — the EML construction is a *symbolic
   interface*, not a new π algorithm.
2. The shortcut result is intentionally trivial; the scientific claim lives in
   *hard mode*.
3. Extrapolation beyond the training envelope (N > 9 999) is an open question
   we report honestly.

---

## Roadmap / GitHub plan

1. **v0.1 (this repo)**: π demonstration, 1–9 999 digits, shortcut ON/OFF.
2. **v0.2**: same controller, different symbolic constants (`e`, Catalan,
   ln 2) — show the LSTM transfers.
3. **v0.3**: multi-head LSTM that drives **arbitrary EML expression trees** —
   i.e. genuine symbolic regression with learned precision.
4. **v1.0**: paper + benchmarks; compare against direct seq2seq π generation
   (expected to fail at high N) and against hand-tuned `dps = N + 4`.

**GitHub structure** (to publish):

```
lstm-eml-pi/
├── README.md                  <- this file
├── concept_EML_LSTM_pi.md
├── critic_EML_LSTM_pi.md
├── src/
│   ├── eml_operator.py
│   ├── pi_generator.py
│   ├── lstm_eml_model.py
│   ├── train.py
│   └── inference.py
├── src_mps/ # for macbook gpu
│   ├── eml_operator.py
│   ├── pi_generator.py
│   ├── lstm_eml_model.py
│   ├── train.py
│   └── inference.py
├── src_DLD_Loss/
│   ├── eml_lstm_pi.py
│   └── README.md
├── figures/
│   ├── architecture.png       <- the block diagram above, rendered
│   ├── loss_landscape.png     <- surrogate loss vs. P' at various N
│   ├── curriculum_curve.png   <- training loss across phases
│   └── digit_recall.png       <- perfect-match rate vs. N (shortcut vs. hard)
└── LICENSE                    <- MIT
```

---

## License

MIT. The EML operator code in is adapted from the original paper's reference implementation.

---

## Citation

If you use this hybrid LSTM-EML controller, please cite:

- Odrzywołek, A. (2026). The EML Operator and Elementary Functions. Andrzej Odrzywołek, All elementary functions from a single operator: eml⁡(x,y) = exp⁡(x)−ln⁡(y), https://arxiv.org/html/2603.21852v2
- Hochreiter S., Schmidhuber J.: Long Short-Term Memory. In: Neural Computation. 9. Jahrgang, Nr. 8, 1. November 1997, ISSN 0899-7667, S. 1735–1780, doi:10.1162/neco.1997.9.8.1735, PMID 9377276 (englisch).

## AI Disclosure

Claude Opus 4.7, Gemini 3.1
