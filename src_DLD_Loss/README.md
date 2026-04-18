# LSTM-EML Hybrid: Arbitrary Precision Pi Calculation

> **An alternative we also tried — the Dual-Loss Decomposition (DLD).**
> A parallel implementation shipped in
> [`src_DLD_Loss/eml_lstm_pi.py`](https://github.com/Mastermindless/LSTM-EML-Operator-for-Pi/blob/main/src_DLD_Loss/eml_lstm_pi.py)
> keeps the same LSTM-drives-EML pattern but swaps the scale-homogeneous
> hinge for a **two-headed loss**: one head regresses the precision `P'` with
> a plain `MSELoss`, and a second head emits a classification distribution
> over the first 10 digits of π, trained with digit-wise
> `CrossEntropyLoss`. The total objective is
> `L = MSE(P', N) + CE(digits_pred, digits_true)`. The attraction is
> directness — the LSTM is supervised on *actual digits* rather than on a
> surrogate, which makes failure modes legible (wrong 7th digit is
> immediately visible in the confusion matrix). The cost is exactly the
> scale-homogeneity we were trying to buy: value-MSE on precision
> re-introduces the low-N/high-N gradient imbalance, and the fixed 10-digit
> classifier does not scale to N = 9 999 without a recurrent decoder. We
> treat DLD as a useful *diagnostic* companion to the main hinge loss —
> excellent for debugging the first 10 digits at low N, but not the path to
> extrapolation. The repository keeps both so readers can compare them
> side-by-side; the
> [`src_DLD_Loss/README.md`](https://github.com/Mastermindless/LSTM-EML-Operator-for-Pi/blob/main/src_DLD_Loss/README.md)
> documents the standalone variant.

## 🚀 Overview
This repository contains a **Neuro-Symbolic** implementation for calculating $\pi$ to arbitrary precision. It combines a **Long Short-Term Memory (LSTM)** network with the **EML (Exp-Minus-Log)** operator to create a system that learns to "trigger" symbolic mathematical evaluations.

### The EML Operator
The EML operator is defined as:
$$E(x, y) = e^x - \ln(y)$$
By nesting this operator with the constant 1, we can generate $\pi$ using the complex logarithm property:
$$\text{Im}(\ln(-1)) = \pi$$

## 🧠 Architecture
1.  **LSTM Controller**: Learns the mapping between a requested decimal precision and the internal state required to generate that precision.
2.  **Symbolic Node**: A high-precision evaluator using `mpmath` that executes the EML operation.
3.  **DLD Loss**: A custom loss function that optimizes both scalar precision alignment and digit-wise cross-entropy.

## 🛠 Installation
```bash
pip install torch mpmath numpy
```

## 📈 Usage
To train the model and run inference for 100 digits of $\pi$:
```bash
python3 eml_lstm_pi.py --train --epochs 5000 --precision 10000
```

To run inference on a pre-trained concept:
```bash
python3 eml_lstm_pi.py --precision 100
```

## 📂 Project Structure
- `eml_lstm_pi.py`: Core implementation (PyTorch).
- `concept_EML_LSTM_pi.md`: Detailed mathematical and architectural concept.
- `critic_EML_LSTM_pi.md`: Analysis of the model's strengths and limitations.
- `EML_operator.md`: background on the symbolic EML framework.

## 🌟 Future Goals
- Expand beyond $\pi$ to other transcendental numbers.
- Implement a fully differentiable EML tree search using Reinforcement Learning.
- Scalability to 1,000,000+ digits.

---
*This is a purely conceptual project exploring the intersection of deep learning and symbolic regression.*
