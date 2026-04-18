# LSTM-EML Hybrid: Arbitrary Precision Pi Calculation

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
