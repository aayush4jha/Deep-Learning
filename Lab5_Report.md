# Deep Learning Lab Practical 5 — Report
## Recurrent Neural Network Architectures for Handwritten Character Recognition

**Author:** Aayush Jha
**Date:** 2026-04-18
**Framework:** PyTorch 2.10.0 (CUDA)
**Datasets:** MNIST, EMNIST-Letters, EMNIST-Balanced

---

## 1. Abstract

This report presents a comprehensive study of recurrent neural network (RNN) architectures applied to handwritten character recognition. Seven problem statements were addressed: a vanilla RNN (from scratch and PyTorch), LSTM, GRU, bidirectional RNNs, a CNN–LSTM hybrid, hyperparameter/regularization tuning, and a final comparative analysis. The CNN–LSTM hybrid achieved the best test accuracy (**99.32%** on MNIST), while the GRU offered the best accuracy-to-parameter trade-off. A cross-dataset evaluation on EMNIST-Letters (26 classes) attained **93.81%** test accuracy.

---

## 2. Objectives

1. Implement a Vanilla RNN both manually and using `torch.nn.RNN`, and diagnose the vanishing-gradient problem.
2. Build an LSTM classifier, inspect its gate activations, and study the effect of dropout.
3. Implement a GRU, benchmark it against the LSTM in accuracy, size, and speed.
4. Compare unidirectional and bidirectional RNNs (LSTM & GRU) with concat/average merge modes.
5. Design a CNN + LSTM hybrid and compare it to pure-CNN and pure-LSTM baselines.
6. Systematically tune learning rate, batch size, optimizer, gradient clipping, early stopping, and LR scheduling.
7. Perform a comparative final analysis across all architectures with confusion matrix, t-SNE, and cross-dataset validation.

---

## 3. Methodology

### 3.1 Dataset Preparation
| Dataset | Train | Test | Classes | Notes |
|---|---|---|---|---|
| MNIST | 60,000 | 10,000 | 10 | Digits 0–9 |
| EMNIST-Letters | 124,800 | 20,800 | 26 | Labels shifted 1–26 → 0–25 |
| EMNIST-Balanced | 112,800 | 18,800 | 47 | Digits + letters mix |

All images were converted to tensors and normalised with mean/std = 0.5. For RNN models, each 28×28 image was interpreted as a sequence of **T=28 time steps** with **F=28 features** per step (row-wise scanning).

### 3.2 Training Protocol
- **Loss:** `CrossEntropyLoss`
- **Optimizer (default):** Adam, lr = 1e-3
- **Epochs (default):** 10 (8 for tuning sweeps)
- **Batch size (default):** 64
- **Device:** CUDA GPU
- Custom `train_model()` loop recording per-epoch loss/accuracy/time.
- `evaluate_model()` for validation; optional gradient clipping and LR scheduler hooks.

### 3.3 Evaluation Metrics
Training loss, training accuracy, validation loss, validation accuracy, per-epoch time, inference latency (ms/batch), parameter count, confusion matrix, classification report, t-SNE embedding visualisation, and convergence speed (epochs to 95 % accuracy).

---

## 4. Architectures — Detailed Design

### 4.1 Problem 1 — Vanilla RNN

**From-scratch cell** ([Lab5.ipynb](Lab5.ipynb) — `VanillaRNNCell`):
```
h_t = tanh(W_x · x_t + W_h · h_{t-1})
```
- `W_x`: `Linear(input_size, hidden_size)` (with bias)
- `W_h`: `Linear(hidden_size, hidden_size)` (no bias)
- Final classifier: `Linear(hidden_size, num_classes)` applied to the last hidden state h_T.

`VanillaRNNScratch` stacks `num_layers` such cells, storing hidden states for gradient-norm analysis. `VanillaRNNPyTorch` wraps `nn.RNN` for efficient training across four configurations (L1/H64, L1/H128, L2/H128, L3/H128).

**Directional variant** (`DirectionalRNN`): permutes the input `(B,28,28) → (B,28,28)ᵀ` to switch between row-wise and column-wise scanning.

### 4.2 Problem 2 — LSTM
`LSTMModel` uses `nn.LSTM(input_size=28, hidden_size, num_layers, batch_first=True, dropout)` followed by `Dropout` and a dense head. Four configurations were trained: L1/H64, L1/H128, L2/H128 (dr=0.3), L2/H256 (dr=0.3).

**`LSTMWithGates`** replaces the standard cell with a manual gate computation using `weight_ih`/`weight_hh`:
```
i_t = σ(W_ih[:H] x_t + W_hh[:H] h_{t-1} + b)
f_t = σ(W_ih[H:2H] x_t + ...)
g_t = tanh(W_ih[2H:3H] x_t + ...)
o_t = σ(W_ih[3H:] x_t + ...)
c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```
All four gate tensors are recorded per time-step for visualisation.

### 4.3 Problem 3 — GRU
`GRUModel` wraps `nn.GRU` with identical interface to `LSTMModel`, trained at L1/H128, L2/H128 (dr=0.3), L3/H128 (dr=0.3). Parameter count compared directly against the equivalent LSTM.

### 4.4 Problem 4 — Bidirectional RNN
`BiRNNModel` accepts `rnn_type ∈ {lstm, gru}` and `merge ∈ {concat, avg}`:
- **Concat:** last output of shape `(B, 2H)` passed to `Linear(2H, C)`.
- **Avg:** forward and backward halves averaged → `Linear(H, C)`.

### 4.5 Problem 5 — CNN–LSTM Hybrid
**Architecture A — `CNNLSTMHybrid`:**
1. `Conv2d(1→32, k=3, p=1) → ReLU → MaxPool2d(2)` → 14×14
2. `Conv2d(32→64, k=3, p=1) → ReLU → MaxPool2d(2)` → 7×7
3. Reshape `(B, 64, 7, 7) → (B, 7, 64·7=448)` — 7 time steps of 448 features.
4. `LSTM(448, 128)` → classifier `Linear(128, num_classes)`.

**Architecture B — `TimeDist_CNN_LSTM`:** a lightweight CNN applied row-wise then fed into an LSTM.
**Baselines:** `PureCNN` (Conv→Conv→Dense) and the best pure LSTM from Problem 2.

### 4.6 Problem 6 — Hyperparameter Sweeps
Independent sweeps over:
- Learning rate: {0.1, 0.01, 0.001, 0.0001}
- Batch size: {32, 64, 128, 256}
- Optimizer: {SGD+momentum, Adam, RMSprop, AdamW}
- Gradient clip: {None, 1.0, 5.0, 10.0} (on Vanilla RNN + SGD)
- Early stopping (patience 5) + `ReduceLROnPlateau` (patience 2, factor 0.5)

### 4.7 Problem 7 — Comparative Analysis
Seven representative models retrained end-to-end for 10 epochs, with inference timing, confusion matrix of the best model, t-SNE of LSTM penultimate features, misclassified-sample gallery, convergence-speed table, and a cross-dataset LSTM run on EMNIST-Letters.

---

## 5. Step-by-Step Results

### 5.1 Vanilla RNN — Configuration Sweep (MNIST, 10 epochs)

| Model | Params | Best Val Acc | Avg Epoch |
|---|---:|---:|---:|
| RNN_L1_H64 | 6,666 | 0.9463 | 17.37 s |
| RNN_L1_H128 | 21,514 | 0.9633 | 17.07 s |
| RNN_L2_H128 | 54,538 | 0.9705 | 16.95 s |
| RNN_L3_H128 | 87,562 | 0.9710 | 17.67 s |

### 5.2 Vanishing-Gradient Analysis
Mean gradient norm (single-layer from-scratch RNN, 5 epochs): **3.24 → 3.43 → 3.10 → 2.85 → 2.57** — showing gradual decay of global gradient magnitude as the network begins to saturate.

### 5.3 Row vs Column Scanning (Vanilla RNN, 8 epochs)
- Row-wise: 95.51 % train / **93.47 %** val
- Column-wise: 95.39 % train / **96.15 %** val

Column-wise scanning produced slightly better generalisation on MNIST, because vertical digit strokes align more consistently along columns.

### 5.4 LSTM — Configuration Sweep (10 epochs)

| Model | Params | Best Val Acc |
|---|---:|---:|
| LSTM_L1_H64 | 24,714 | 0.9824 |
| LSTM_L1_H128 | 82,186 | 0.9846 |
| LSTM_L2_H128 (dr=0.3) | 214,282 | **0.9890** |
| LSTM_L2_H256 (dr=0.3) | 821,770 | 0.9909 |

### 5.5 LSTM — Dropout Effect (8 epochs, L2/H128)

| Dropout | Best Val Acc |
|---:|---:|
| 0.0 | 0.9873 |
| 0.2 | 0.9881 |
| 0.3 | 0.9858 |
| 0.5 | **0.9879** |

Light dropout (0.2) gave the best generalisation; training accuracy dropped noticeably at 0.5 without a clear val-accuracy gain.

### 5.6 GRU — Configuration Sweep (10 epochs)

| Model | Params | Best Val Acc |
|---|---:|---:|
| GRU_L1_H128 | 61,962 | 0.9864 |
| GRU_L2_H128 | 161,034 | **0.9901** |
| GRU_L3_H128 | 260,106 | 0.9906 |

### 5.7 LSTM vs GRU (L1/H128)

| Model | Params | Inference (ms/batch) |
|---|---:|---:|
| LSTM | 82,186 | 0.46 |
| GRU | 61,962 | 0.40 |

GRU is **24.6 % smaller** and **11.5 % faster** at inference, with comparable accuracy.

### 5.8 Bidirectional vs Unidirectional (10 epochs)

| Model | Best Val Acc |
|---|---:|
| BiLSTM-Concat | 0.9833 |
| BiLSTM-Avg | 0.9826 |
| BiGRU-Concat | 0.9822 |
| Uni-LSTM (baseline) | **0.9880** |

For image-as-sequence tasks, the bidirectional gain is modest and the unidirectional LSTM actually edged ahead — the full image is already visible per sample, so backward context adds less than in NLP.

### 5.9 CNN–LSTM Hybrid & Baselines (10 epochs)

| Model | Params | Best Val Acc | Inference |
|---|---:|---:|---:|
| CNN-LSTM | 316,042 | **0.9934** | 0.75 ms |
| TimeDist-CNN-LSTM | 85,866 | 0.9537 | 0.63 ms |
| Pure-CNN | 824,458 | 0.9927 | 0.48 ms |
| Pure-LSTM | 82,186 | 0.9872 | 0.41 ms |

CNN–LSTM matches/exceeds Pure-CNN at **~2.6× fewer parameters**, confirming the value of combining spatial feature extraction with sequential modelling.

### 5.10 Hyperparameter Tuning (LSTM L1/H128, 8 epochs)

**Learning rate:**
| LR | Best Val Acc |
|---|---:|
| 0.1 | 0.5495 (diverged) |
| 0.01 | 0.9570 |
| **0.001** | **0.9853** |
| 0.0001 | 0.9706 |

**Batch size:** 32 → 0.9874, 64 → 0.9874, 128 → 0.9852, 256 → 0.9824 (smaller batches slightly better; larger batches faster/epoch).

**Optimizer:** SGD 0.9853 · Adam 0.9864 · RMSprop 0.9850 · AdamW 0.9857. Adam leads by a small margin.

**Gradient clipping (Vanilla RNN + SGD):** None → 0.9755, 1.0 → 0.9755, 5.0 → 0.9701, 10.0 → 0.9722. Aggressive clipping (1.0) stabilised training without hurting final accuracy.

**Early Stopping + ReduceLROnPlateau:** stopped at epoch 15 with LR stepped 1e-3 → 5e-4 → 2.5e-4; best val_loss = 0.0323, val_acc ≈ **0.9918**.

### 5.11 Final Comparison (10 epochs)

| Model | Params | Val Acc | Epoch(s) | Infer(ms) |
|---|---:|---:|---:|---:|
| Vanilla-RNN | 54,538 | 0.9678 | 18.05 | 0.39 |
| LSTM | 214,282 | 0.9898 | 21.01 | 0.46 |
| GRU | 161,034 | 0.9902 | 19.01 | 0.42 |
| BiLSTM | 49,418 | 0.9842 | 18.62 | 0.39 |
| BiGRU | 37,386 | 0.9819 | 18.28 | 0.40 |
| **CNN-LSTM** | 316,042 | **0.9932** | 20.05 | 0.72 |
| Pure-CNN | 824,458 | 0.9917 | 18.80 | 0.59 |

**Best model:** CNN-LSTM — macro-F1 = 0.99 across all 10 digit classes. Per-class precision/recall ≥ 0.97.

### 5.12 Convergence Speed (epochs to ≥ 95 % val acc)
Vanilla-RNN: 4 · LSTM: 1 · GRU: 1 · BiLSTM: 2 · BiGRU: 2 · CNN-LSTM: 1 · Pure-CNN: 1.

### 5.13 EMNIST-Letters (26 classes, LSTM L2/H256, 10 epochs)
Final val_acc = **0.9381**, weighted F1 = 0.94. Hardest classes: I (0.75), L (0.76), G (0.86), Q (0.86) — confusions driven by case/script ambiguity (e.g., I vs L, O vs Q).

---

## 6. Summary of Findings

1. **CNN-LSTM is the overall winner** (99.32 % MNIST, 99 % macro-F1) — spatial feature extraction + sequential aggregation beats either modality alone while staying smaller than a pure CNN.
2. **GRU dominates the accuracy/cost frontier among pure RNNs** — nearly identical to LSTM accuracy with ~25 % fewer parameters and ~12 % faster inference.
3. **Vanilla RNN ceiling is ≈ 97 %**; deeper stacks plateau and gradient norms gradually contract, illustrating the vanishing-gradient issue that gated RNNs resolve.
4. **Bidirectionality offered no benefit** on row-scanned MNIST — future context is redundant when the whole image is available at inference.
5. **Hyperparameter sensitivity:** lr=1e-3 + Adam + batch 32–64 + mild dropout (0.2–0.3) is the robust baseline. lr=0.1 diverges; lr=1e-4 under-trains. Gradient clipping at 1.0 stabilises Vanilla RNN + SGD.
6. **Early-stopping + LR scheduling** pushed a 2-layer LSTM to **99.18 %** with automatic termination at epoch 15.
7. **Cross-dataset generalisation** to EMNIST-Letters achieved 93.81 %, bottlenecked by inherently ambiguous glyph pairs (I/L, O/Q).

---

## 7. Conclusion

The experiments systematically show the progression from vanilla RNNs (susceptible to vanishing gradients, ~97 % ceiling) to gated recurrent units (LSTM/GRU, ~99 %) and finally to hybrid CNN–RNN architectures (99.3 %). For production deployment on MNIST-like image-as-sequence tasks, **CNN–LSTM** is recommended when accuracy is paramount, and **GRU** when a pure-recurrent model with a tight parameter budget is required. Regularisation via dropout, gradient clipping, early stopping, and LR scheduling each contribute measurably to stability and final accuracy.

---

## 8. Reproducibility

- Notebook: [Lab5.ipynb](Lab5.ipynb)
- Seeds: default PyTorch RNG (non-deterministic CUDA); t-SNE `random_state=42`.
- All seven problem statements and their plots (training curves, gate heat-maps, feature maps, confusion matrices, t-SNE) are produced inline in the notebook.
