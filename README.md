# NanoGPT: A Deep Dive into Generative Pre-training

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)

## Executive Summary
**NanoGPT** is a high-performance, minimalist implementation of a Decoder-only Transformer, modeled after the GPT (Generative Pre-trained Transformer) architecture. This project serves as an architectural exploration into the mechanics of causal self-attention, layer normalization, and the optimization bottlenecks inherent in training large-scale autoregressive models.

By prioritizing code transparency and mathematical alignment with the original GPT-2/3 specifications, this repository demonstrates the bridge between the seminal "Attention is All You Need" (Vaswani et al., 2017) paper and modern production-grade LLMs.

---

## Technical Architecture

The core of NanoGPT is a character-level (or sub-word) language model that predicts the next token in a sequence by modeling the conditional probability distribution $P(x_t \mid x_{t-1}, \dots, x_1)$.

### Key Design Pillars:
* **Causal Self-Attention**: Implemented with a masked multi-head attention mechanism to ensure that the prediction for position $i$ can only depend on known outputs at positions $j < i$.
* **Residual Connections & LayerNorm**: Utilizes a **Pre-LN** (Pre-Layer Normalization) architecture to improve training stability and gradient flow in deeper networks.
* **Weight Tieing**: Optionally shares weights between the embedding and pre-softmax linear layers to reduce parameter count and regularize the model.
* **GeLU Activation**: Employs the Gaussian Error Linear Unit (GeLU) in the feed-forward blocks for smoother non-linearity compared to standard ReLU.

---

## Mathematical Foundation

The Scaled Dot-Product Attention at the heart of each head is computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Where:
* $Q, K, V$ are the Query, Key, and Value matrices.
* $d_k$ is the dimensionality of the keys (used as a scaling factor to prevent gradient vanishing in the softmax).
* $M$ is the **Causal Mask**, where $M_{ij} = -\infty$ for $i < j$, effectively zeroing out future context.

---

## Features & Engineering Rigor

* **Efficient Computation**: Optimized using PyTorch's vectorized operations and `torch.compile` to maximize GPU TFLOPS.
* **Hyperparameter Configurability**: Full control over `n_layer`, `n_head`, `n_embd`, and `dropout` via a clean configuration interface.
* **Optimization Strategy**: Implements the **AdamW** optimizer with weight decay and a **Cosine Learning Rate Decay** schedule with a linear warmup phase.
* **Scalability**: Scripts support training on small corpora (e.g., Shakespeare) for proof-of-concept, with the architectural flexibility to scale to larger datasets.

---

## Getting Started

### Prerequisites
* Python 3.8+
* PyTorch 2.0+ 
* NumPy

### Installation
git clone [https://github.com/AKDev32/NanoGPT.git](https://github.com/AKDev32/NanoGPT.git)
cd NanoGPT
pip install -r requirements.txt

---

## Educational Objectives & Research Value

This implementation was strategically developed to explore the following domains of neural architecture and optimization:

1. **The Impact of Sparsity**: Systematically observing how varying **Dropout** levels affect the model's loss curves, particularly when training on smaller datasets (e.g., *TinyShakespeare*), to mitigate over-fitting.
2. **Inference Dynamics**: Investigating the heuristic relationship between **Temperature** and **Top-K sampling**. This includes analyzing the trade-off between stochastic "creativity" and structural coherence in the resulting autoregressive generation.
3. **Efficiency Bottlenecks**: Quantifying the computational and memory overhead of the $O(N^2)$ attention mechanism during the forward pass, and assessing the limits of sequence length within standard GPU VRAM constraints.

---

## References

* **Vaswani, A., et al. (2017).** *Attention is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
* **Radford, A., et al. (2019).** *Language Models are Unsupervised Multitask Learners* (GPT-2). [OpenAI Blog](https://openai.com/research/better-language-models).
* **Karpathy, A. (2022).** *Let's build GPT: from scratch, in code, spelled out.* [GitHub/Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero).

---

**Author:** [Aman Kumar / AKDev32]  
**Academic Focus:** Natural Language Processing / Neural Architectures
