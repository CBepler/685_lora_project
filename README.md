# 685_lora_project

## Overview of Efficient Fine-Tuning Techniques

This project explores four state-of-the-art methods for efficient fine-tuning of Large Language Models (LLMs).

---

## 1. LoRA (Low-Rank Adaptation)

### Summary
LoRA reduces the computational cost of fine-tuning by decomposing weight updates into low-rank matrices. Instead of updating the full weight matrix W (N×M), LoRA keeps W frozen and trains two smaller matrices A (N×r) and B (r×M), where r << min(N,M). The adapted output is: h = Wx + BAx.

### Key Benefits
- Drastically reduces trainable parameters (from millions to thousands)
- No additional inference latency (matrices can be merged)
- Memory efficient during training

### Dataflow Diagram
```
Input (x)
    |
    ├─────────────────────┐
    |                     |
    v                     v
[Frozen W]            [Trainable B]
  (N×M)                  (r×M)
    |                     ^
    |                     |
    |                [Trainable A]
    |                   (N×r)
    |                     ^
    |                     |
    |─────────────────────┘
    |          (input x also flows here)
    v
  Sum: Wx + BAx
    |
    v
  Output

Legend:
  N = input dimension
  M = output dimension
  r = rank (r << min(N,M), typically 4-64)
```

---

## 2. Sparse LoRA

### Summary
Sparse LoRA extends standard LoRA by introducing sparsity constraints on the low-rank matrices A and B. This is achieved through L1 regularization or magnitude-based pruning, forcing many weights to zero. Sparse LoRA further reduces computational cost and memory footprint while maintaining competitive accuracy.

### Key Benefits
- Fewer non-zero parameters than standard LoRA
- Faster inference through sparse matrix operations
- Potential for better generalization through implicit regularization
- Can accelerate both training and inference

### Dataflow Diagram
```
Input (x)
    |
    ├─────────────────────┐
    |                     |
    v                     v
[Frozen W]         [Sparse Trainable B]
  (N×M)               (r×M, ~30-70% zeros)
    |                     ^
    |                     |
    |              [Sparse Trainable A]
    |                (N×r, ~30-70% zeros)
    |                     ^
    |                     |
    |─────────────────────┘
    |
    v
  Sum: Wx + BAx (with sparse BA)
    |
    v
  Output

Sparsity Techniques:
  - L1 Regularization: L = L_task + λ||A||₁ + λ||B||₁
  - Magnitude Pruning: Set smallest |weights| to 0
  - Structured Sparsity: Prune entire rows/columns
```

---

## 3. HiRA (High-Rank Adaptation)

### Summary
HiRA uses high-rank matrices with Hadamard (element-wise) multiplication instead of matrix multiplication. By decomposing updates as W' = W ⊙ (AB^T) where A (N×r) and B (M×r) with r potentially larger than LoRA's rank, HiRA achieves more expressive adaptations while maintaining efficiency through the Hadamard product.

### Key Benefits
- Higher expressiveness than low-rank LoRA
- Captures more complex adaptations
- Element-wise operations are computationally efficient
- Better performance on complex downstream tasks

### Dataflow Diagram
```
Input (x)
    |
    v
[Frozen W] ⊙ [Hadamard Mask]
  (N×M)         (N×M)
    |              ^
    |              |
    |         [Trainable A] × [Trainable B]^T
    |            (N×r)          (M×r)
    |              |
    |         (Outer Product)
    |              |
    v              v
  W' = W ⊙ (AB^T)
    |
    v
  W'x
    |
    v
  Output

Key Difference:
  - LoRA: Wx + BAx (additive, low-rank)
  - HiRA: (W ⊙ AB^T)x (multiplicative, high-rank)
  - ⊙ denotes element-wise (Hadamard) multiplication
  - r can be larger than LoRA's rank while remaining efficient
```

---

## 4. QLoRA (Quantized LoRA)

### Summary
QLoRA combines 4-bit quantization of the base model with LoRA adapters to dramatically reduce memory requirements. The pretrained weights are quantized to 4-bit NormalFloat (NF4) format, while LoRA adapters remain in full precision. This enables fine-tuning of very large models (e.g., 65B parameters) on consumer GPUs.

### Key Benefits
- 4x memory reduction for base model (4-bit vs 16-bit)
- Enables fine-tuning larger models on limited hardware
- Minimal accuracy loss compared to full-precision LoRA
- Paged optimizers handle memory spikes

### Dataflow Diagram
```
Input (x)
    |
    ├─────────────────────────┐
    |                         |
    v                         v
[Quantized W]           [FP16 Trainable B]
(N×M, 4-bit NF4)           (r×M, 16-bit)
    |                         ^
    |                         |
[Dequantize]            [FP16 Trainable A]
    |                      (N×r, 16-bit)
    v                         ^
[FP16 W']                     |
    |                         |
    |─────────────────────────┘
    v
Sum: W'x + BAx (computed in FP16)
    |
    v
  Output

Memory Optimization Techniques:
  1. 4-bit NormalFloat (NF4): Information-theoretically optimal
  2. Double Quantization: Quantize the quantization constants
  3. Paged Optimizers: Use unified memory for optimizer states

Memory Comparison (for 65B model):
  - Full Fine-tuning: ~780 GB
  - Standard LoRA (16-bit): ~320 GB
  - QLoRA (4-bit): ~48 GB
```

---

## Comparison Summary

| Method | Parameters | Memory | Speed | Accuracy | Best Use Case |
|--------|-----------|---------|-------|----------|---------------|
| **LoRA** | Low (r×(N+M)) | Medium | Fast | High | General-purpose efficient fine-tuning |
| **Sparse LoRA** | Very Low | Low | Fastest | Medium-High | Resource-constrained deployment |
| **HiRA** | Medium (r×(N+M)) | Medium | Medium | Highest | Complex tasks requiring expressiveness |
| **QLoRA** | Low (r×(N+M)) | Very Low | Medium | High | Large models on limited hardware |

---

## Project Structure

This project implements and compares all four methods on:
- **Datasets**: SST-2, IMDB, WikiText-2
- **Base Model**: DistilBERT
- **Metrics**: Accuracy, inference speed, memory usage, training time
