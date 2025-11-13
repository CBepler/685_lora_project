# Literature Review: Efficient Fine-Tuning Methods for Large Language Models

**Course**: ECE 685D - Fall 2025
**Project**: Exploring Sparsity in LLMs via LoRA
**Date**: November 2025

---

## 1. LoRA: Low-Rank Adaptation of Large Language Models

**Citation**: Hu, Edward J., et al. "LoRA: Low-rank adaptation of large language models." *ICLR* 1, no. 2 (2022): 3.

### Technical Innovation

LoRA introduces parameter-efficient fine-tuning by freezing pre-trained model weights and injecting trainable low-rank decomposition matrices into each Transformer layer. Instead of updating all parameters during fine-tuning, LoRA adds pairs of smaller matrices whose product approximates necessary parameter changes. This exploits the observation that task adaptation may exist in a lower-dimensional subspace than the full parameter space.

### Key Technical Details

The method decomposes weight updates into low-rank matrices, constraining the intrinsic dimensionality of adaptation based on findings of rank-deficiency in language model adaptation. During inference, the low-rank matrices can be merged with the frozen weights, introducing **no additional latency** compared to the original model—a significant advantage over adapter-based approaches.

### Performance Achievements

- **Parameter Efficiency**: For GPT-3 175B, LoRA reduces trainable parameters by **10,000×** and GPU memory by **3×**
- **Model Coverage**: Validated on RoBERTa, DeBERTa, GPT-2, GPT-3, and GLUE benchmarks
- **Quality**: Achieves performance parity with full fine-tuning while maintaining higher training throughput
- **Scalability**: Enables practical deployment of fine-tuned models at massive scale

### Significance

LoRA established the foundation for low-rank adaptation methods, demonstrating that substantial parameter reduction is achievable without sacrificing model quality. Its zero-latency inference characteristic makes it particularly attractive for production deployments.

---

## 2. SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity

**Citation**: Khaki, Samir, et al. "SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity." *arXiv preprint arXiv:2506.16500* (2025).

### Technical Innovation

SparseLoRA addresses a critical limitation of existing parameter-efficient methods: while techniques like LoRA and QLoRA reduce trainable parameters and memory, **they do not decrease computational cost** and may actually slow down training. SparseLoRA introduces computational acceleration through selective weight utilization with contextual awareness.

### Sparsity Mechanism

The method employs a **lightweight, training-free SVD sparsity estimator** that dynamically selects sparse subsets of weights for loss and gradient computation. By using Singular Value Decomposition to identify the most meaningful weight contributions, the method avoids learning sparsity patterns during training—eliminating associated overhead.

### Contextual Adaptation

SparseLoRA's sparsity selection adapts across three critical dimensions:
1. **Layers**: Different network depths require different sparsity patterns
2. **Tokens**: Input context influences which weights are activated
3. **Training Steps**: Sparsity evolves throughout the training process

This multi-dimensional contextual awareness enables more intelligent weight selection than static sparsity patterns.

### Performance Improvements

- **Computational Reduction**: Up to **2.2× reduction** in computational cost
- **Measured Speedup**: Up to **1.6× faster** training while maintaining accuracy
- **Task Coverage**: Validated on reasoning, code generation, and instruction-following benchmarks

### Significance

SparseLoRA demonstrates that sparsity can accelerate training without accuracy loss, addressing the computational bottleneck that standard LoRA leaves unresolved. The training-free nature of the sparsity estimator makes it practically deployable without additional overhead.

---

## 3. QLoRA: Efficient Finetuning of Quantized LLMs

**Citation**: Dettmers, Tim, et al. "QLoRA: Efficient Finetuning of Quantized LLMs." *arXiv preprint arXiv:2305.14314* (2023).

### Technical Innovation

QLoRA combines 4-bit quantization with LoRA adapters, enabling fine-tuning of massive models on consumer hardware. The method backpropagates gradients through a **frozen, 4-bit quantized pretrained model** into Low-Rank Adapters, keeping base weights compressed while training only small adapter modules.

### NormalFloat (NF4) Data Type

QLoRA introduces a novel 4-bit quantization format specifically optimized for neural network weights. NF4 is **information-theoretically optimal for normally distributed weights**, providing superior numerical representation compared to generic 4-bit formats. This theoretical foundation ensures minimal information loss despite extreme compression.

### Double Quantization

To achieve further memory savings, QLoRA implements **double quantization**—quantizing the quantization constants themselves. This nested approach compresses not only weights but also the metadata needed for storage and retrieval, creating cumulative memory reductions.

### Paged Optimizers

The system employs **paged optimizers** that use CPU memory as overflow capacity when GPU memory reaches limits. This prevents out-of-memory errors during training of very large models, enabling memory management that would otherwise be impossible.

### Performance Achievements

- **Memory Efficiency**: Fine-tune **65B parameter models on a single 48GB GPU**
- **Accuracy Preservation**: **99.3%** of ChatGPT performance with the Guanaco model family
- **Training Time**: Achieves competitive performance in just **24 hours** on single GPU
- **Precision Parity**: Maintains **full 16-bit fine-tuning performance** despite 4-bit quantization
- **Scale**: Over 1,000 models fine-tuned across LLaMA, T5, and multiple instruction-following datasets

### Significance

QLoRA democratizes large model fine-tuning by making it accessible on consumer hardware. The combination of theoretical optimality (NF4), aggressive compression (double quantization), and memory management (paged optimizers) creates a practical system for resource-constrained scenarios.

---

## 4. HiRA: Parameter-Efficient Hadamard High-Rank Adaptation

**Citation**: Huang, Qiushi, et al. "HiRA: Parameter-efficient hadamard high-rank adaptation for large language models." *ICLR* (2025).

### Technical Innovation

HiRA challenges LoRA's fundamental assumption that low-rank updates suffice for effective adaptation. While LoRA constrains updates to low-rank matrices for parameter efficiency, HiRA maintains **high-rank update parameters** through Hadamard (element-wise) multiplication, enhancing model expressiveness without proportional parameter growth.

### Hadamard Product Approach

Instead of constraining weight modifications through low-rank decomposition, HiRA uses **element-wise Hadamard multiplication** to achieve high-rank updates efficiently. This mathematical technique avoids the expressiveness limitations inherent in strict low-rank constraints while controlling total parameter overhead.

### Why High-Rank Matters

Low-rank adaptation, while parameter-efficient, may restrict the model's ability to capture complex task-specific transformations. The rank constraint limits the space of possible weight modifications. **High-rank updates permit richer modifications** to weight matrices, enabling superior performance on diverse tasks where required adaptations exceed what low-rank matrices can express.

### Performance Validation

- **Comparative Performance**: HiRA **outperforms LoRA and its variants** across multiple tasks
- **Task Coverage**: Validated on commonsense reasoning, dialogue generation, and mathematical reasoning
- **Ablation Studies**: Extensive experiments validate the effectiveness of the high-rank approach
- **Inference Efficiency**: HiRA **merges seamlessly with pre-trained weights**, introducing **no extra inference overhead**

### Architectural Balance

HiRA achieves parameter efficiency through architectural choices involving Hadamard multiplication rather than through rank restrictions alone. This design balances model capacity with computational resource constraints—critical for large language models.

### Significance

HiRA demonstrates that the low-rank assumption in LoRA may be unnecessarily restrictive. By using Hadamard products to maintain high-rank capacity, it shows that greater expressiveness is achievable within practical parameter budgets.

---

## 5. DistilBERT: A Distilled Version of BERT

**Citation**: Sanh, Victor, et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." *arXiv preprint arXiv:1910.01108* (2019).

### Technical Innovation

DistilBERT applies knowledge distillation during the pre-training phase rather than post-hoc task-specific distillation. The method employs a **triple loss combining language modeling, distillation, and cosine-distance losses** to transfer knowledge from BERT while maintaining linguistic understanding.

### Distillation Process

The methodology leverages inductive biases learned by BERT during pre-training. Three complementary loss components work synergistically:
1. **Language Modeling Loss**: Maintains task understanding
2. **Distillation Loss**: Transfers knowledge from teacher model
3. **Cosine-Distance Loss**: Preserves representation alignment

This multi-objective optimization during pre-training enables the smaller model to approximate BERT's behavior effectively.

### Compression Achievements

- **Size Reduction**: **40% smaller** than BERT-base
- **Speed Improvement**: **60% faster** than BERT-base
- **Performance Retention**: Retains **97% of language understanding capabilities**
- **Fine-tuning Versatility**: Effective across diverse downstream tasks without substantial degradation

### Fine-Tuning Suitability

DistilBERT proves highly suitable for fine-tuning experiments. Its compressed architecture reduces computational requirements while maintaining strong performance across a wide range of tasks. The substantial speed improvement (60% faster) makes it ideal for rapid experimentation and iteration during method development.

### Significance

DistilBERT demonstrates that model compression through distillation can achieve dramatic efficiency gains with minimal performance loss. Its balance of size, speed, and capability makes it an ideal base model for parameter-efficient fine-tuning research.

---

## Synthesis: Complementary Approaches to Efficiency

These five papers represent complementary strategies for efficient LLM adaptation:

1. **LoRA** establishes the foundation: low-rank adaptation with zero inference overhead
2. **SparseLoRA** adds computational efficiency through contextual sparsity
3. **QLoRA** enables memory-constrained fine-tuning through quantization
4. **HiRA** challenges rank limitations through Hadamard multiplication
5. **DistilBERT** provides an efficient base model for experimentation

### Key Trade-offs

| Method | Primary Benefit | Key Trade-off | Best Use Case |
|--------|----------------|---------------|---------------|
| LoRA | Parameter efficiency | May limit expressiveness | Standard fine-tuning with GPU constraints |
| SparseLoRA | Computational speedup | Requires sparsity estimation | Training acceleration with accuracy preservation |
| QLoRA | Memory efficiency | Quantization overhead | Large models on consumer hardware |
| HiRA | Model expressiveness | Slightly higher parameters than LoRA | Complex tasks requiring rich adaptations |
| DistilBERT | Base model efficiency | Smaller capacity than full BERT | Rapid prototyping and experimentation |

### Research Questions for This Project

1. How do sparsity constraints (SparseLoRA) affect accuracy compared to baseline LoRA?
2. What memory savings does QLoRA achieve, and at what accuracy cost?
3. Does HiRA's high-rank approach justify its additional parameters for our tasks?
4. How do these methods compare on sentiment classification vs. text generation?
5. Can we combine approaches (e.g., QLoRA + sparsity) for cumulative benefits?

---

## Conclusion

The literature reveals a rich landscape of parameter-efficient fine-tuning techniques, each addressing different efficiency dimensions: parameters (LoRA), computation (SparseLoRA), memory (QLoRA), and expressiveness (HiRA). Combined with an efficient base model (DistilBERT), this project will systematically compare these approaches to understand their trade-offs and identify optimal strategies for different resource constraints and task requirements.
