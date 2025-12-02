---
base_model: bert-base-uncased
library_name: peft
tags:
- sparse_lora
- lora
- language-modeling
- wikitext2
datasets:
- wikitext2
metrics:
- accuracy
- f1
---

# Sparse LoRA - bert-base-uncased fine-tuned on WikiText-2

## Model Description

Sparse LoRA - LoRA with sparsity constraints for improved efficiency

This model has been fine-tuned on the **WikiText-2** dataset for **Language Modeling** using Sparse LoRA.

- **Base Model:** bert-base-uncased
- **Task:** Language Modeling
- **Dataset:** WikiText-2
- **Method:** Sparse LoRA
- **Fine-tuning Technique:** Parameter-Efficient Fine-Tuning (PEFT)

## Training Configuration

### LoRA/Adapter Configuration
- **Rank (r):** 8
- **Alpha:** 16
- **Target Modules:** `q_lin`, `v_lin`
- **Dropout:** 0.1
- **Sparsity:** 50.0% (magnitude-based pruning)

### Training Hyperparameters
- **Number of Epochs:** 3
- **Batch Size:** 16
- **Learning Rate:** 3e-4
- **Weight Decay:** 0.01
- **Warmup Steps:** 100
- **Optimizer:** AdamW
- **Scheduler:** Linear with warmup

## Performance Metrics

### Model Size and Efficiency
- **Total Parameters:** 67,132,986
- **Trainable Parameters:** 147,456 (0.22% of total)
- **Model Size:** N/A MB

### Language Modeling Results
- **Test Perplexity:** N/A

## Usage

### Loading the Model

```python
from transformers import AutoModelForSequenceClassification
from peft import PeftModel

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Load adapter
model = PeftModel.from_pretrained(base_model, "path/to/adapter")
```

### Inference Example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Prepare input
text = "This is a great movie!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Get predictions
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)
```

## Comparison with Other Methods

This model is part of a comprehensive comparison of efficient fine-tuning methods:
- **LoRA**: Standard low-rank adaptation
- **Sparse LoRA**: LoRA with sparsity constraints
- **QLoRA**: Quantized LoRA (4-bit)
- **HiRA**: High-rank adaptation with Hadamard multiplication

See the [project repository](.) for detailed comparisons across all methods.

## Limitations

- This model is specifically fine-tuned for language modeling on WikiText-2
- Performance may vary on out-of-distribution data
- The model inherits biases present in the base model and training data

## Citation

If you use this model, please cite the original papers:

**LoRA:**
```bibtex
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

**Sparse LoRA:**
```bibtex
@article{khaki2025sparselora,
  title={SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity},
  author={Khaki, Samir and Li, Xiuyu and Guo, Junxian and others},
  journal={arXiv preprint arXiv:2506.16500},
  year={2025}
}
```

## Framework Versions
- **Transformers:** 4.x
- **PEFT:** 0.18.0
- **PyTorch:** 2.x
- **Python:** 3.9+

## License

This adapter follows the license of the base model (bert-base-uncased).

## Model Card Authors

Generated as part of ECE 685D course project (Fall 2025) on efficient fine-tuning methods for large language models.

---

*This model card was generated on 2025-12-02*
