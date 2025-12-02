---
base_model: distilbert-base-uncased
library_name: peft
tags:
- hira
- lora
- sentiment-analysis
- imdb
datasets:
- imdb
metrics:
- accuracy
- f1
---

# HiRA - distilbert-base-uncased fine-tuned on IMDB

## Model Description

HiRA - High-Rank Adaptation using Hadamard multiplication for increased expressiveness

This model has been fine-tuned on the **IMDB** dataset for **Sentiment Analysis** using HiRA.

- **Base Model:** distilbert-base-uncased
- **Task:** Sentiment Analysis
- **Dataset:** IMDB
- **Method:** HiRA
- **Fine-tuning Technique:** Parameter-Efficient Fine-Tuning (PEFT)

## Training Configuration

### LoRA/Adapter Configuration
- **Rank (r):** 8
- **Alpha:** 16
- **Target Modules:** `q_lin`, `v_lin`
- **Dropout:** 0.1

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
- **Total Parameters:** 68,136,964
- **Trainable Parameters:** 1,181,954 (1.73% of total)
- **Model Size:** N/A MB

### Classification Results
- **Test Accuracy:** 0.9293
- **Test F1 Score:** 0.9293

## Usage

### Loading the Model

```python
from transformers import AutoModelForSequenceClassification
from peft import PeftModel

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Load adapter
model = PeftModel.from_pretrained(base_model, "path/to/adapter")
```

### Inference Example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

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

- This model is specifically fine-tuned for sentiment analysis on IMDB
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

**HiRA:**
```bibtex
@inproceedings{huang2025hira,
  title={HiRA: Parameter-efficient Hadamard High-rank Adaptation for Large Language Models},
  author={Huang, Qiushi and Ko, Tom and Zhuang, Zhan and Tang, Lilian and Zhang, Yu},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

## Framework Versions
- **Transformers:** 4.x
- **PEFT:** 0.18.0
- **PyTorch:** 2.x
- **Python:** 3.9+

## License

This adapter follows the license of the base model (distilbert-base-uncased).

## Model Card Authors

Generated as part of ECE 685D course project (Fall 2025) on efficient fine-tuning methods for large language models.

---

*This model card was generated on 2025-12-02*
