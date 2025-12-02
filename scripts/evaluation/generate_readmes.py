#!/usr/bin/env python3
"""
Generate comprehensive README files for all model checkpoints.
"""

import json
from pathlib import Path
from datetime import datetime

# Model/dataset combinations
COMBINATIONS = [
    ("lora", "imdb", "IMDB", "Sentiment Analysis"),
    ("lora", "sst2", "SST-2", "Sentiment Analysis"),
    ("lora", "wikitext2", "WikiText-2", "Language Modeling"),
    ("sparse_lora", "imdb", "IMDB", "Sentiment Analysis"),
    ("sparse_lora", "sst2", "SST-2", "Sentiment Analysis"),
    ("sparse_lora", "wikitext2", "WikiText-2", "Language Modeling"),
    ("qlora", "imdb", "IMDB", "Sentiment Analysis"),
    ("qlora", "sst2", "SST-2", "Sentiment Analysis"),
    ("qlora", "wikitext2", "WikiText-2", "Language Modeling"),
    ("hira", "imdb", "IMDB", "Sentiment Analysis"),
    ("hira", "sst2", "SST-2", "Sentiment Analysis"),
    ("hira", "wikitext2", "WikiText-2", "Language Modeling"),
]

METHOD_NAMES = {
    "lora": "LoRA",
    "sparse_lora": "Sparse LoRA",
    "qlora": "QLoRA",
    "hira": "HiRA",
}

METHOD_DESCRIPTIONS = {
    "lora": "Low-Rank Adaptation (LoRA) - Parameter-efficient fine-tuning using low-rank matrix decomposition",
    "sparse_lora": "Sparse LoRA - LoRA with sparsity constraints for improved efficiency",
    "qlora": "QLoRA - Quantized LoRA using 4-bit precision for reduced memory footprint",
    "hira": "HiRA - High-Rank Adaptation using Hadamard multiplication for increased expressiveness",
}

BASE_MODELS = {
    "imdb": "distilbert-base-uncased",
    "sst2": "distilbert-base-uncased",
    "wikitext2": "bert-base-uncased",
}


def load_metrics(model_dir, dataset, model_type):
    """Load metrics from JSON file."""
    metrics_file = model_dir / f"{dataset}_{model_type}_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {}


def format_number(value, decimals=4):
    """Format number with appropriate precision."""
    if isinstance(value, (int, float)):
        if value > 1000:
            return f"{value:,.0f}"
        else:
            return f"{value:.{decimals}f}"
    return str(value)


def generate_readme(model_type, dataset, dataset_display, task_type, metrics):
    """Generate README content for a model."""
    method_name = METHOD_NAMES[model_type]
    method_desc = METHOD_DESCRIPTIONS[model_type]
    base_model = BASE_MODELS.get(dataset, "distilbert-base-uncased")
    task_type_lower = task_type.lower()

    # Get key metrics with fallbacks
    test_acc = metrics.get("test_accuracy", metrics.get("test/accuracy", "N/A"))
    test_f1 = metrics.get("test_f1", metrics.get("test/f1", "N/A"))
    test_perplexity = metrics.get("test_perplexity", metrics.get("test/perplexity", "N/A"))
    trainable_params = metrics.get("trainable_parameters", metrics.get("model/trainable_parameters", "N/A"))
    total_params = metrics.get("total_parameters", metrics.get("model/total_parameters", "N/A"))
    trainable_pct = metrics.get("trainable_percentage", metrics.get("model/trainable_percentage", "N/A"))
    training_time = metrics.get("training_time_seconds", "N/A")
    throughput = metrics.get("inference_throughput_samples_per_sec",
                            metrics.get("inference/throughput_samples_per_second", "N/A"))
    model_size_mb = metrics.get("model_size_mb", metrics.get("final_model/model_size_mb", "N/A"))
    gpu_memory = metrics.get("peak_gpu_memory_gb",
                            metrics.get("memory/gpu_memory_max_allocated_gb", "N/A"))

    # Model-specific parameters
    lora_rank = metrics.get("model/lora_rank", 8)
    lora_alpha = metrics.get("model/lora_alpha", 16)
    sparsity = metrics.get("sparsity_percentage", "N/A")

    # Format metrics
    if test_acc != "N/A":
        test_acc = format_number(test_acc, 4)
    if test_f1 != "N/A":
        test_f1 = format_number(test_f1, 4)
    if test_perplexity != "N/A":
        test_perplexity = format_number(test_perplexity, 2)
    if trainable_params != "N/A":
        trainable_params = format_number(trainable_params, 0)
    if total_params != "N/A":
        total_params = format_number(total_params, 0)
    if trainable_pct != "N/A":
        trainable_pct = format_number(trainable_pct, 2)
    if training_time != "N/A":
        training_time_min = float(training_time) / 60
        training_time = f"{format_number(training_time, 1)}s (~{training_time_min:.1f} min)"
    if throughput != "N/A":
        throughput = format_number(throughput, 2)
    if model_size_mb != "N/A":
        model_size_mb = format_number(model_size_mb, 2)
    if gpu_memory != "N/A":
        gpu_memory = format_number(gpu_memory, 2)

    readme = f"""---
base_model: {base_model}
library_name: peft
tags:
- {model_type}
- lora
- {task_type.lower().replace(' ', '-')}
- {dataset}
datasets:
- {dataset}
metrics:
- accuracy
- f1
---

# {method_name} - {base_model} fine-tuned on {dataset_display}

## Model Description

{method_desc}

This model has been fine-tuned on the **{dataset_display}** dataset for **{task_type}** using {method_name}.

- **Base Model:** {base_model}
- **Task:** {task_type}
- **Dataset:** {dataset_display}
- **Method:** {method_name}
- **Fine-tuning Technique:** Parameter-Efficient Fine-Tuning (PEFT)

## Training Configuration

### LoRA/Adapter Configuration"""

    if model_type == "qlora":
        readme += """
- **Quantization:** 4-bit NormalFloat (NF4)
- **Double Quantization:** Enabled"""

    readme += f"""
- **Rank (r):** {lora_rank}
- **Alpha:** {lora_alpha}
- **Target Modules:** `q_lin`, `v_lin`
- **Dropout:** 0.1"""

    if model_type == "sparse_lora" and sparsity != "N/A":
        readme += f"""
- **Sparsity:** {sparsity}% (magnitude-based pruning)"""

    readme += """

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
- **Total Parameters:** {total_params}
- **Trainable Parameters:** {trainable_params} ({trainable_pct}% of total)
- **Model Size:** {model_size_mb} MB""".format(
        total_params=total_params,
        trainable_params=trainable_params,
        trainable_pct=trainable_pct,
        model_size_mb=model_size_mb
    )

    if task_type == "Sentiment Analysis":
        readme += f"""

### Classification Results
- **Test Accuracy:** {test_acc}
- **Test F1 Score:** {test_f1}"""
    else:
        readme += f"""

### Language Modeling Results
- **Test Perplexity:** {test_perplexity}"""

    if training_time != "N/A" or throughput != "N/A" or gpu_memory != "N/A":
        readme += f"""

### Training and Inference Performance"""
        if training_time != "N/A":
            readme += f"""
- **Training Time:** {training_time}"""
        if throughput != "N/A":
            readme += f"""
- **Inference Throughput:** {throughput} samples/second"""
        if gpu_memory != "N/A":
            readme += f"""
- **Peak GPU Memory:** {gpu_memory} GB"""

    readme += """

## Usage

### Loading the Model

```python
from transformers import AutoModelForSequenceClassification
from peft import PeftModel

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained("{base_model}")

# Load adapter
model = PeftModel.from_pretrained(base_model, "path/to/adapter")
```

### Inference Example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{base_model}")

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

- This model is specifically fine-tuned for {task_type_lower} on {dataset_display}
- Performance may vary on out-of-distribution data
- The model inherits biases present in the base model and training data

## Citation

If you use this model, please cite the original papers:

**LoRA:**
```bibtex
@inproceedings{{hu2022lora,
  title={{LoRA: Low-Rank Adaptation of Large Language Models}},
  author={{Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu}},
  booktitle={{International Conference on Learning Representations}},
  year={{2022}}
}}
```""".format(base_model=base_model, task_type=task_type, task_type_lower=task_type_lower, dataset_display=dataset_display)

    if model_type == "sparse_lora":
        readme += """

**Sparse LoRA:**
```bibtex
@article{khaki2025sparselora,
  title={SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity},
  author={Khaki, Samir and Li, Xiuyu and Guo, Junxian and others},
  journal={arXiv preprint arXiv:2506.16500},
  year={2025}
}
```"""

    if model_type == "qlora":
        readme += """

**QLoRA:**
```bibtex
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```"""

    if model_type == "hira":
        readme += """

**HiRA:**
```bibtex
@inproceedings{huang2025hira,
  title={HiRA: Parameter-efficient Hadamard High-rank Adaptation for Large Language Models},
  author={Huang, Qiushi and Ko, Tom and Zhuang, Zhan and Tang, Lilian and Zhang, Yu},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```"""

    readme += """

## Framework Versions
- **Transformers:** 4.x
- **PEFT:** 0.18.0
- **PyTorch:** 2.x
- **Python:** 3.9+

## License

This adapter follows the license of the base model ({base_model}).

## Model Card Authors

Generated as part of ECE 685D course project (Fall 2025) on efficient fine-tuning methods for large language models.

---

*This model card was generated on {date}*
""".format(base_model=base_model, date=datetime.now().strftime("%Y-%m-%d"))

    return readme


def main():
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "results"

    print("Generating README files for all checkpoints...")
    print("=" * 60)

    for model_type, dataset, dataset_display, task_type in COMBINATIONS:
        print(f"\nGenerating README for: {model_type} on {dataset}")

        # Load metrics
        model_dir = results_dir / model_type / dataset
        metrics = load_metrics(model_dir, dataset, model_type)

        # Generate README
        readme_content = generate_readme(model_type, dataset, dataset_display, task_type, metrics)

        # Save README
        readme_path = model_dir / "best_model" / "README.md"
        readme_path.parent.mkdir(parents=True, exist_ok=True)

        with open(readme_path, 'w') as f:
            f.write(readme_content)

        print(f"  ✓ Saved README to: {readme_path}")

    print("\n" + "=" * 60)
    print("README generation complete!")


if __name__ == "__main__":
    main()
