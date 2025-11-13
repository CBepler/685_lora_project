# Exploring Sparsity in LLMs via LoRA

**ECE 685D - Fall 2025**
**Project**: Efficient Fine-Tuning for Large Language Models

## Overview

This project explores four state-of-the-art parameter-efficient fine-tuning (PEFT) methods for large language models, comparing their trade-offs between accuracy, efficiency, and computational cost:

1. **LoRA** - Low-Rank Adaptation (Baseline)
2. **Sparse LoRA** - LoRA with sparsity constraints
3. **QLoRA** - Quantized LoRA (4-bit NF4 quantization)
4. **HiRA** - High-Rank Adaptation

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets
python scripts/prepare_datasets.py
```

### Training

```bash
# Train baseline LoRA on SST-2
python scripts/training/train_lora.py --dataset sst2 --config configs/config.yaml

# Train on IMDB
python scripts/training/train_lora.py --dataset imdb --config configs/config.yaml

# Train on WikiText-2
python scripts/training/train_lora.py --dataset wikitext2 --config configs/config.yaml
```

## Project Structure

```
├── configs/              # Configuration files
├── data/                 # Datasets (raw and processed)
├── docs/                 # Documentation and reports
│   ├── literature_review.md
│   └── PROJECT_STATUS.md
├── papers/               # Research paper references
├── results/              # Experiment results
├── scripts/              # Training and evaluation scripts
└── src/
    ├── models/           # Model implementations (LoRA, Sparse LoRA, QLoRA, HiRA)
    └── utils/            # Utility functions and metrics
```

## Datasets

- **SST-2**: Stanford Sentiment Treebank (sentiment classification)
- **IMDB**: Movie review sentiment analysis
- **WikiText-2**: Text generation

## Methods Implemented

### 1. LoRA (Baseline)
- Rank: 8, Alpha: 16
- Trainable parameters: ~739K (1.09% of DistilBERT)
- Zero inference latency overhead

### 2. Sparse LoRA
- Sparsity: 50% (configurable)
- Methods: Magnitude pruning, Top-k, L1 regularization
- Gradual or one-shot pruning schedules

### 3. QLoRA
- 4-bit NF4 quantization
- Double quantization enabled
- Expected ~75% memory reduction

### 4. HiRA
- High-rank adaptation (r=32 vs standard r=8)
- Trainable parameters: ~2.9M (4.3% of DistilBERT)
- Greater model expressiveness

## Documentation

- **Literature Review**: `docs/literature_review.md` - Comprehensive analysis of all 5 papers
- **Project Status**: `docs/PROJECT_STATUS.md` - Detailed progress tracking
- **Project Specification**: `CLAUDE.md` - Complete project requirements

## Current Status

✅ **Phase 1: Literature Review & Setup** - Complete
✅ **Phase 2: Implementation** - Complete (All 4 methods)
🔄 **Phase 3: Training & Evaluation** - In Progress
⏳ **Phase 4: Analysis & Reporting** - Pending

See `docs/PROJECT_STATUS.md` for detailed progress information.

## Key Features

- ✨ Clean, modular implementations of all 4 PEFT methods
- 📊 Comprehensive metrics tracking (accuracy, speed, memory, parameters)
- 🔧 Centralized configuration management
- 📈 Automatic experiment logging and checkpointing
- 🎯 Support for both classification and generation tasks

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT 0.4+
- bitsandbytes 0.40+
- See `requirements.txt` for complete list

## Citation

If you use this code, please cite the original papers:

```bibtex
@article{hu2021lora,
  title={LoRA: Low-rank adaptation of large language models},
  author={Hu, Edward J and others},
  journal={ICLR},
  year={2022}
}

@article{khaki2025sparselora,
  title={SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity},
  author={Khaki, Samir and others},
  journal={arXiv preprint arXiv:2506.16500},
  year={2025}
}

@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and others},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}

@inproceedings{huang2025hira,
  title={HiRA: Parameter-efficient hadamard high-rank adaptation for large language models},
  author={Huang, Qiushi and others},
  booktitle={ICLR},
  year={2025}
}
```

## License

This project is for educational purposes as part of ECE 685D coursework.

## Contact

For questions or issues, please refer to the course materials or contact the TAs (Haoming, Zihao).