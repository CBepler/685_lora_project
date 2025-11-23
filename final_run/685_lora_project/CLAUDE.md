# Project 1: Exploring Sparsity in LLMs via LoRA

## Project Information

- **Course**: ECE 685D - Fall 2025
- **TAs in Charge**: Haoming, Zihao
- **Team Size**: Maximum 4 students per group
- **Project Type**: Efficient Fine-tuning for Large Language Models

## Objective

Large Language Models (LLMs) typically have billions of parameters, making (re)training for specific tasks computationally expensive. This project explores **LoRA (Low-Rank Adaptation)** methods for efficient fine-tuning of LLMs, comparing different adaptation techniques to understand trade-offs between accuracy, efficiency, and cost.

## Methods to Implement

You will implement and compare four state-of-the-art efficient tuning methods:

1. **LoRA** (Low Rank Adaptors)
   - Fine-tune with low-rank matrices without sparsity
   - Baseline method for comparison

2. **Sparse LoRA**
   - Low-rank adaptation with sparsity constraints
   - Apply L1 regularization or pruning to LoRA matrices

3. **QLoRA** (Quantized LoRA)
   - Combines 4-bit quantization with LoRA adapters
   - Dramatically reduces memory footprint
   - Enables fine-tuning of larger models on limited hardware

4. **HiRA** (High Rank Adaptors)
   - High-rank adaptation with Hadamard multiplication
   - More expressive than standard LoRA

## Datasets

Use the following benchmark datasets for evaluation:

1. **SST-2** (Stanford Sentiment Treebank)
   - Sentiment classification task
   - URL: https://huggingface.co/datasets/stanfordnlp/sst2

2. **IMDB Reviews**
   - Movie review sentiment analysis
   - URL: https://huggingface.co/datasets/stanfordnlp/imdb

3. **WikiText-2** (Small subset)
   - Text generation task
   - URL: https://huggingface.co/datasets/mindchain/wikitext2

## Recommended Model

- **DistilBERT** or **BERT-base**
  - Small-scale LLM suitable for this project
  - Computationally feasible for training experiments
  - Well-documented and widely supported

## Step-by-Step Instructions

### Step 1: Literature Review

Review and summarize the following papers:

- [ ] **Paper 1**: Hu, Edward J., et al. "LoRA: Low-rank adaptation of large language models." *ICLR* 1, no. 2 (2022): 3.
  - Summarize the LoRA technique
  - Understand low-rank decomposition approach
  - Note key innovations and limitations

- [ ] **Paper 2**: Khaki, Samir, et al. "SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity." *arXiv preprint arXiv:2506.16500* (2025).
  - Understand sparsity constraints in LoRA
  - Learn about acceleration techniques
  - Compare with standard LoRA

- [ ] **Paper 3**: Dettmers, Tim, et al. "QLoRA: Efficient Finetuning of Quantized LLMs." *arXiv preprint arXiv:2305.14314* (2023).
  - Understand 4-bit quantization with NormalFloat (NF4)
  - Learn about double quantization technique
  - Understand paged optimizers for memory management
  - Compare memory efficiency with standard LoRA

- [ ] **Paper 4**: Huang, Qiushi, et al. "HiRA: Parameter-efficient hadamard high-rank adaptation for large language models." *ICLR* (2025).
  - Understand high-rank adaptation
  - Learn about Hadamard multiplication approach
  - Analyze performance characteristics

- [ ] **Paper 5**: Sanh, Victor, et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." *arXiv preprint arXiv:1910.01108* (2019).
  - Understand the base model architecture
  - Note distillation techniques

**Deliverable**: Write a summary (1-2 pages) of technical innovations from each paper

### Step 2: Implementation

#### 2.1 Environment Setup

- [ ] Set up Python environment (recommended: Python 3.8+)
- [ ] Install required libraries:
  - PyTorch or TensorFlow
  - Transformers (HuggingFace)
  - Datasets library
  - PEFT (Parameter-Efficient Fine-Tuning) library
  - bitsandbytes (for QLoRA quantization)
  - accelerate (for distributed training)
- [ ] Download and prepare datasets
- [ ] Load pre-trained DistilBERT model

#### 2.2 Baseline LoRA Implementation

- [ ] Implement standard LoRA adapter layers
- [ ] Configure low-rank parameters (rank r)
- [ ] Fine-tune on downstream tasks (SST-2, IMDB, WikiText-2)
- [ ] Track training metrics:
  - Training time
  - Memory usage
  - Number of trainable parameters
  - Task accuracy

#### 2.3 Sparse LoRA Implementation

- [ ] Extend LoRA implementation with sparsity constraints
- [ ] Implement sparsity techniques:
  - L1 regularization on LoRA matrices
  - Magnitude-based pruning
  - Structured sparsity (optional)
- [ ] Fine-tune with same hyperparameters as baseline
- [ ] Measure sparsity levels (% of zero weights)
- [ ] Track same metrics as baseline

#### 2.4 QLoRA Implementation

- [ ] Implement 4-bit quantization for the base model
- [ ] Use NormalFloat (NF4) data type for weights
- [ ] Apply LoRA adapters on top of quantized model
- [ ] Implement double quantization (optional, for further memory savings)
- [ ] Configure paged optimizers to handle memory spikes
- [ ] Fine-tune on same tasks
- [ ] Track memory savings compared to standard LoRA:
  - Peak memory usage during training
  - Model size on disk
  - Memory usage during inference
- [ ] Compare accuracy with full-precision LoRA

#### 2.5 HiRA Implementation

- [ ] Implement high-rank adaptation with Hadamard multiplication
- [ ] Configure rank parameters
- [ ] Fine-tune on same tasks
- [ ] Track computational requirements
- [ ] Compare with LoRA and Sparse LoRA

### Step 3: Evaluation and Analysis

#### 3.1 Quantitative Evaluation

Measure and compare the following metrics across all four methods:

- [ ] **Task Accuracy**
  - Test set accuracy for SST-2
  - Test set accuracy for IMDB
  - Perplexity for WikiText-2 generation

- [ ] **Inference Speed**
  - Average inference time per sample
  - Throughput (samples/second)
  - Latency measurements

- [ ] **Efficiency Metrics**
  - Number of trainable parameters
  - Memory footprint during training (especially important for QLoRA)
  - Memory footprint during inference
  - Model size on disk (compare quantized vs. full precision)
  - Training time per epoch
  - Total training time

- [ ] **Cost Analysis**
  - Computational cost (FLOPs)
  - GPU hours required
  - Storage requirements for adapted weights

#### 3.2 Comparative Analysis

- [ ] Create comparison tables for all metrics
- [ ] Generate visualizations:
  - Accuracy vs. number of parameters
  - Inference speed vs. accuracy
  - Training time comparison
  - Memory usage comparison
- [ ] Analyze trade-offs between methods

#### 3.3 Discussion

- [ ] Discuss pros and cons of each method
- [ ] Identify scenarios where each method excels
- [ ] Analyze the impact of sparsity on performance (Sparse LoRA)
- [ ] Analyze the impact of quantization on accuracy and memory (QLoRA)
- [ ] Compare high-rank vs. low-rank approaches (HiRA vs. LoRA)
- [ ] Discuss trade-offs between memory efficiency and accuracy

#### 3.4 Innovation (Optional)

- [ ] Propose potential improvements to LoRA methods
- [ ] Suggest novel sparsity patterns or constraints
- [ ] Explore hybrid approaches (e.g., combining QLoRA with sparsity)
- [ ] Consider adaptive rank selection strategies
- [ ] Investigate combining quantization with high-rank adaptation

## Expected Deliverables

1. **Code Implementation**
   - Clean, well-documented code
   - Separate implementations for LoRA, Sparse LoRA, QLoRA, and HiRA
   - Training and evaluation scripts
   - Configuration files for reproducibility

2. **Experimental Results**
   - Comprehensive evaluation on all three datasets
   - Performance metrics for all methods
   - Visualization of results

3. **Written Report**
   - Literature review summary
   - Methodology description
   - Results and analysis
   - Discussion of trade-offs
   - Proposed innovations (optional)
   - Conclusions and future work

4. **Presentation** (if required)
   - Key findings
   - Method comparisons
   - Insights and recommendations

## Evaluation Criteria

Your project will be evaluated based on:

1. **Correctness of Implementation** (40%)
   - Proper implementation of all four methods
   - Correct use of datasets and evaluation metrics
   - Reproducible results

2. **Experimental Rigor** (30%)
   - Comprehensive evaluation
   - Fair comparison across methods
   - Statistical significance of results

3. **Analysis Quality** (20%)
   - Depth of understanding
   - Insightful discussion of trade-offs
   - Quality of proposed innovations

4. **Presentation** (10%)
   - Code quality and documentation
   - Report clarity and organization
   - Visualization quality

## Resources and References

### Key Papers

1. **LoRA**: Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. "Lora: Low-rank adaptation of large language models." *ICLR* 1, no. 2 (2022): 3.

2. **SparseLoRA**: Khaki, Samir, Xiuyu Li, Junxian Guo, Ligeng Zhu, Chenfeng Xu, Konstantinos N. Plataniotis, Amir Yazdanbakhsh, Kurt Keutzer, Song Han, and Zhijian Liu. "SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity." *arXiv preprint arXiv:2506.16500* (2025).

3. **QLoRA**: Dettmers, Tim, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. "QLoRA: Efficient Finetuning of Quantized LLMs." *arXiv preprint arXiv:2305.14314* (2023).

4. **HiRA**: Huang, Qiushi, Tom Ko, Zhan Zhuang, Lilian Tang, and Yu Zhang. "HiRA: Parameter-efficient hadamard high-rank adaptation for large language models." In *The Thirteenth International Conference on Learning Representations*. 2025.

5. **DistilBERT**: Sanh, Victor, Lysandre Debut, Julien Chaumond, and Thomas Wolf. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." *arXiv preprint arXiv:1910.01108* (2019).

### Useful Libraries

- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **PEFT Library**: https://huggingface.co/docs/peft
- **bitsandbytes**: https://github.com/TimDettmers/bitsandbytes (for QLoRA quantization)
- **PyTorch**: https://pytorch.org/docs/stable/index.html
- **Datasets**: https://huggingface.co/docs/datasets

### Tutorials and Guides

- LoRA Tutorial: https://huggingface.co/docs/peft/conceptual_guides/lora
- QLoRA Tutorial: https://huggingface.co/blog/4bit-transformers-bitsandbytes
- Fine-tuning with PEFT: https://huggingface.co/blog/peft
- DistilBERT Fine-tuning: https://huggingface.co/distilbert-base-uncased

## Project Timeline Suggestions

| Week | Tasks |
|------|-------|
| 1 | Literature review, environment setup |
| 2 | Implement baseline LoRA |
| 3 | Implement Sparse LoRA |
| 4 | Implement QLoRA |
| 5 | Implement HiRA |
| 6 | Run experiments and collect results |
| 7 | Analysis, report writing, and presentation preparation |

**Note**: With four methods to implement, consider starting early or dividing work among team members.

## Tips for Success

1. **Start Early**: Literature review and setup take time
2. **Use Existing Libraries**: Leverage HuggingFace PEFT library when possible
3. **Track Everything**: Log all experiments with clear parameters
4. **Version Control**: Use git to track code changes
5. **Small Scale First**: Test on small subsets before full training
6. **Document as You Go**: Don't leave documentation for the end
7. **Compare Fairly**: Use same hyperparameters and random seeds across methods
8. **Seek Help**: Contact TAs (Haoming, Zihao) when stuck

## Notes

- Focus on understanding the **trade-offs** between methods
- Sparsity can reduce computation but may impact accuracy
- Quantization (QLoRA) dramatically reduces memory but may introduce precision loss
- QLoRA enables fine-tuning larger models on consumer-grade GPUs
- Consider both training and inference efficiency
- Real-world applicability is important in your analysis

---

*Good luck with your project! This is an opportunity to gain hands-on experience with cutting-edge efficient fine-tuning techniques for LLMs.*
