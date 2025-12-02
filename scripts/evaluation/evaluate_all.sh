#!/bin/bash
# Evaluate all trained models

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "================================"
echo "Evaluating All LoRA Variants"
echo "================================"

# LoRA evaluations
echo -e "\n[1/12] Evaluating LoRA on IMDB..."
python scripts/evaluation/evaluate_models.py \
    --model_type lora \
    --dataset imdb \
    --checkpoint_path results/lora/imdb/best_model

echo -e "\n[2/12] Evaluating LoRA on SST-2..."
python scripts/evaluation/evaluate_models.py \
    --model_type lora \
    --dataset sst2 \
    --checkpoint_path results/lora/sst2/best_model

echo -e "\n[3/12] Evaluating LoRA on WikiText-2..."
python scripts/evaluation/evaluate_models.py \
    --model_type lora \
    --dataset wikitext2 \
    --checkpoint_path results/lora/wikitext2/best_model

# Sparse LoRA evaluations
echo -e "\n[4/12] Evaluating Sparse LoRA on IMDB..."
python scripts/evaluation/evaluate_models.py \
    --model_type sparse_lora \
    --dataset imdb \
    --checkpoint_path results/sparse_lora/imdb/best_model

echo -e "\n[5/12] Evaluating Sparse LoRA on SST-2..."
python scripts/evaluation/evaluate_models.py \
    --model_type sparse_lora \
    --dataset sst2 \
    --checkpoint_path results/sparse_lora/sst2/best_model

echo -e "\n[6/12] Evaluating Sparse LoRA on WikiText-2..."
python scripts/evaluation/evaluate_models.py \
    --model_type sparse_lora \
    --dataset wikitext2 \
    --checkpoint_path results/sparse_lora/wikitext2/best_model

# QLoRA evaluations
echo -e "\n[7/12] Evaluating QLoRA on IMDB..."
python scripts/evaluation/evaluate_models.py \
    --model_type qlora \
    --dataset imdb \
    --checkpoint_path results/qlora/imdb/best_model

echo -e "\n[8/12] Evaluating QLoRA on SST-2..."
python scripts/evaluation/evaluate_models.py \
    --model_type qlora \
    --dataset sst2 \
    --checkpoint_path results/qlora/sst2/best_model

echo -e "\n[9/12] Evaluating QLoRA on WikiText-2..."
python scripts/evaluation/evaluate_models.py \
    --model_type qlora \
    --dataset wikitext2 \
    --checkpoint_path results/qlora/wikitext2/best_model

# HiRA evaluations
echo -e "\n[10/12] Evaluating HiRA on IMDB..."
python scripts/evaluation/evaluate_models.py \
    --model_type hira \
    --dataset imdb \
    --checkpoint_path results/hira/imdb/best_model

echo -e "\n[11/12] Evaluating HiRA on SST-2..."
python scripts/evaluation/evaluate_models.py \
    --model_type hira \
    --dataset sst2 \
    --checkpoint_path results/hira/sst2/best_model

echo -e "\n[12/12] Evaluating HiRA on WikiText-2..."
python scripts/evaluation/evaluate_models.py \
    --model_type hira \
    --dataset wikitext2 \
    --checkpoint_path results/hira/wikitext2/best_model

echo -e "\n================================"
echo "All evaluations complete!"
echo "================================"
