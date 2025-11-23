#!/bin/bash
# Master script to submit all 12 LoRA training jobs to SLURM

echo "========================================="
echo "LoRA Training Job Submission"
echo "Submitting all 12 experiments (4 methods x 3 datasets)"
echo "========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project directory
cd "$PROJECT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Array to store job IDs
declare -a JOB_IDS

echo ""
echo "Submitting LoRA jobs..."
JOB1=$(sbatch "$SCRIPT_DIR/train_lora_sst2.sh" | awk '{print $4}')
echo "  [1/12] LoRA + SST-2: Job ID $JOB1"
JOB_IDS+=($JOB1)

JOB2=$(sbatch "$SCRIPT_DIR/train_lora_imdb.sh" | awk '{print $4}')
echo "  [2/12] LoRA + IMDB: Job ID $JOB2"
JOB_IDS+=($JOB2)

JOB3=$(sbatch "$SCRIPT_DIR/train_lora_wikitext2.sh" | awk '{print $4}')
echo "  [3/12] LoRA + WikiText-2: Job ID $JOB3"
JOB_IDS+=($JOB3)

echo ""
echo "Submitting Sparse LoRA jobs..."
JOB4=$(sbatch "$SCRIPT_DIR/train_sparse_lora_sst2.sh" | awk '{print $4}')
echo "  [4/12] Sparse LoRA + SST-2: Job ID $JOB4"
JOB_IDS+=($JOB4)

JOB5=$(sbatch "$SCRIPT_DIR/train_sparse_lora_imdb.sh" | awk '{print $4}')
echo "  [5/12] Sparse LoRA + IMDB: Job ID $JOB5"
JOB_IDS+=($JOB5)

JOB6=$(sbatch "$SCRIPT_DIR/train_sparse_lora_wikitext2.sh" | awk '{print $4}')
echo "  [6/12] Sparse LoRA + WikiText-2: Job ID $JOB6"
JOB_IDS+=($JOB6)

echo ""
echo "Submitting QLoRA jobs..."
JOB7=$(sbatch "$SCRIPT_DIR/train_qlora_sst2.sh" | awk '{print $4}')
echo "  [7/12] QLoRA + SST-2: Job ID $JOB7"
JOB_IDS+=($JOB7)

JOB8=$(sbatch "$SCRIPT_DIR/train_qlora_imdb.sh" | awk '{print $4}')
echo "  [8/12] QLoRA + IMDB: Job ID $JOB8"
JOB_IDS+=($JOB8)

JOB9=$(sbatch "$SCRIPT_DIR/train_qlora_wikitext2.sh" | awk '{print $4}')
echo "  [9/12] QLoRA + WikiText-2: Job ID $JOB9"
JOB_IDS+=($JOB9)

echo ""
echo "Submitting HiRA jobs..."
JOB10=$(sbatch "$SCRIPT_DIR/train_hira_sst2.sh" | awk '{print $4}')
echo "  [10/12] HiRA + SST-2: Job ID $JOB10"
JOB_IDS+=($JOB10)

JOB11=$(sbatch "$SCRIPT_DIR/train_hira_imdb.sh" | awk '{print $4}')
echo "  [11/12] HiRA + IMDB: Job ID $JOB11"
JOB_IDS+=($JOB11)

JOB12=$(sbatch "$SCRIPT_DIR/train_hira_wikitext2.sh" | awk '{print $4}')
echo "  [12/12] HiRA + WikiText-2: Job ID $JOB12"
JOB_IDS+=($JOB12)

echo ""
echo "========================================="
echo "All 12 jobs submitted successfully!"
echo "========================================="
echo ""
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "To check job status, run:"
echo "  bash $SCRIPT_DIR/check_jobs.sh"
echo ""
echo "To view output logs:"
echo "  tail -f logs/lora_sst2_*.out"
echo ""
echo "To cancel all jobs:"
echo "  bash $SCRIPT_DIR/cancel_all_jobs.sh"
echo ""
