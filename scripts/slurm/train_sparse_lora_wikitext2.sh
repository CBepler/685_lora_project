#!/bin/bash
#SBATCH --job-name=sparse_wiki
#SBATCH --output=logs/sparse_lora_wikitext2_%j.out
#SBATCH --error=logs/sparse_lora_wikitext2_%j.err
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# Print job information
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="

# Set project directory (automatically detect from script location)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd $PROJECT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Install requirements if needed
echo "Installing requirements..."
pip install -q -r requirements.txt

# Download datasets if not already downloaded
echo "Checking for datasets..."
if [ ! -d "data/sst2" ] && [ ! -d "data/imdb" ] && [ ! -d "data/wikitext2" ]; then
    echo "Downloading datasets..."
    python scripts/prepare_datasets.py
else
    echo "Datasets already present."
fi

# Print GPU information
echo "========================================="
echo "GPU Information:"
nvidia-smi
echo "========================================="

# Run training
echo "Starting Sparse LoRA training on WikiText-2..."
python scripts/training/train_sparse_lora.py \
    --dataset wikitext2 \
    --config configs/config.yaml \
    --output_dir results/sparse_lora \
    --data_dir data

# Print completion information
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
