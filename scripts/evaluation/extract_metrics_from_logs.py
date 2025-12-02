#!/usr/bin/env python3
"""
Extract metrics from training logs and create comprehensive metrics files.
This is faster than re-running full evaluations since training already computed these metrics.
"""

import re
import json
from pathlib import Path
import sys

# Model/dataset combinations to process
COMBINATIONS = [
    ("lora", "imdb", "40293858"),
    ("lora", "sst2", "40293857"),
    ("lora", "wikitext2", "40294194"),
    ("sparse_lora", "imdb", "40294631"),
    ("sparse_lora", "sst2", "40294630"),
    ("sparse_lora", "wikitext2", "40294633"),
    ("qlora", "imdb", "40296840"),
    ("qlora", "sst2", "40296839"),
    ("qlora", "wikitext2", "40296841"),
    ("hira", "imdb", "40293867"),
    ("hira", "sst2", "40293866"),
    ("hira", "wikitext2", "40294197"),
]


def extract_metrics_from_log(log_path):
    """Extract all metrics from a training log file."""
    metrics = {}

    if not log_path.exists():
        print(f"Warning: Log file not found: {log_path}")
        return metrics

    with open(log_path, 'r') as f:
        content = f.read()

    # Extract model parameters
    param_match = re.search(r'trainable params:\s*([0-9,]+)', content)
    if param_match:
        metrics['trainable_parameters'] = int(param_match.group(1).replace(',', ''))

    total_param_match = re.search(r'all params:\s*([0-9,]+)', content)
    if total_param_match:
        metrics['total_parameters'] = int(total_param_match.group(1).replace(',', ''))

    trainable_pct_match = re.search(r'trainable%:\s*([0-9.]+)', content)
    if trainable_pct_match:
        metrics['trainable_percentage'] = float(trainable_pct_match.group(1))

    # Extract test/validation accuracy (look for the last occurrence)
    accuracy_matches = list(re.finditer(r"Test Accuracy:\s*([0-9.]+)", content))
    if not accuracy_matches:
        accuracy_matches = list(re.finditer(r"Validation Accuracy:\s*([0-9.]+)", content))
    if accuracy_matches:
        metrics['test_accuracy'] = float(accuracy_matches[-1].group(1))

    # Extract perplexity for language modeling tasks
    perplexity_matches = list(re.finditer(r"Test Perplexity:\s*([0-9.]+)", content))
    if not perplexity_matches:
        perplexity_matches = list(re.finditer(r"Validation Perplexity:\s*([0-9.]+)", content))
    if perplexity_matches:
        metrics['test_perplexity'] = float(perplexity_matches[-1].group(1))

    # Extract F1 score
    f1_matches = list(re.finditer(r"F1:\s*([0-9.]+)", content))
    if f1_matches:
        metrics['test_f1'] = float(f1_matches[-1].group(1))

    # Extract training time
    time_match = re.search(r'Training time:\s*([0-9.]+)\s*seconds', content)
    if time_match:
        metrics['training_time_seconds'] = float(time_match.group(1))

    # Extract memory usage
    gpu_mem_match = re.search(r'Peak GPU Memory:\s*([0-9.]+)\s*GB', content)
    if gpu_mem_match:
        metrics['peak_gpu_memory_gb'] = float(gpu_mem_match.group(1))

    # Extract throughput
    throughput_match = re.search(r'Throughput:\s*([0-9.]+)\s*samples/sec', content)
    if throughput_match:
        metrics['inference_throughput_samples_per_sec'] = float(throughput_match.group(1))

    # Extract model size
    model_size_match = re.search(r'Model size:\s*([0-9.]+)\s*MB', content)
    if model_size_match:
        metrics['model_size_mb'] = float(model_size_match.group(1))

    # Extract sparsity (for sparse models)
    sparsity_match = re.search(r'Sparsity:\s*([0-9.]+)%', content)
    if sparsity_match:
        metrics['sparsity_percentage'] = float(sparsity_match.group(1))

    return metrics


def load_existing_metrics(results_dir):
    """Load existing metrics JSON if available."""
    metrics_files = list(results_dir.glob("*_metrics.json"))
    if metrics_files:
        with open(metrics_files[0], 'r') as f:
            return json.load(f)
    return {}


def load_sparsity_stats(model_dir):
    """Load sparsity stats if available."""
    sparsity_file = model_dir / "best_model" / "sparsity_stats.json"
    if sparsity_file.exists():
        with open(sparsity_file, 'r') as f:
            return json.load(f)
    return {}


def main():
    project_root = Path(__file__).parent.parent.parent
    logs_dir = project_root / "logs"
    results_dir = project_root / "results"

    print("Extracting metrics from training logs...")
    print("=" * 60)

    for model_type, dataset, job_id in COMBINATIONS:
        print(f"\nProcessing: {model_type} on {dataset} (Job {job_id})")

        # Find log file
        log_pattern = f"{model_type}_{dataset}_{job_id}.out"
        log_file = logs_dir / log_pattern

        if not log_file.exists():
            print(f"  Warning: Log file not found: {log_file}")
            continue

        # Extract metrics from log
        log_metrics = extract_metrics_from_log(log_file)

        # Load existing metrics JSON if available
        model_results_dir = results_dir / model_type / dataset
        existing_metrics = load_existing_metrics(model_results_dir)

        # Merge metrics (prefer existing JSON, supplement with log data)
        final_metrics = {**log_metrics, **existing_metrics}

        # Add sparsity stats for sparse_lora
        if model_type == "sparse_lora":
            sparsity_stats = load_sparsity_stats(model_results_dir)
            if sparsity_stats:
                final_metrics.update(sparsity_stats)

        # Save consolidated metrics
        output_file = model_results_dir / f"{dataset}_{model_type}_metrics.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)

        print(f"  ✓ Saved metrics to: {output_file}")
        print(f"  Metrics found: {len(final_metrics)} fields")

        # Print key metrics
        if 'test_accuracy' in final_metrics:
            print(f"    Accuracy: {final_metrics['test_accuracy']:.4f}")
        if 'test_perplexity' in final_metrics:
            print(f"    Perplexity: {final_metrics['test_perplexity']:.4f}")
        if 'trainable_parameters' in final_metrics:
            print(f"    Trainable params: {final_metrics['trainable_parameters']:,}")

    print("\n" + "=" * 60)
    print("Metrics extraction complete!")


if __name__ == "__main__":
    main()
