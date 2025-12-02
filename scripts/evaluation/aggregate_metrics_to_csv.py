#!/usr/bin/env python3
"""
Aggregate all metrics from JSON files into a single CSV for easy comparison.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any


def load_metrics_file(metrics_file: Path) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {metrics_file}: {e}")
        return {}


def extract_method_and_dataset(metrics_file: Path) -> tuple:
    """Extract method and dataset from file path."""
    # File pattern: results/{method}/{dataset}/{dataset}_{method}_metrics.json
    parts = metrics_file.parts
    method = parts[-3]  # e.g., 'lora', 'sparse_lora', 'qlora', 'hira'
    dataset = parts[-2]  # e.g., 'sst2', 'imdb', 'wikitext2'
    return method, dataset


def get_metric_value(metrics: Dict, *keys):
    """Get a metric value trying multiple possible keys."""
    for key in keys:
        if key in metrics:
            value = metrics[key]
            # Handle nested keys like "model/trainable_parameters"
            if isinstance(value, dict):
                continue
            return value
    return None


def format_value(value, decimals=4):
    """Format a value for CSV output."""
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if abs(value) >= 1000:
            return f"{value:,.{max(0, decimals-2)}f}"
        else:
            return f"{value:.{decimals}f}"
    return str(value)


def aggregate_metrics_to_csv(results_dir: Path, output_file: Path):
    """
    Aggregate all metrics JSON files into a single CSV.

    Args:
        results_dir: Directory containing results (e.g., results/)
        output_file: Output CSV file path
    """
    # Find all metrics JSON files
    metrics_files = list(results_dir.glob("*/*/*_metrics.json"))

    if not metrics_files:
        print(f"No metrics files found in {results_dir}")
        return

    print(f"Found {len(metrics_files)} metrics files")

    # Define all metrics to extract (in order for CSV columns)
    metric_definitions = [
        # Basic info
        ("method", None),
        ("dataset", None),

        # Model size and parameters
        ("total_parameters", ["total_parameters", "model/total_parameters", "final_model/total_parameters"]),
        ("trainable_parameters", ["trainable_parameters", "model/trainable_parameters", "final_model/trainable_parameters"]),
        ("trainable_percentage", ["trainable_percentage", "model/trainable_percentage", "final_model/trainable_percentage"]),
        ("model_size_mb", ["model_size_mb", "final_model/model_size_mb"]),

        # LoRA configuration
        ("lora_rank", ["model/lora_rank", "lora_rank"]),
        ("lora_alpha", ["model/lora_alpha", "lora_alpha"]),

        # Sparsity (for Sparse LoRA)
        ("sparsity_percentage", ["sparsity_percentage"]),
        ("zero_parameters", ["zero_parameters"]),

        # Training metrics
        ("training_time_seconds", ["training_time_seconds"]),
        ("training_time_minutes", None),  # Computed

        # Test performance (classification)
        ("test_accuracy", ["test_accuracy", "test/accuracy"]),
        ("test_f1", ["test_f1", "test/f1"]),
        ("test_precision", ["test_precision", "test/precision"]),
        ("test_recall", ["test_recall", "test/recall"]),

        # Test performance (generation)
        ("test_perplexity", ["test_perplexity", "test/perplexity"]),

        # Inference performance
        ("inference_time_per_sample", ["avg_inference_time_per_sample", "inference/avg_inference_time_per_sample"]),
        ("inference_throughput_samples_per_sec", ["throughput_samples_per_second", "inference/throughput_samples_per_second"]),

        # Memory usage
        ("peak_gpu_memory_gb", ["gpu_memory_max_allocated_gb", "memory/gpu_memory_max_allocated_gb"]),
        ("gpu_memory_allocated_gb", ["gpu_memory_allocated_gb", "memory/gpu_memory_allocated_gb"]),
        ("system_memory_used_gb", ["system_memory_used_gb", "memory/system_memory_used_gb"]),
    ]

    # Collect all rows
    rows = []

    for metrics_file in sorted(metrics_files):
        print(f"Processing: {metrics_file}")

        # Extract method and dataset from path
        method, dataset = extract_method_and_dataset(metrics_file)

        # Load metrics
        metrics = load_metrics_file(metrics_file)
        if not metrics:
            continue

        # Build row
        row = {}
        for col_name, keys in metric_definitions:
            if col_name == "method":
                row[col_name] = method
            elif col_name == "dataset":
                row[col_name] = dataset
            elif col_name == "training_time_minutes":
                # Compute from seconds
                seconds = get_metric_value(metrics, "training_time_seconds")
                if seconds is not None:
                    row[col_name] = format_value(seconds / 60, decimals=2)
                else:
                    row[col_name] = "N/A"
            elif keys:
                value = get_metric_value(metrics, *keys)
                # Format based on metric type
                if col_name in ["total_parameters", "trainable_parameters", "zero_parameters"]:
                    row[col_name] = format_value(value, decimals=0)
                elif col_name in ["trainable_percentage"]:
                    row[col_name] = format_value(value, decimals=2)
                elif col_name in ["test_accuracy", "test_f1", "test_precision", "test_recall"]:
                    row[col_name] = format_value(value, decimals=4)
                elif col_name in ["test_perplexity"]:
                    row[col_name] = format_value(value, decimals=2)
                elif col_name in ["inference_time_per_sample"]:
                    row[col_name] = format_value(value, decimals=6)
                elif col_name in ["training_time_seconds"]:
                    row[col_name] = format_value(value, decimals=1)
                else:
                    row[col_name] = format_value(value, decimals=3)
            else:
                row[col_name] = "N/A"

        rows.append(row)

    # Write to CSV
    if rows:
        fieldnames = [col_name for col_name, _ in metric_definitions]

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n{'='*60}")
        print(f"Successfully aggregated {len(rows)} metric files")
        print(f"Output saved to: {output_file}")
        print(f"{'='*60}\n")

        # Print summary table to console
        print("\nSummary of aggregated metrics:")
        print("-" * 60)
        for row in rows:
            print(f"{row['method']:15s} | {row['dataset']:10s} | Params: {row['trainable_parameters']:>12s} | Acc/PPL: {row.get('test_accuracy', row.get('test_perplexity', 'N/A')):>8s}")
    else:
        print("No valid metrics to aggregate")


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "results"
    output_file = results_dir / "all_metrics_comparison.csv"

    print("="*60)
    print("Aggregating Metrics to CSV")
    print("="*60)
    print(f"Results directory: {results_dir}")
    print(f"Output file: {output_file}")
    print()

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    aggregate_metrics_to_csv(results_dir, output_file)


if __name__ == "__main__":
    main()
