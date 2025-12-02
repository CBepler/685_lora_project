#!/usr/bin/env python3
"""
Unified evaluation script for all LoRA variants.
Evaluates trained models and generates comprehensive metrics.
"""

import os
import sys
import yaml
import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.lora_model import LoRAModel
from src.models.sparse_lora_model import SparseLoRAModel
from src.models.qlora_model import QLoRAModel
from src.models.hira_model import HiRAModel
from src.utils.data_utils import DatasetLoader
from src.utils.metrics import MetricsTracker, measure_inference_time


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained LoRA models")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["lora", "sparse_lora", "qlora", "hira"],
        help="Type of model to evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["sst2", "imdb", "wikitext2"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (defaults to checkpoint_path)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    return parser.parse_args()


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_type: str, checkpoint_path: str, dataset_config: dict, config: dict):
    """Load the appropriate model type with adapters."""
    print(f"\n{'='*60}")
    print(f"Loading {model_type.upper()} model from {checkpoint_path}")
    print(f"{'='*60}\n")

    base_model = dataset_config.get("model_name", config["model"]["name"])
    task_type = dataset_config["task_type"]
    num_labels = dataset_config.get("num_labels", None)

    if model_type == "lora":
        model_wrapper = LoRAModel(
            model_name=base_model,
            task_type=task_type,
            num_labels=num_labels,
            lora_config=config["lora"],
        )
    elif model_type == "sparse_lora":
        model_wrapper = SparseLoRAModel(
            model_name=base_model,
            task_type=task_type,
            num_labels=num_labels,
            lora_config=config["sparse_lora"],
            sparsity_config=config["sparse_lora"],
        )
    elif model_type == "qlora":
        model_wrapper = QLoRAModel(
            model_name=base_model,
            task_type=task_type,
            num_labels=num_labels,
            lora_config=config["qlora"],
            quantization_config=config["qlora"],
        )
    elif model_type == "hira":
        model_wrapper = HiRAModel(
            model_name=base_model,
            task_type=task_type,
            num_labels=num_labels,
            lora_config=config["hira"],
            hira_config=config["hira"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load the trained adapters
    model = model_wrapper.load_model(checkpoint_path)
    model.eval()

    return model, model_wrapper


def evaluate(model, dataloader, device, task_type):
    """Evaluate model on dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    print("\nEvaluating model...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)

            # Get predictions
            if task_type == "classification":
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

            # Track loss
            if hasattr(outputs, "loss") and outputs.loss is not None:
                total_loss += outputs.loss.item()
                num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    return {
        "predictions": all_preds,
        "labels": all_labels,
        "avg_loss": avg_loss,
    }


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    dataset_config = config["datasets"][args.dataset]

    # Set output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint_path).parent)

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(args.output_dir)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    base_model = dataset_config.get("model_name", config["model"]["name"])

    data_loader = DatasetLoader(
        dataset_name=args.dataset,
        config=dataset_config,
        tokenizer_name=base_model,
    )

    dataloaders = data_loader.get_dataloaders(
        batch_size=args.batch_size,
        num_workers=0,  # Use 0 for evaluation to avoid multiprocessing issues
        data_dir="data",
    )

    # Get test dataloader (use validation if test is not available)
    test_dataloader = dataloaders.get("test", dataloaders.get("validation"))

    # Load model
    model, model_wrapper = load_model(
        args.model_type,
        args.checkpoint_path,
        dataset_config,
        config,
    )
    model.to(args.device)

    # Get model info
    model_info = model_wrapper.get_model_info()
    print("\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    # Log model info
    metrics_tracker.log_metrics(model_info, prefix="model")

    # Get memory usage before evaluation
    memory_before = metrics_tracker.get_memory_usage()
    metrics_tracker.log_metrics(memory_before, prefix="memory/before")

    # Evaluate
    task_type = dataset_config["task_type"]
    eval_results = evaluate(model, test_dataloader, args.device, task_type)

    # Compute metrics based on task type
    if task_type == "classification":
        classification_metrics = metrics_tracker.compute_classification_metrics(
            eval_results["predictions"],
            eval_results["labels"],
        )
        metrics_tracker.log_metrics(classification_metrics, prefix="test")

        print("\nTest Results:")
        for key, value in classification_metrics.items():
            print(f"  {key}: {value:.4f}")

    elif task_type in ["generation", "masked_lm"]:
        perplexity = metrics_tracker.compute_perplexity(eval_results["avg_loss"])
        metrics_tracker.log_metrics(
            {
                "loss": eval_results["avg_loss"],
                "perplexity": perplexity,
            },
            prefix="test",
        )

        print(f"\nTest Results:")
        print(f"  Loss: {eval_results['avg_loss']:.4f}")
        print(f"  Perplexity: {perplexity:.4f}")

    # Measure inference time
    print("\nMeasuring inference performance...")
    inference_metrics = measure_inference_time(
        model,
        test_dataloader,
        args.device,
        num_batches=10,
    )
    metrics_tracker.log_metrics(inference_metrics, prefix="inference")

    print(f"\nInference Performance:")
    print(f"  Avg time per sample: {inference_metrics['avg_inference_time_per_sample']:.4f}s")
    print(f"  Throughput: {inference_metrics['throughput_samples_per_second']:.2f} samples/sec")

    # Get final memory usage
    memory_after = metrics_tracker.get_memory_usage()
    metrics_tracker.log_metrics(memory_after, prefix="memory/after")

    # Get model size
    model_size = metrics_tracker.get_model_size(model)
    metrics_tracker.log_metrics(model_size, prefix="final_model")

    # For Sparse LoRA, compute sparsity
    if args.model_type == "sparse_lora":
        sparsity_metrics = metrics_tracker.get_sparsity(model)
        metrics_tracker.log_metrics(sparsity_metrics, prefix="sparsity")

        print(f"\nSparsity Metrics:")
        print(f"  Sparsity: {sparsity_metrics['sparsity_percentage']:.2f}%")
        print(f"  Zero parameters: {sparsity_metrics['zero_parameters']:,}")
        print(f"  Total trainable: {sparsity_metrics['total_trainable']:,}")

    # Save metrics
    metrics_filename = f"{args.dataset}_{args.model_type}_metrics.json"
    metrics_tracker.save_metrics(metrics_filename)

    # Print summary
    metrics_tracker.print_summary()

    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
