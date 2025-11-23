#!/usr/bin/env python3
"""
Training script for QLoRA (Quantized LoRA) model.
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.models.qlora_model import QLoRAModel
from src.utils.data_utils import DatasetLoader
from src.utils.metrics import MetricsTracker, measure_inference_time


def parse_args():
    parser = argparse.ArgumentParser(description="Train QLoRA model")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["sst2", "imdb", "wikitext2"],
        help="Dataset to train on",
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
        default="results/qlora",
        help="Output directory for results",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    return parser.parse_args()


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, optimizer, scheduler, device, metrics_tracker):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def evaluate(model, dataloader, device, task_type="classification"):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            # Track metrics
            total_loss += loss.item()
            num_batches += 1

            if task_type == "classification":
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    results = {"loss": avg_loss}

    if task_type == "classification":
        results["predictions"] = all_predictions
        results["labels"] = all_labels
    elif task_type == "generation":
        results["perplexity"] = torch.exp(torch.tensor(avg_loss)).item()

    return results


def main():
    args = parse_args()

    # Check if CUDA is available for QLoRA
    if not torch.cuda.is_available():
        print("WARNING: QLoRA requires CUDA. Running on CPU may not work properly.")
        print("Please use a GPU for QLoRA training.")

    # Load configuration
    config = load_config(args.config)
    dataset_config = config["datasets"][args.dataset]
    training_config = config["training"]
    qlora_config = config["qlora"]

    # Setup output directory
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(str(output_dir))

    print("\n" + "=" * 60)
    print(f"Training QLoRA on {args.dataset.upper()}")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    dataset_loader = DatasetLoader(
        dataset_name=args.dataset,
        config=dataset_config,
        tokenizer_name=config["model"]["name"],
    )

    dataloaders = dataset_loader.get_dataloaders(
        batch_size=training_config["batch_size"],
        num_workers=training_config.get("dataloader_num_workers", 4),
        data_dir=args.data_dir,
    )

    # Initialize model
    print("\nInitializing QLoRA model with 4-bit quantization...")
    task_type = dataset_config["task_type"]
    qlora_model = QLoRAModel(
        model_name=config["model"]["name"],
        task_type=task_type,
        num_labels=dataset_config.get("num_labels"),
        lora_config=qlora_config,
    )

    model = qlora_model.get_model()
    # Note: QLoRA model is already on the correct device due to device_map

    # Log model info
    model_info = qlora_model.get_model_info()
    metrics_tracker.log_metrics(model_info, prefix="model")
    print(f"\nModel Info:")
    print(f"  Total parameters: {model_info['total_parameters']:,}")
    print(f"  Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"  Trainable %: {model_info['trainable_percentage']:.2f}%")
    print(f"  Quantization: 4-bit NF4")
    print(f"  Double quantization: {qlora_config.get('double_quant', True)}")

    # Get memory footprint
    memory_footprint = qlora_model.get_memory_footprint()
    print(f"\nMemory Footprint:")
    print(f"  Total memory: {memory_footprint['total_memory_mb']:.2f} MB")
    print(f"  Quantized model: {memory_footprint['quantized_model_mb']:.2f} MB")
    print(f"  LoRA adapters: {memory_footprint['lora_adapters_mb']:.2f} MB")
    metrics_tracker.log_metrics(memory_footprint, prefix="memory_footprint")

    # Setup optimizer and scheduler
    print("\nSetting up optimizer and scheduler...")
    optimizer = AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    num_training_steps = len(dataloaders["train"]) * training_config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config["warmup_steps"],
        num_training_steps=num_training_steps,
    )

    # Training loop
    print(f"\nStarting training for {training_config['num_epochs']} epochs...")
    metrics_tracker.start_timer()

    best_val_loss = float("inf")

    for epoch in range(training_config["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{training_config['num_epochs']} ---")

        # Train
        train_loss = train_epoch(
            model, dataloaders["train"], optimizer, scheduler, args.device, metrics_tracker
        )

        print(f"Training Loss: {train_loss:.4f}")
        metrics_tracker.log_metrics({"train_loss": train_loss}, prefix=f"epoch_{epoch+1}")

        # Evaluate on validation set
        if "validation" in dataloaders:
            eval_split = "validation"
        elif "test" in dataloaders:
            eval_split = "test"
        else:
            print("No validation or test set available for evaluation")
            continue

        eval_results = evaluate(model, dataloaders[eval_split], args.device, task_type)

        print(f"Validation Loss: {eval_results['loss']:.4f}")

        if task_type == "classification":
            # Compute classification metrics
            class_metrics = metrics_tracker.compute_classification_metrics(
                eval_results["predictions"], eval_results["labels"]
            )
            print(f"Validation Accuracy: {class_metrics['accuracy']:.4f}")
            print(f"Validation F1: {class_metrics['f1']:.4f}")
            metrics_tracker.log_metrics(class_metrics, prefix=f"epoch_{epoch+1}/validation")
        elif task_type == "generation":
            print(f"Validation Perplexity: {eval_results['perplexity']:.4f}")
            metrics_tracker.log_metrics(
                {"perplexity": eval_results["perplexity"]},
                prefix=f"epoch_{epoch+1}/validation"
            )

        # Save best model
        if eval_results["loss"] < best_val_loss:
            best_val_loss = eval_results["loss"]
            print(f"New best model! Saving to {output_dir / 'best_model'}")
            qlora_model.save_model(str(output_dir / "best_model"))

        # Track memory usage
        memory_metrics = metrics_tracker.get_memory_usage()
        metrics_tracker.log_metrics(memory_metrics, prefix="memory")

    training_time = metrics_tracker.stop_timer()
    print(f"\nTotal training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    metrics_tracker.log_metrics({"training_time_seconds": training_time})

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    # Load best model
    qlora_model.load_model(str(output_dir / "best_model"))
    model = qlora_model.model  # Device is already set

    # Evaluate on test set
    if "test" in dataloaders:
        test_results = evaluate(model, dataloaders["test"], args.device, task_type)
        print(f"\nTest Loss: {test_results['loss']:.4f}")

        if task_type == "classification":
            test_metrics = metrics_tracker.compute_classification_metrics(
                test_results["predictions"], test_results["labels"]
            )
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Test F1: {test_metrics['f1']:.4f}")
            print(f"Test Precision: {test_metrics['precision']:.4f}")
            print(f"Test Recall: {test_metrics['recall']:.4f}")
            metrics_tracker.log_metrics(test_metrics, prefix="test")
        elif task_type == "generation":
            print(f"Test Perplexity: {test_results['perplexity']:.4f}")
            metrics_tracker.log_metrics({"perplexity": test_results["perplexity"]}, prefix="test")

    # Measure inference time
    print("\nMeasuring inference time...")
    inference_metrics = measure_inference_time(
        model,
        dataloaders["test"] if "test" in dataloaders else dataloaders["validation"],
        args.device,
        num_batches=10,
    )
    print(f"Avg inference time per sample: {inference_metrics['avg_inference_time_per_sample']:.6f} seconds")
    print(f"Throughput: {inference_metrics['throughput_samples_per_second']:.2f} samples/sec")
    metrics_tracker.log_metrics(inference_metrics, prefix="inference")

    # Get final model size
    final_model_info = metrics_tracker.get_model_size(model)
    metrics_tracker.log_metrics(final_model_info, prefix="final_model")

    # Final memory footprint
    final_memory = qlora_model.get_memory_footprint()
    print(f"\nFinal Memory Footprint:")
    print(f"  Total memory: {final_memory['total_memory_mb']:.2f} MB")
    print(f"  Memory reduction vs full precision: ~75%")
    metrics_tracker.log_metrics(final_memory, prefix="final_memory_footprint")

    # Save metrics
    metrics_tracker.save_metrics(f"{args.dataset}_qlora_metrics.json")
    metrics_tracker.print_summary()

    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
