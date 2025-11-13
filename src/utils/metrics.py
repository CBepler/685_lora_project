"""
Metrics utilities for tracking model performance.
"""

import time
import torch
import psutil
import numpy as np
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
from pathlib import Path


class MetricsTracker:
    """Track and log metrics during training and evaluation."""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
        self.start_time = None

    def start_timer(self):
        """Start timing."""
        self.start_time = time.time()

    def stop_timer(self) -> float:
        """Stop timing and return elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        elapsed = time.time() - self.start_time
        self.start_time = None
        return elapsed

    def compute_classification_metrics(self, predictions, labels) -> Dict[str, float]:
        """Compute classification metrics."""
        preds = np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions
        labels = np.array(labels)

        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
            "precision": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        }

    def compute_perplexity(self, loss: float) -> float:
        """Compute perplexity from loss."""
        return np.exp(loss)

    def get_model_size(self, model) -> Dict[str, Any]:
        """Get model size information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Estimate model size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100 if total_params > 0 else 0,
            "model_size_mb": model_size_mb,
        }

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = {}

        # System memory
        memory_info["system_memory_percent"] = psutil.virtual_memory().percent
        memory_info["system_memory_used_gb"] = psutil.virtual_memory().used / (1024 ** 3)

        # GPU memory (if available)
        if torch.cuda.is_available():
            memory_info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
            memory_info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
            memory_info["gpu_memory_max_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)

        return memory_info

    def get_sparsity(self, model) -> Dict[str, float]:
        """Compute sparsity of model parameters."""
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            if param.requires_grad:
                total_params += param.numel()
                zero_params += (param.data == 0).sum().item()

        sparsity = (zero_params / total_params) * 100 if total_params > 0 else 0

        return {
            "sparsity_percentage": sparsity,
            "zero_parameters": zero_params,
            "total_trainable": total_params,
        }

    def log_metrics(self, metrics: Dict[str, Any], prefix: str = ""):
        """Log metrics to internal storage."""
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.metrics.update(metrics)

    def save_metrics(self, filename: str = "metrics.json"):
        """Save metrics to JSON file."""
        filepath = self.save_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")

    def print_summary(self):
        """Print a summary of collected metrics."""
        print("\n" + "=" * 60)
        print("METRICS SUMMARY")
        print("=" * 60)
        for key, value in self.metrics.items():
            if isinstance(value, float):
                print(f"{key:40s}: {value:.4f}")
            else:
                print(f"{key:40s}: {value}")
        print("=" * 60 + "\n")


def measure_inference_time(model, dataloader, device, num_batches=None):
    """Measure average inference time per sample."""
    model.eval()
    times = []
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if num_batches and i >= num_batches:
                break

            # Move batch to device
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            batch_size = inputs["input_ids"].size(0)

            # Time inference
            start = time.time()
            _ = model(**inputs)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start

            times.append(elapsed)
            total_samples += batch_size

    avg_time_per_sample = sum(times) / total_samples if total_samples > 0 else 0
    throughput = total_samples / sum(times) if sum(times) > 0 else 0

    return {
        "avg_inference_time_per_sample": avg_time_per_sample,
        "throughput_samples_per_second": throughput,
        "total_samples_measured": total_samples,
    }
