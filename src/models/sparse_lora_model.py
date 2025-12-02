"""
Sparse LoRA model implementation with sparsity constraints.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)
from typing import Optional, Dict, Any
import copy


class SparseLoRAModel:
    """
    LoRA model with sparsity constraints.
    Implements magnitude-based pruning and L1 regularization.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        task_type: str = "classification",
        num_labels: Optional[int] = 2,
        lora_config: Optional[Dict[str, Any]] = None,
        sparsity_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Sparse LoRA model.

        Args:
            model_name: Base model name
            task_type: Task type (classification or generation)
            num_labels: Number of labels for classification
            lora_config: LoRA configuration dictionary
            sparsity_config: Sparsity configuration dictionary
        """
        self.model_name = model_name
        self.task_type = task_type
        self.num_labels = num_labels
        self.model = None
        self.lora_config = lora_config or self._default_lora_config()
        self.sparsity_config = sparsity_config or self._default_sparsity_config()
        self.pruning_step = 0

    def _default_lora_config(self) -> Dict[str, Any]:
        """Default LoRA configuration."""
        return {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": ["q_lin", "v_lin"],
            "bias": "none",
        }

    def _default_sparsity_config(self) -> Dict[str, Any]:
        """Default sparsity configuration."""
        return {
            "sparsity_method": "magnitude",  # or "l1", "topk"
            "sparsity_ratio": 0.5,  # target 50% sparsity
            "pruning_schedule": "gradual",  # or "oneshot"
            "pruning_freq": 100,  # steps between pruning
            "l1_lambda": 0.001,  # L1 regularization coefficient
        }

    def load_base_model(self):
        """Load the base model."""
        print(f"Loading base model: {self.model_name}")

        if self.task_type == "classification":
            config = AutoConfig.from_pretrained(self.model_name)
            config.num_labels = self.num_labels
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=config,
            )
        elif self.task_type == "generation":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        elif self.task_type == "masked_lm":
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        return self.model

    def apply_lora(self):
        """Apply LoRA adapters to the model."""
        if self.model is None:
            self.load_base_model()

        # Determine task type for PEFT
        if self.task_type == "classification":
            peft_task_type = TaskType.SEQ_CLS
        elif self.task_type == "generation":
            peft_task_type = TaskType.CAUSAL_LM
        elif self.task_type == "masked_lm":
            peft_task_type = TaskType.FEATURE_EXTRACTION
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        # Create LoRA config
        peft_config = LoraConfig(
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            lora_dropout=self.lora_config["lora_dropout"],
            target_modules=self.lora_config["target_modules"],
            bias=self.lora_config["bias"],
            task_type=peft_task_type,
        )

        # Apply LoRA
        print("Applying LoRA adapters with sparsity constraints...")
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        return self.model

    def get_model(self):
        """Get the Sparse LoRA-adapted model."""
        if self.model is None:
            self.load_base_model()
            self.apply_lora()
        return self.model

    def apply_magnitude_pruning(self, sparsity_ratio: float):
        """
        Apply magnitude-based pruning to LoRA parameters.

        Args:
            sparsity_ratio: Fraction of weights to prune (0-1)
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Only prune LoRA parameters
                if param.requires_grad and ("lora_A" in name or "lora_B" in name):
                    # Calculate threshold for this parameter
                    abs_param = torch.abs(param.data)
                    threshold = torch.quantile(abs_param.flatten(), sparsity_ratio)

                    # Create mask and apply
                    mask = (abs_param > threshold).float()
                    param.data *= mask

    def apply_topk_pruning(self, k_ratio: float):
        """
        Keep only top-k weights by magnitude.

        Args:
            k_ratio: Fraction of weights to keep (0-1)
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and ("lora_A" in name or "lora_B" in name):
                    # Calculate k
                    k = int(param.numel() * k_ratio)

                    # Get top-k values
                    flat_param = param.data.flatten()
                    topk_values, topk_indices = torch.topk(torch.abs(flat_param), k)
                    threshold = topk_values[-1]

                    # Create mask
                    mask = (torch.abs(param.data) >= threshold).float()
                    param.data *= mask

    def compute_l1_loss(self) -> torch.Tensor:
        """
        Compute L1 regularization loss on LoRA parameters.

        Returns:
            L1 loss tensor
        """
        l1_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for name, param in self.model.named_parameters():
            if param.requires_grad and ("lora_A" in name or "lora_B" in name):
                l1_loss += torch.abs(param).sum()

        return l1_loss * self.sparsity_config["l1_lambda"]

    def should_prune(self) -> bool:
        """Check if pruning should be applied at this step."""
        if self.sparsity_config["pruning_schedule"] == "oneshot":
            return self.pruning_step == 0
        elif self.sparsity_config["pruning_schedule"] == "gradual":
            return self.pruning_step % self.sparsity_config["pruning_freq"] == 0
        return False

    def apply_sparsity(self):
        """Apply sparsity based on configured method."""
        method = self.sparsity_config["sparsity_method"]
        sparsity_ratio = self.sparsity_config["sparsity_ratio"]

        if method == "magnitude":
            self.apply_magnitude_pruning(sparsity_ratio)
        elif method == "topk":
            self.apply_topk_pruning(1.0 - sparsity_ratio)
        # L1 regularization is applied during training, not here

        self.pruning_step += 1

    def get_sparsity_stats(self) -> Dict[str, float]:
        """Get current sparsity statistics."""
        total_params = 0
        zero_params = 0

        for name, param in self.model.named_parameters():
            if param.requires_grad and ("lora_A" in name or "lora_B" in name):
                total_params += param.numel()
                zero_params += (param.data == 0).sum().item()

        sparsity = (zero_params / total_params * 100) if total_params > 0 else 0

        return {
            "sparsity_percentage": sparsity,
            "zero_parameters": zero_params,
            "total_lora_parameters": total_params,
            "nonzero_parameters": total_params - zero_params,
        }

    def save_model(self, save_path: str):
        """Save the Sparse LoRA adapters."""
        if self.model is None:
            raise ValueError("No model to save")

        print(f"Saving Sparse LoRA adapters to {save_path}")
        self.model.save_pretrained(save_path)

        # Also save sparsity stats
        import json
        from pathlib import Path

        stats = self.get_sparsity_stats()
        stats_path = Path(save_path) / "sparsity_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    def load_model(self, adapter_path: str):
        """Load Sparse LoRA adapters."""
        print(f"Loading Sparse LoRA adapters from {adapter_path}")

        if self.model is None:
            self.load_base_model()

        self.model = PeftModel.from_pretrained(self.model, adapter_path)

        return self.model

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        if self.model is None:
            raise ValueError("Model not initialized")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        info = {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100,
            "lora_rank": self.lora_config["r"],
            "lora_alpha": self.lora_config["lora_alpha"],
            "target_modules": self.lora_config["target_modules"],
            "sparsity_method": self.sparsity_config["sparsity_method"],
            "target_sparsity": self.sparsity_config["sparsity_ratio"],
        }

        # Add current sparsity stats
        info.update(self.get_sparsity_stats())

        return info


def create_sparse_lora_model(
    model_name: str,
    task_type: str,
    num_labels: Optional[int] = None,
    lora_config: Optional[Dict[str, Any]] = None,
    sparsity_config: Optional[Dict[str, Any]] = None,
):
    """
    Factory function to create a Sparse LoRA model.

    Args:
        model_name: Base model name
        task_type: Task type (classification or generation)
        num_labels: Number of labels for classification
        lora_config: LoRA configuration dictionary
        sparsity_config: Sparsity configuration dictionary

    Returns:
        Sparse LoRA-adapted model
    """
    sparse_lora_model = SparseLoRAModel(
        model_name=model_name,
        task_type=task_type,
        num_labels=num_labels,
        lora_config=lora_config,
        sparsity_config=sparsity_config,
    )

    return sparse_lora_model.get_model()
