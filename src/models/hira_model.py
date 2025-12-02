"""
HiRA model implementation with high-rank adaptation.

Note: This is a simplified implementation using high-rank LoRA.
A full HiRA implementation would require custom layers with Hadamard product operations.
For this project, we approximate HiRA by using significantly higher rank than standard LoRA.
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


class HiRAModel:
    """
    High-Rank Adaptation (HiRA) model.

    This implementation uses high-rank LoRA as an approximation to HiRA.
    True HiRA uses Hadamard products for efficient high-rank updates.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        task_type: str = "classification",
        num_labels: Optional[int] = 2,
        lora_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize HiRA model.

        Args:
            model_name: Base model name
            task_type: Task type (classification or generation)
            num_labels: Number of labels for classification
            lora_config: LoRA configuration dictionary (with higher rank)
        """
        self.model_name = model_name
        self.task_type = task_type
        self.num_labels = num_labels
        self.model = None
        self.lora_config = lora_config or self._default_lora_config()

    def _default_lora_config(self) -> Dict[str, Any]:
        """
        Default HiRA configuration (high-rank LoRA).

        Note: HiRA uses higher rank than standard LoRA to achieve
        greater expressiveness. Typical LoRA uses r=8, while HiRA
        uses r=32 or higher.
        """
        return {
            "r": 32,  # Higher rank than standard LoRA (typically 8)
            "lora_alpha": 64,  # Scaled proportionally with rank
            "lora_dropout": 0.1,
            "target_modules": ["q_lin", "v_lin"],
            "bias": "none",
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

        print(f"Base model loaded: {self.model.__class__.__name__}")
        return self.model

    def apply_lora(self):
        """Apply high-rank LoRA adapters to the model."""
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

        # Create LoRA config with high rank
        peft_config = LoraConfig(
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            lora_dropout=self.lora_config["lora_dropout"],
            target_modules=self.lora_config["target_modules"],
            bias=self.lora_config["bias"],
            task_type=peft_task_type,
        )

        # Apply high-rank LoRA
        print(f"Applying high-rank LoRA adapters (rank={self.lora_config['r']})...")
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        return self.model

    def get_model(self):
        """Get the HiRA-adapted model."""
        if self.model is None:
            self.load_base_model()
            self.apply_lora()
        return self.model

    def save_model(self, save_path: str):
        """Save the HiRA adapters."""
        if self.model is None:
            raise ValueError("No model to save")

        print(f"Saving HiRA adapters to {save_path}")
        self.model.save_pretrained(save_path)

    def load_model(self, adapter_path: str):
        """Load HiRA adapters."""
        print(f"Loading HiRA adapters from {adapter_path}")

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

        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100,
            "lora_rank": self.lora_config["r"],
            "lora_alpha": self.lora_config["lora_alpha"],
            "target_modules": self.lora_config["target_modules"],
            "method": "high_rank_lora",
            "note": "Simplified HiRA using high-rank LoRA approximation",
        }

    def merge_and_unload(self):
        """Merge LoRA weights with base model and unload adapters."""
        if self.model is None:
            raise ValueError("Model not initialized")

        print("Merging HiRA weights with base model...")
        self.model = self.model.merge_and_unload()
        return self.model


def create_hira_model(
    model_name: str,
    task_type: str,
    num_labels: Optional[int] = None,
    lora_config: Optional[Dict[str, Any]] = None,
):
    """
    Factory function to create a HiRA model.

    Args:
        model_name: Base model name
        task_type: Task type (classification or generation)
        num_labels: Number of labels for classification
        lora_config: LoRA configuration dictionary (should specify high rank)

    Returns:
        HiRA-adapted model (using high-rank LoRA approximation)
    """
    hira_model = HiRAModel(
        model_name=model_name,
        task_type=task_type,
        num_labels=num_labels,
        lora_config=lora_config,
    )

    return hira_model.get_model()
