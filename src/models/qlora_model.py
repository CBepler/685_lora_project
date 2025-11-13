"""
QLoRA model implementation with 4-bit quantization.
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)
from typing import Optional, Dict, Any


class QLoRAModel:
    """
    QLoRA model with 4-bit quantization using bitsandbytes.
    Implements NF4 quantization, double quantization, and paged optimizers.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        task_type: str = "classification",
        num_labels: Optional[int] = 2,
        lora_config: Optional[Dict[str, Any]] = None,
        quantization_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize QLoRA model.

        Args:
            model_name: Base model name
            task_type: Task type (classification or generation)
            num_labels: Number of labels for classification
            lora_config: LoRA configuration dictionary
            quantization_config: Quantization configuration dictionary
        """
        self.model_name = model_name
        self.task_type = task_type
        self.num_labels = num_labels
        self.model = None
        self.lora_config = lora_config or self._default_lora_config()
        self.quantization_config = quantization_config or self._default_quantization_config()

    def _default_lora_config(self) -> Dict[str, Any]:
        """Default LoRA configuration."""
        return {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": ["q_lin", "v_lin"],
            "bias": "none",
        }

    def _default_quantization_config(self) -> Dict[str, Any]:
        """Default quantization configuration."""
        return {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",  # NormalFloat 4-bit
            "bnb_4bit_use_double_quant": True,  # Double quantization
            "bnb_4bit_quant_storage": "uint8",
        }

    def get_bnb_config(self) -> BitsAndBytesConfig:
        """Create BitsAndBytes configuration for 4-bit quantization."""
        compute_dtype = getattr(torch, self.quantization_config["bnb_4bit_compute_dtype"])

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.quantization_config["load_in_4bit"],
            bnb_4bit_quant_type=self.quantization_config["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.quantization_config["bnb_4bit_use_double_quant"],
        )

        return bnb_config

    def load_base_model(self):
        """Load the base model with 4-bit quantization."""
        print(f"Loading base model with 4-bit quantization: {self.model_name}")

        bnb_config = self.get_bnb_config()

        if self.task_type == "classification":
            config = AutoConfig.from_pretrained(self.model_name)
            config.num_labels = self.num_labels

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=config,
                quantization_config=bnb_config,
                device_map="auto",  # Automatically handle device placement
            )
        elif self.task_type == "generation":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        print(f"Base model loaded with 4-bit quantization: {self.model.__class__.__name__}")
        return self.model

    def apply_lora(self):
        """Apply LoRA adapters to the quantized model."""
        if self.model is None:
            self.load_base_model()

        # Determine task type for PEFT
        if self.task_type == "classification":
            peft_task_type = TaskType.SEQ_CLS
        elif self.task_type == "generation":
            peft_task_type = TaskType.CAUSAL_LM
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
        print("Applying LoRA adapters to quantized model...")
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        return self.model

    def get_model(self):
        """Get the QLoRA-adapted model."""
        if self.model is None:
            self.load_base_model()
            self.apply_lora()
        return self.model

    def save_model(self, save_path: str):
        """Save the QLoRA adapters (quantized weights are not saved)."""
        if self.model is None:
            raise ValueError("No model to save")

        print(f"Saving QLoRA adapters to {save_path}")
        self.model.save_pretrained(save_path)

        # Save quantization config
        import json
        from pathlib import Path

        config_path = Path(save_path) / "quantization_config.json"
        with open(config_path, "w") as f:
            json.dump(self.quantization_config, f, indent=2)

    def load_model(self, adapter_path: str):
        """Load QLoRA adapters."""
        print(f"Loading QLoRA adapters from {adapter_path}")

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
            "quantization_type": self.quantization_config["bnb_4bit_quant_type"],
            "double_quantization": self.quantization_config["bnb_4bit_use_double_quant"],
            "compute_dtype": self.quantization_config["bnb_4bit_compute_dtype"],
        }

        return info

    def get_memory_footprint(self) -> Dict[str, float]:
        """Get model memory footprint."""
        if self.model is None:
            raise ValueError("Model not initialized")

        # Get model memory footprint
        memory_footprint = self.model.get_memory_footprint() / (1024 ** 2)  # Convert to MB

        memory_info = {
            "model_memory_footprint_mb": memory_footprint,
        }

        # Add GPU memory if available
        if torch.cuda.is_available():
            memory_info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
            memory_info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
            memory_info["gpu_memory_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

        return memory_info


def create_qlora_model(
    model_name: str,
    task_type: str,
    num_labels: Optional[int] = None,
    lora_config: Optional[Dict[str, Any]] = None,
    quantization_config: Optional[Dict[str, Any]] = None,
):
    """
    Factory function to create a QLoRA model.

    Args:
        model_name: Base model name
        task_type: Task type (classification or generation)
        num_labels: Number of labels for classification
        lora_config: LoRA configuration dictionary
        quantization_config: Quantization configuration dictionary

    Returns:
        QLoRA-adapted model
    """
    qlora_model = QLoRAModel(
        model_name=model_name,
        task_type=task_type,
        num_labels=num_labels,
        lora_config=lora_config,
        quantization_config=quantization_config,
    )

    return qlora_model.get_model()
