#!/usr/bin/env python3
"""
Test if DistilBERT actually supports 4-bit quantization.
"""

import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

print("Testing DistilBERT with 4-bit quantization...")
print("=" * 60)

try:
    # Create quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("\n1. Attempting to load DistilBERT with 4-bit quantization...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        quantization_config=bnb_config,
        device_map={"": 0},
    )

    print("✅ SUCCESS! DistilBERT loaded with 4-bit quantization")
    print(f"   Model type: {type(model)}")
    print(f"   Device: {next(model.parameters()).device}")
    print(f"   Dtype: {next(model.parameters()).dtype}")

    # Test forward pass
    print("\n2. Testing forward pass...")
    dummy_input = {
        "input_ids": torch.randint(0, 1000, (2, 128)).to("cuda"),
        "attention_mask": torch.ones(2, 128).to("cuda"),
    }

    with torch.no_grad():
        output = model(**dummy_input)

    print("✅ Forward pass successful!")
    print(f"   Output shape: {output.logits.shape}")

    print("\n" + "=" * 60)
    print("CONCLUSION: DistilBERT DOES support 4-bit quantization!")
    print("The model switch to BERT-base may have been unnecessary.")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    print("\nDistilBERT does NOT support 4-bit quantization")
    print("=" * 60)
