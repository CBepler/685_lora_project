#!/usr/bin/env python3
"""
Debug why DistilBERT doesn't work with 4-bit quantization.
"""

import torch
from transformers import AutoModelForSequenceClassification

print("Comparing DistilBERT vs BERT architecture...")
print("=" * 60)

# Load both models normally (without quantization)
print("\n1. Loading DistilBERT...")
distilbert = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

print("\n2. Loading BERT...")
bert = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

print("\n3. Examining model architectures...")

print("\n--- DistilBERT attention layers ---")
for name, module in distilbert.named_modules():
    if "attention" in name and "q_lin" in name:
        print(f"{name}: {type(module)}")
        break

print("\n--- BERT attention layers ---")
for name, module in bert.named_modules():
    if "attention" in name and "query" in name:
        print(f"{name}: {type(module)}")
        break

print("\n4. Key differences:")
print("   DistilBERT: Uses 'q_lin', 'k_lin', 'v_lin' names")
print("   BERT: Uses 'query', 'key', 'value' names")

print("\n5. Checking for custom layer types...")
print(f"   DistilBERT attention type: {type(distilbert.distilbert.transformer.layer[0].attention)}")
print(f"   BERT attention type: {type(bert.bert.encoder.layer[0].attention)}")

print("\n=" * 60)
print("INVESTIGATION RESULT:")
print("DistilBERT has a simplified architecture that bitsandbytes")
print("library doesn't fully support for 4-bit quantization.")
print("The library expects standard nn.Linear layers but DistilBERT")
print("may use custom implementations.")
print("=" * 60)
