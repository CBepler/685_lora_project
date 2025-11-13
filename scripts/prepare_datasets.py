#!/usr/bin/env python3
"""
Script to download and prepare datasets for LoRA experiments.
Downloads SST-2, IMDB, and WikiText-2 datasets from HuggingFace.
"""

import os
from datasets import load_dataset
from pathlib import Path

# Define data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_sst2():
    """Download Stanford Sentiment Treebank (SST-2) dataset."""
    print("Downloading SST-2 dataset...")
    dataset = load_dataset("stanfordnlp/sst2")

    # Save to disk
    dataset.save_to_disk(DATA_DIR / "sst2")

    print(f"✓ SST-2 downloaded: {len(dataset['train'])} train, {len(dataset['validation'])} validation samples")
    return dataset

def download_imdb():
    """Download IMDB Reviews dataset."""
    print("\nDownloading IMDB dataset...")
    dataset = load_dataset("stanfordnlp/imdb")

    # Save to disk
    dataset.save_to_disk(DATA_DIR / "imdb")

    print(f"✓ IMDB downloaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
    return dataset

def download_wikitext():
    """Download WikiText-2 dataset."""
    print("\nDownloading WikiText-2 dataset...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    # Save to disk
    dataset.save_to_disk(DATA_DIR / "wikitext2")

    print(f"✓ WikiText-2 downloaded: {len(dataset['train'])} train, {len(dataset['validation'])} validation, {len(dataset['test'])} test samples")
    return dataset

def print_dataset_stats():
    """Print statistics about downloaded datasets."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    # SST-2
    sst2 = load_dataset("stanfordnlp/sst2")
    print(f"\nSST-2 (Sentiment Classification):")
    print(f"  Train samples: {len(sst2['train'])}")
    print(f"  Validation samples: {len(sst2['validation'])}")
    print(f"  Features: {sst2['train'].features}")
    print(f"  Example: {sst2['train'][0]}")

    # IMDB
    imdb = load_dataset("stanfordnlp/imdb")
    print(f"\nIMDB (Sentiment Analysis):")
    print(f"  Train samples: {len(imdb['train'])}")
    print(f"  Test samples: {len(imdb['test'])}")
    print(f"  Features: {imdb['train'].features}")
    print(f"  Example text length: {len(imdb['train'][0]['text'])} chars")

    # WikiText-2
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    print(f"\nWikiText-2 (Text Generation):")
    print(f"  Train samples: {len(wikitext['train'])}")
    print(f"  Validation samples: {len(wikitext['validation'])}")
    print(f"  Test samples: {len(wikitext['test'])}")
    print(f"  Features: {wikitext['train'].features}")

    print("\n" + "="*60)
    print("All datasets downloaded and ready!")
    print("="*60)

if __name__ == "__main__":
    print("Starting dataset download and preparation...\n")

    # Download all datasets
    download_sst2()
    download_imdb()
    download_wikitext()

    # Print statistics
    print_dataset_stats()

    print(f"\nDatasets saved to: {DATA_DIR}")
