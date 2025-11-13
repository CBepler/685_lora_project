"""
Data loading and preprocessing utilities.
"""

from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
from typing import Optional, Dict, Any


class DatasetLoader:
    """Unified dataset loader for SST-2, IMDB, and WikiText-2."""

    def __init__(self, dataset_name: str, config: Dict[str, Any], tokenizer_name: str = "distilbert-base-uncased"):
        """
        Initialize dataset loader.

        Args:
            dataset_name: Name of the dataset (sst2, imdb, wikitext2)
            config: Dataset configuration from config.yaml
            tokenizer_name: Name of the tokenizer to use
        """
        self.dataset_name = dataset_name
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset = None

    def load_data(self, data_dir: Optional[str] = None):
        """Load dataset from disk or HuggingFace."""
        if data_dir:
            # Load from disk
            dataset_path = Path(data_dir) / "raw" / self.dataset_name
            if dataset_path.exists():
                print(f"Loading {self.dataset_name} from {dataset_path}")
                self.dataset = load_from_disk(str(dataset_path))
            else:
                print(f"Local dataset not found, downloading {self.dataset_name}...")
                self._download_dataset()
        else:
            self._download_dataset()

        return self.dataset

    def _download_dataset(self):
        """Download dataset from HuggingFace."""
        if self.dataset_name == "wikitext2":
            self.dataset = load_dataset(
                self.config["name"],
                self.config["config"]
            )
        else:
            self.dataset = load_dataset(self.config["name"])

    def preprocess_classification(self, examples):
        """Preprocess examples for classification tasks."""
        text_column = self.config["text_column"]
        texts = examples[text_column]

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors=None,
        )

        # Add labels
        if self.config["label_column"] in examples:
            tokenized["labels"] = examples[self.config["label_column"]]

        return tokenized

    def preprocess_generation(self, examples):
        """Preprocess examples for generation tasks."""
        text_column = self.config["text_column"]
        texts = examples[text_column]

        # Filter empty texts
        texts = [text if text and len(text) > 0 else " " for text in texts]

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors=None,
        )

        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def prepare_datasets(self, data_dir: Optional[str] = None):
        """Load and preprocess datasets."""
        # Load raw data
        self.load_data(data_dir)

        # Preprocess based on task type
        if self.config["task_type"] == "classification":
            preprocessor = self.preprocess_classification
        elif self.config["task_type"] == "generation":
            preprocessor = self.preprocess_generation
        else:
            raise ValueError(f"Unknown task type: {self.config['task_type']}")

        # Apply preprocessing
        processed_dataset = self.dataset.map(
            preprocessor,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
            desc=f"Preprocessing {self.dataset_name}",
        )

        # Set format to torch tensors
        processed_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        return processed_dataset

    def get_dataloaders(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        data_dir: Optional[str] = None
    ) -> Dict[str, DataLoader]:
        """
        Get dataloaders for training and evaluation.

        Args:
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            data_dir: Optional path to local data directory

        Returns:
            Dictionary with train, validation, and/or test dataloaders
        """
        # Prepare datasets
        dataset = self.prepare_datasets(data_dir)

        dataloaders = {}

        # Training dataloader
        if "train" in dataset:
            dataloaders["train"] = DataLoader(
                dataset["train"],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )

        # Validation dataloader
        if "validation" in dataset:
            dataloaders["validation"] = DataLoader(
                dataset["validation"],
                batch_size=batch_size * 2,  # Larger batch for eval
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

        # Test dataloader
        if "test" in dataset:
            dataloaders["test"] = DataLoader(
                dataset["test"],
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

        return dataloaders


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get basic information about a dataset."""
    info = {
        "sst2": {
            "description": "Stanford Sentiment Treebank - Binary sentiment classification",
            "task": "sentiment_classification",
            "num_classes": 2,
            "metrics": ["accuracy", "f1"],
        },
        "imdb": {
            "description": "IMDB Movie Reviews - Binary sentiment classification",
            "task": "sentiment_classification",
            "num_classes": 2,
            "metrics": ["accuracy", "f1"],
        },
        "wikitext2": {
            "description": "WikiText-2 - Language modeling dataset",
            "task": "text_generation",
            "num_classes": None,
            "metrics": ["perplexity"],
        },
    }

    return info.get(dataset_name, {})
