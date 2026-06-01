"""
Data utilities for PrivHSD v2
==============================
Loading, preprocessing, and privacy-augmented dataset creation
for Jigsaw Toxic Comment Classification and HateXplain datasets.

Supports:
  - Jigsaw Toxic Comment Classification
  - HateXplain
  - Privacy-augmented variants with synthetic pseudo-author labels
  - Multilingual extension point (XLM-RoBERTa tokenization)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from typing import Optional, Dict, List, Tuple, Callable
import logging
from pathlib import Path
import json
import hashlib

logger = logging.getLogger(__name__)


class HateSpeechDataset(Dataset):
    """Unified dataset for hate speech detection with author identity labels.

    Each sample contains:
      - input_ids: (max_seq_len,) tokenized text
      - attention_mask: (max_seq_len,) attention mask
      - hate_labels: (,) binary hate/not-hate label
      - author_labels: (,) optional pseudo-author identity label
    """

    def __init__(
        self,
        texts: List[str],
        hate_labels: List[int],
        author_ids: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 256,
        is_test: bool = False,
    ):
        self.texts = texts
        self.hate_labels = hate_labels
        self.author_ids = author_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        if self.tokenizer is not None:
            encodings = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].squeeze(0)          # (T,)
            attention_mask = encodings["attention_mask"].squeeze(0) # (T,)
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "hate_labels": torch.tensor(self.hate_labels[idx], dtype=torch.long),
        }

        if self.author_ids is not None:
            item["author_labels"] = torch.tensor(
                self.author_ids[idx], dtype=torch.long
            )

        return item


def create_author_labels(
    texts: List[str],
    n_authors: int = 100,
    random_seed: int = 42,
) -> List[int]:
    """Create synthetic pseudo-author labels for adversarial disentanglement.

    Partitions the dataset into pseudo-author groups based on stylistic
    features: text length, punctuation density, capitalization ratio, etc.
    This simulates the scenario where we want the model to be invariant
    to author identity without having ground-truth author labels.

    Args:
        texts: List of input strings.
        n_authors: Number of pseudo-author classes (output range).
        random_seed: Seed for reproducibility.

    Returns:
        List[int] of length len(texts), each entry in [0, n_authors-1].
    """
    rng = np.random.RandomState(random_seed)
    author_ids = []

    for text in texts:
        if not text:
            author_ids.append(rng.randint(0, n_authors))
            continue

        text_str = str(text)
        n_chars = len(text_str)
        n_words = len(text_str.split())
        n_sentences = len(text_str.split(".")) if "." in text_str else 1
        n_excl = text_str.count("!")
        n_question = text_str.count("?")
        n_upper = sum(1 for c in text_str if c.isupper())
        n_punct = sum(1 for c in text_str if c in ".,;:!?\"'()-")

        # Create hash from stylistic features
        feature_str = f"{n_chars}_{n_words}_{n_sentences}_{n_excl}_{n_question}_{n_upper}_{n_punct}"
        hash_val = int(hashlib.md5((feature_str + str(random_seed)).encode()).hexdigest(), 16)
        author_id = abs(hash_val) % n_authors
        author_ids.append(author_id)

    return author_ids


def load_jigsaw_dataset(
    data_dir: str = "data/jigsaw",
    tokenizer=None,
    max_length: int = 256,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> Tuple[HateSpeechDataset, HateSpeechDataset, HateSpeechDataset, int]:
    """Load Jigsaw Toxic Comment Classification dataset.

    Returns train/val/test splits and number of author classes.
    If data files don't exist, generates synthetic data for prototyping.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "train.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        texts = df["comment_text"].tolist()

        # Multi-label toxicity → binary hate label
        toxic_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        available_cols = [c for c in toxic_cols if c in df.columns]
        if available_cols:
            hate_labels = (df[available_cols].sum(axis=1) > 0).astype(int).tolist()
        else:
            raise ValueError("No toxicity columns found in Jigsaw CSV")
    else:
        logger.warning(
            f"Jigsaw dataset not found at {csv_path}. "
            f"Generating synthetic data for prototyping."
        )
        rng = np.random.RandomState(random_seed)
        n_samples = sample_size or 5000
        texts = [
            f"This is sample text {i} for prototyping the privacy-preserving "
            f"hate speech detection model. {'Hateful content here.' if rng.random() > 0.8 else ''}"
            for i in range(n_samples)
        ]
        hate_labels = [1 if "Hateful" in t else 0 for t in texts]

    # Create pseudo-author labels
    n_authors = min(100, len(texts) // 20)
    author_labels = create_author_labels(texts, n_authors=n_authors, random_seed=random_seed)

    # Split into train/val/test
    n = len(texts)
    indices = np.random.RandomState(random_seed).permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Apply sample size limit
    if sample_size and sample_size < n_train:
        train_idx = train_idx[:sample_size]

    datasets = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        datasets[name] = HateSpeechDataset(
            texts=[texts[i] for i in idx],
            hate_labels=[hate_labels[i] for i in idx],
            author_ids=[author_labels[i] for i in idx],
            tokenizer=tokenizer,
            max_length=max_length,
            is_test=(name == "test"),
        )

    logger.info(
        f"Jigsaw dataset loaded: train={len(datasets['train'])}, "
        f"val={len(datasets['val'])}, test={len(datasets['test'])}, "
        f"authors={n_authors}"
    )

    return datasets["train"], datasets["val"], datasets["test"], n_authors


def load_hatexplain_dataset(
    data_dir: str = "data/hatexplain",
    tokenizer=None,
    max_length: int = 256,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> Tuple[HateSpeechDataset, HateSpeechDataset, HateSpeechDataset, int]:
    """Load HateXplain dataset.

    Returns train/val/test splits and number of author classes.
    Generates synthetic data if data files don't exist.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    json_path = data_dir / "hatexplain.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)

        texts = []
        hate_labels = []

        for post_id, post_data in data.items():
            tokens = post_data.get("tokens", [])
            texts.append(" ".join(tokens))

            # Majority vote for label
            annotations = post_data.get("annotators", [])
            label_votes = [a.get("label", 0) for a in annotations]
            # Map: 0=hate, 1=offensive, 2=normal → binary hate/not
            if label_votes:
                majority = max(set(label_votes), key=label_votes.count)
                hate_labels.append(1 if majority < 2 else 0)
            else:
                hate_labels.append(0)
    else:
        logger.warning(
            f"HateXplain dataset not found at {json_path}. "
            f"Generating synthetic data for prototyping."
        )
        rng = np.random.RandomState(random_seed)
        n_samples = sample_size or 3000
        texts = [
            f"HateXplain synthetic sample {i}. "
            f"{'This contains hate speech.' if rng.random() > 0.75 else 'This is normal text.'}"
            for i in range(n_samples)
        ]
        hate_labels = [1 if "hate speech" in t else 0 for t in texts]

    # Create author labels
    n_authors = min(50, len(texts) // 20)
    author_labels = create_author_labels(texts, n_authors=n_authors, random_seed=random_seed)

    # Split
    n = len(texts)
    indices = np.random.RandomState(random_seed).permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    if sample_size and sample_size < n_train:
        train_idx = train_idx[:sample_size]

    datasets = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        datasets[name] = HateSpeechDataset(
            texts=[texts[i] for i in idx],
            hate_labels=[hate_labels[i] for i in idx],
            author_ids=[author_labels[i] for i in idx],
            tokenizer=tokenizer,
            max_length=max_length,
            is_test=(name == "test"),
        )

    logger.info(
        f"HateXplain dataset loaded: train={len(datasets['train'])}, "
        f"val={len(datasets['val'])}, test={len(datasets['test'])}, "
        f"authors={n_authors}"
    )

    return datasets["train"], datasets["val"], datasets["test"], n_authors


def create_privacy_augmented_variant(
    dataset: HateSpeechDataset,
    epsilon_level: str = "medium",
    random_seed: int = 42,
) -> HateSpeechDataset:
    """Create a privacy-augmented variant of the dataset.

    Simulates data pre-processed with:
      - Label noise injection (DP in data curation)
      - Word dropout / stylized perturbation
      - (Future) Entity masking for PII removal

    Levels:
      - 'low':   label_flip=0.02, word_drop=0.05
      - 'medium': label_flip=0.05, word_drop=0.10
      - 'high':  label_flip=0.10, word_drop=0.20
    """
    rng = np.random.RandomState(random_seed)

    noise_params = {
        "low": {"label_flip_prob": 0.02, "word_drop_prob": 0.05},
        "medium": {"label_flip_prob": 0.05, "word_drop_prob": 0.10},
        "high": {"label_flip_prob": 0.10, "word_drop_prob": 0.20},
    }
    params = noise_params.get(epsilon_level, noise_params["medium"])

    new_texts = []
    new_labels = []

    for text, label in zip(dataset.texts, dataset.hate_labels):
        # Label flipping noise
        if rng.random() < params["label_flip_prob"]:
            label = 1 - label

        # Word dropout noise (stylized perturbation)
        words = str(text).split()
        if len(words) > 5:
            keep_mask = rng.random(len(words)) > params["word_drop_prob"]
            words = [w for w, keep in zip(words, keep_mask) if keep]
        new_texts.append(" ".join(words) if words else text)
        new_labels.append(label)

    return HateSpeechDataset(
        texts=new_texts,
        hate_labels=new_labels,
        author_ids=dataset.author_ids,
        tokenizer=dataset.tokenizer,
        max_length=dataset.max_length,
        is_test=dataset.is_test,
    )


def get_dataloaders(
    train_dataset: HateSpeechDataset,
    val_dataset: HateSpeechDataset,
    test_dataset: HateSpeechDataset,
    batch_size: int = 16,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/val/test splits.

    Note: For DP-SGD with Opacus, Poisson sampling is enabled via
    Opacus's privacy engine, which wraps the DataLoader. Standard
    DataLoader shuffling is used as a fallback.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
