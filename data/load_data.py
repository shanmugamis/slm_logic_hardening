"""Utilities for loading reasoning datasets.

Supported datasets (pass name to load_datasets_for_training()):
    folio       — yale-nlp/FOLIO  (first-order logic, 3-class)
    proofwriter — proofwriter  (deductive reasoning, 3-class)

To add a new dataset:
    1. Add a loader function: load_<name>() → HF DatasetDict
    2. Add a preprocessor in data/preprocess.py: preprocess_<name>()
    3. Register both in DATASET_REGISTRY below
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _load_hf(dataset_id: str, **kwargs) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. "
            "Install with `pip install -r requirements.txt`."
        ) from exc
    return load_dataset(dataset_id, **kwargs)


def load_folio() -> Any:
    """Load the raw FOLIO dataset from Hugging Face."""
    return _load_hf("yale-nlp/FOLIO")


def load_proofwriter(sample_size: int = 10000, seed: int = 42) -> Any:
    """Load ProofWriter from tasksource — deductive reasoning, True/False only.

    Fields: theory (premises), question (hypothesis), answer (True/False).
    No 'Uncertain' label — open-world assumption handled differently from FOLIO.

    sample_size: number of training examples to use (full dataset has 585k).
    Default 10k gives ~10x FOLIO size — enough for data scaling study.
    """
    dataset = _load_hf("tasksource/proofwriter")
    if sample_size and len(dataset["train"]) > sample_size:
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(sample_size))
    return dataset


def load_ruletaker() -> Any:
    """Load RuleTaker — rule-based deduction, True/False/Unknown."""
    return _load_hf("allenai/ruletaker", name="depth-5")


def load_prontoqa() -> Any:
    """Load PrOntoQA — first-order logic proofs, True/False labels."""
    return _load_hf("qixuanj/PrOntoQA")

DATASET_LOADERS: dict[str, Any] = {
    "folio": load_folio,
    "proofwriter": load_proofwriter,
    "ruletaker": load_ruletaker,
    "prontoqa": load_prontoqa,
}


def load_datasets_for_training(
    dataset_names: list[str],
    sample_sizes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Load and return raw datasets by name.

    sample_sizes: optional dict mapping dataset name max training examples.
    Only applies to datasets that support it (e.g. proofwriter).
    """
    sample_sizes = sample_sizes or {}
    loaded = {}
    for name in dataset_names:
        if name not in DATASET_LOADERS:
            valid = list(DATASET_LOADERS.keys())
            raise ValueError(f"Unknown dataset '{name}'. Valid options: {valid}")
        print(f"Loading dataset: {name}")
        loader = DATASET_LOADERS[name]
        n = sample_sizes.get(name)
        try:
            loaded[name] = loader(sample_size=n) if n else loader()
        except TypeError:
            loaded[name] = loader()
    return loaded


def load_folio_with_split(validation_size: float = 0.1, seed: int = 42) -> Mapping[str, Any]:
    """
    Load FOLIO and ensure a validation split is available.

    If the source dataset already contains a ``validation`` split, it is returned
    unchanged. Otherwise, the train split is partitioned into train/validation.
    """

    dataset = load_folio()

    if "validation" in dataset:
        return dataset

    if "train" not in dataset:
        raise KeyError("Expected the FOLIO dataset to contain a 'train' split.")

    split_dataset = dataset["train"].train_test_split(
        test_size=validation_size,
        seed=seed,
    )

    output_splits: dict[str, Any] = {
        "train": split_dataset["train"],
        "validation": split_dataset["test"],
    }

    for split_name, split_value in dataset.items():
        if split_name != "train":
            output_splits[split_name] = split_value

    return output_splits
