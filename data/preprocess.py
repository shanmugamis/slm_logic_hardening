from __future__ import annotations
from functools import partial
from transformers import AutoTokenizer

from typing import Any

LABEL_TO_ID = {
    "true": 0,
    "false": 1,
    "uncertain": 2,
}

LABEL_TO_TARGET = {
    "true": "True",
    "false": "False",
    "uncertain": "Uncertain",
}



def get_label_id(label: str) -> int:
    normalized_label = str(label).strip().lower()
    return LABEL_TO_ID[normalized_label]


def format_folio(example: dict[str, Any], cot: bool = False) -> dict[str, Any]:
    hypothesis = example.get("hypothesis", example.get("conclusion", ""))

    cot_instruction = "Let's think step by step.\n\n" if cot else ""

    prompt = (
        "You are a Logical Reasoning System.\n\n"
        f"Premises:\n{example['premises']}\n\n"
        f"Hypothesis:\n{hypothesis}\n\n"
        "Question:\n"
        "Is the hypothesis True, False, or Uncertain?\n\n"
        f"{cot_instruction}"
        "Answer:\n"
    )

    normalized_label = str(example["label"]).strip().lower()
    target_text = LABEL_TO_TARGET[normalized_label]

    return {
        "prompt": prompt,
        "label": normalized_label,
        "label_id": LABEL_TO_ID[normalized_label],
        "target_text": target_text,
        "full_text": prompt + target_text,
        "no_of_premises": get_no_of_premises(example['premises'])
    }

def format_prompt_for_llama(prompt: str, tokenizer) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def preprocess_folio(dataset: Any, cot: bool = False) -> Any:
    fn = partial(format_folio, cot=cot)
    return dataset.map(fn, load_from_cache_file=False)


def format_proofwriter(example: dict[str, Any]) -> dict[str, Any]:
    """Format a tasksource/proofwriter example into the same schema as FOLIO.

    Fields: theory (premises), question (hypothesis), answer (True/False only — no Uncertain).
    Note for paper: ProofWriter has no Uncertain examples. Training on it may reduce
    the model's tendency to predict Uncertain on FOLIO — worth analyzing in results.
    """
    premises = example.get("theory", "")
    hypothesis = example.get("question", "")

    prompt = (
        "You are a Logical Reasoning System.\n\n"
        f"Premises:\n{premises}\n\n"
        f"Hypothesis:\n{hypothesis}\n\n"
        "Question:\n"
        "Is the hypothesis True, False, or Uncertain?\n\n"
        "Answer:\n"
    )

    normalized_label = str(example.get("answer", "")).strip().lower()
    target_text = LABEL_TO_TARGET[normalized_label]

    return {
        "prompt": prompt,
        "label": normalized_label,
        "label_id": LABEL_TO_ID[normalized_label],
        "target_text": target_text,
        "full_text": prompt + target_text,
        "no_of_premises": get_no_of_premises(premises),
    }


def preprocess_proofwriter(dataset: Any) -> Any:
    # Filter to only valid labels before mapping — map() cannot handle None returns
    valid_labels = set(LABEL_TO_TARGET.keys())
    dataset = dataset.filter(
        lambda x: str(x.get("answer", "")).strip().lower() in valid_labels
    )
    return dataset.map(format_proofwriter, load_from_cache_file=False)

def format_ruletaker(example: dict[str, Any]) -> dict[str, Any]:
    """Format an allenai/ruletaker example into the same schema as FOLIO.

    Fields: context (premises), question (hypothesis), answer (bool True/False — no Uncertain).
    """
    premises = example.get("context", "")
    hypothesis = example.get("question", "")

    prompt = (
        "You are a Logical Reasoning System.\n\n"
        f"Premises:\n{premises}\n\n"
        f"Hypothesis:\n{hypothesis}\n\n"
        "Question:\n"
        "Is the hypothesis True, False, or Uncertain?\n\n"
        "Answer:\n"
    )

    raw_answer = example.get("answer", False)
    normalized_label = "true" if raw_answer else "false"
    target_text = LABEL_TO_TARGET[normalized_label]

    return {
        "prompt": prompt,
        "label": normalized_label,
        "label_id": LABEL_TO_ID[normalized_label],
        "target_text": target_text,
        "full_text": prompt + target_text,
        "no_of_premises": get_no_of_premises(premises),
    }


def preprocess_ruletaker(dataset: Any) -> Any:
    return dataset.map(format_ruletaker, load_from_cache_file=False)


DATASET_PREPROCESSORS: dict = {
    "folio": preprocess_folio,
    "proofwriter": preprocess_proofwriter,
    "ruletaker": preprocess_ruletaker,
}

def get_tokenizer(model_id: str = "microsoft/Phi-3.5-mini-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize(example: dict[str, Any], tokenizer):
    return tokenizer( example["full_text"], truncation=True, padding="max_length", max_length=384)

def get_no_of_premises(premises: str) -> int:
    return len([s for s in premises.split('.') if s.strip()])

    
