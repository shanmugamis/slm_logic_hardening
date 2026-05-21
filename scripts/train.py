"""LoRA fine-tuning script — driven entirely by a YAML config file.

Usage:
    python -m scripts.train --config configs/lora/phi35.yml
    python -m scripts.train --config configs/lora/phi35.yml --experiment_name phi35_lora_r16_ablation
    python -m scripts.train --config configs/lora/phi35.yml --seed 123
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, set_seed

import numpy as np
from data.load_data import load_datasets_for_training
from data.preprocess import DATASET_PREPROCESSORS
from models.model_loader import load_model, get_model_config, get_device
from models.performance import PerformanceMonitor

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_tokenize_fn(tokenizer, max_length: int):
    """Returns a tokenize function that masks prompt tokens with -100.

    This ensures loss is only computed on the answer tokens (True/False/Uncertain),
    not on the premise/hypothesis prompt which the model already knows how to predict.
    """
    def tokenize_fn(example):
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            example["full_text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        labels = full_ids["input_ids"].copy()
        # Mask all prompt positions — PyTorch ignores -100 in loss calculation
        for i in range(len(prompt_ids)):
            labels[i] = -100
        full_ids["labels"] = labels
        return full_ids

    return tokenize_fn


def train(config_path: str, experiment_name_override: str | None = None, seed: int = 42):
    cfg = load_config(config_path)

    set_seed(seed)

    model_key = cfg["model_key"]
    experiment_name = experiment_name_override or cfg["experiment_name"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    # Output directory: outputs/{model_key}/{experiment_name}/seed_{seed}/
    output_dir = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / model_key
        / experiment_name
        / f"seed_{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model — use qlora mode if quantization is enabled in config
    train_mode = "qlora" if cfg.get("quantization", False) else "baseline"
    model, tokenizer, device = load_model(model_key, mode=train_mode)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare datasets — loaded from config, combined if multiple
    dataset_names = cfg.get("datasets", ["folio"])
    sample_sizes = cfg.get("dataset_sample_sizes", {})
    raw_datasets = load_datasets_for_training(dataset_names, sample_sizes=sample_sizes, seed=seed)

    tokenize_fn = build_tokenize_fn(tokenizer, max_length=train_cfg["max_length"])
    columns_to_keep = ["input_ids", "attention_mask", "labels"]

    all_train = []
    folio_val = None  # always validate on FOLIO only — it's the target benchmark

    for name, raw in raw_datasets.items():
        preprocessor = DATASET_PREPROCESSORS[name]
        processed = preprocessor(raw)

        if name == "folio":
            # Hold out 15% of the FOLIO training set (~151 examples) as the
            # in-training validation set used for early stopping. The official
            # FOLIO validation split (203 examples) is reserved exclusively for
            # final evaluation in eval.py and must not be seen during training.
            folio_splits = processed["train"].train_test_split(test_size=0.15, seed=seed)
            t = folio_splits["train"]
            v = folio_splits["test"].map(tokenize_fn, load_from_cache_file=False)
            v = v.remove_columns([c for c in v.column_names if c not in columns_to_keep])
            folio_val = v
        else:
            t = processed["train"]

        t = t.map(tokenize_fn, load_from_cache_file=False)
        t = t.remove_columns([c for c in t.column_names if c not in columns_to_keep])
        all_train.append(t)

    if len(all_train) == 1:
        train_dataset = all_train[0]
    else:
        from datasets import concatenate_datasets
        train_dataset = concatenate_datasets(all_train)

    val_dataset = folio_val

    print(f"Training on: {dataset_names} | Train: {len(train_dataset)} | Val: {len(val_dataset) if val_dataset else 0} (FOLIO only)")

    # Training arguments — all values from YAML
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        eval_accumulation_steps=train_cfg.get("eval_accumulation_steps"),
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        max_grad_norm=train_cfg["max_grad_norm"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        report_to=train_cfg["report_to"],
        seed=seed,
    )

    def compute_metrics(eval_pred):
        """Report token-level accuracy on the answer tokens for eval_accuracy metric."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        mask = labels != -100
        correct = (predictions[mask] == labels[mask]).sum()
        total = mask.sum()
        return {"eval_accuracy": float(correct / total) if total > 0 else 0.0}

    # Build a vocab-sized weight tensor if class_weights are configured (P1 rebalancing).
    # Critical: we must use the IN-CONTEXT token IDs that actually appear in training
    # sequences (after "Answer:\n"), not the bare-word token IDs. Phi-3.5's BPE merges
    # "True" / "False" / "Uncertain" differently depending on preceding context — e.g.
    # bare "True" -> [5852] but "\nTrue" -> [5574]. Also, multi-token labels like
    # "Uncertain" -> [2525, 14082] need ALL subword positions weighted, not just the first.
    trainer_cls = Trainer
    class_weights_cfg = train_cfg.get("class_weights")
    if class_weights_cfg:
        prompt_suffix = "Answer:\n"
        prompt_suffix_ids = tokenizer(prompt_suffix, add_special_tokens=False)["input_ids"]
        label_token_ids: dict[str, list[int]] = {}
        for label in ("True", "False", "Uncertain"):
            full_ids = tokenizer(prompt_suffix + label, add_special_tokens=False)["input_ids"]
            label_token_ids[label] = full_ids[len(prompt_suffix_ids):]

        weight_tensor = torch.ones(model.config.vocab_size, dtype=torch.float32)
        for label, token_ids in label_token_ids.items():
            w = float(class_weights_cfg.get(label, 1.0))
            for tid in token_ids:
                weight_tensor[tid] = w
        weight_tensor = weight_tensor.to(model.device if hasattr(model, "device") else "cpu")
        print(f"Label token IDs (in-context): {label_token_ids}")
        print(f"Weights applied: " + ", ".join(
            f"{lbl}={class_weights_cfg.get(lbl, 1.0)} -> tokens {tids}"
            for lbl, tids in label_token_ids.items()
        ))

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=weight_tensor.to(shift_logits.device),
                    ignore_index=-100,
                )
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                return (loss, outputs) if return_outputs else loss

        trainer_cls = WeightedTrainer
        print(f"Using WeightedTrainer with class weights: {class_weights_cfg}")

    trainer_kwargs: dict = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    if val_dataset is not None:
        trainer_kwargs["eval_dataset"] = val_dataset
        trainer_kwargs["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=2)]

    trainer = trainer_cls(**trainer_kwargs)

    monitor = PerformanceMonitor(device)
    monitor.start()

    trainer.train()

    monitor.stop(n_samples=len(train_dataset))
    monitor.save(
        output_dir / "performance.json",
        extra={
            "model_key": model_key,
            "experiment_name": experiment_name,
            "mode": "training",
            "seed": seed,
            "datasets": dataset_names,
            "n_train_examples": len(train_dataset),
            "lora_r": lora_cfg["r"],
            "learning_rate": train_cfg["learning_rate"],
            "epochs": train_cfg["num_train_epochs"],
            "batch_size": train_cfg["per_device_train_batch_size"],
        },
    )

    
    trainer_state_src = output_dir / "trainer_state.json"
    if trainer_state_src.exists():
        import shutil
        shutil.copy(trainer_state_src, output_dir / "training_history.json")
        print(f"Training history saved to {output_dir / 'training_history.json'}")

    # Save final adapter
    adapter_path = output_dir / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nAdapter saved to {adapter_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for SLM logic hardening")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to LoRA config YAML (e.g. configs/lora/phi35.yml)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Override experiment_name from YAML (sets output subdirectory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config, args.experiment_name, seed=args.seed)
