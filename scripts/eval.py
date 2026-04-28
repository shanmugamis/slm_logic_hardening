"""Unified evaluation script — handles baseline, LoRA, and QLoRA modes.

Usage:
    python -m scripts.eval --model_id phi35 --mode baseline
    python -m scripts.eval --model_id phi35 --mode baseline --cot
    python -m scripts.eval --model_id phi35 --mode lora \
        --adapter_path outputs/phi35/phi35_lora_folio_cot/adapter \
        --training_datasets folio --cot
    python -m scripts.eval --model_id llama8b --mode baseline \
        --experiment_id llama8b_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from tqdm import tqdm

from data.load_data import load_folio
from data.preprocess import preprocess_folio
from models.model_loader import load_model, format_prompt, get_model_config
from models.performance import PerformanceMonitor

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_EVAL_CONFIG = _ROOT / "configs" / "eval" / "default.yml"


def load_eval_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_predicted_label(generated_text: str, valid_labels: list[str]) -> str:
    """Parse model output to extract a canonical label.

    Case-insensitive. Uses last occurrence so CoT conclusions override
    any label mentioned in the reasoning chain.
    Also maps common LLaMA phrasings to Uncertain.
    """
    # Search full text — LoRA models sometimes output the label before repeating
    # the prompt template (e.g. "True\n\nLogical Reasoning System:\nAnswer:\n")
    # so restricting to after "Answer:" would miss it.
    # For CoT models the last occurrence still correctly picks the final conclusion.
    text = generated_text.strip()

    # Strategy: check the first line first — LoRA models output the label
    # directly as the first token/line before repeating the prompt template.
    # Checking first line avoids false matches on "True, False, or Uncertain"
    # in repeated question text.
    first_line = text.split("\n")[0].strip().lower()
    for label in valid_labels:
        if label.lower() in first_line:
            return label

    # Fallback: search full text with phrasing normalization for CoT models
    # (LLaMA generates reasoning then concludes, so answer is near the end)
    lower_part = text.lower()

    phrasing_map = {
        "uncertain": [
            "unknown", "cannot determine", "not enough information",
            "insufficient information", "cannot be determined",
            "it is uncertain",
        ],
    }
    for canonical, phrases in phrasing_map.items():
        for phrase in phrases:
            if phrase in lower_part:
                lower_part = lower_part.replace(phrase, canonical)

    last_pos = {}
    for label in valid_labels:
        pos = lower_part.rfind(label.lower())
        if pos != -1:
            last_pos[label] = pos

    if last_pos:
        return max(last_pos, key=last_pos.get)
    return "invalid"


def extract_reasoning_chain(generated_text: str) -> str:
    """Extract the reasoning steps from a CoT output, before the final answer."""
    if "Answer:" in generated_text:
        return generated_text.split("Answer:")[0].strip()
    return ""


def evaluate(
    model_key: str,
    mode: str,
    adapter_path: str | None,
    eval_config: dict,
    output_dir: Path,
    cot: bool = False,
    experiment_id: str | None = None,
    training_datasets: list[str] | None = None,
):
    """Run inference on the validation set and save results.json."""
    valid_labels = eval_config.get("valid_labels", ["True", "False", "Uncertain"])
    do_sample = eval_config.get("do_sample", False)

    model, tokenizer, device = load_model(model_key, mode=mode, adapter_path=adapter_path)
    model_cfg = get_model_config(model_key)

    if cot:
        max_new_tokens = model_cfg.get("max_new_tokens_cot", model_cfg.get("max_new_tokens", 300))
    else:
        max_new_tokens = model_cfg.get("max_new_tokens", eval_config.get("max_new_tokens", 15))

    exp_id = experiment_id or output_dir.name
    prompting = "cot" if cot else "standard"
    datasets = training_datasets or ([] if mode == "baseline" else ["folio"])

    dataset = load_folio()
    preprocessed = preprocess_folio(dataset, cot=cot)
    val_set = preprocessed["validation"]

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    correct = 0
    output_token_counts = []
    monitor = PerformanceMonitor(device)
    monitor.start()

    for idx, record in enumerate(tqdm(val_set, desc=f"Evaluating [{exp_id}]")):
        prompt = record["prompt"]
        true_label = record["target_text"]

        formatted_prompt = format_prompt(prompt, model_key, tokenizer)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

        monitor.start_example()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
        monitor.end_example()

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        output_token_counts.append(len(generated_ids))
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        predicted_label = extract_predicted_label(generated_text, valid_labels)
        if predicted_label == true_label:
            correct += 1

        results.append({
            "index": idx,
            "story_id": record.get("id", record.get("story_id", str(idx))),
            "prompt": prompt,
            "ground_truth": true_label,
            "predicted": predicted_label,
            "raw_output": generated_text,
            "reasoning_chain": extract_reasoning_chain(generated_text) if cot else "",
            "no_of_premises": record["no_of_premises"],
        })

    avg_output_tokens = round(sum(output_token_counts) / len(output_token_counts), 1) if output_token_counts else 0

    monitor.stop(n_samples=len(val_set))
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor.save(
        output_dir / "performance.json",
        extra={
            "experiment_id": exp_id,
            "model_id": model_cfg["model_id"],
            "model_key": model_key,
            "mode": mode,
            "n_examples": len(val_set),
            "avg_output_tokens": avg_output_tokens,
        },
    )

    accuracy = (correct / len(val_set)) * 100
    n_invalid = sum(1 for r in results if r["predicted"] == "invalid")

    print(f"\nEM Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct} / {len(val_set)}")
    print(f"Invalid predictions: {n_invalid}")

    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "experiment_id": exp_id,
                "model_id": model_cfg["model_id"],
                "model_key": model_key,
                "mode": mode,
                "prompting": prompting,
                "training_datasets": datasets,
                "adapter_path": adapter_path,
                "accuracy": accuracy,
                "n_correct": correct,
                "n_total": len(val_set),
                "n_invalid": n_invalid,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SLM on FOLIO validation set")
    parser.add_argument("--model_id", type=str, required=True,
        help="Model key from configs/models.yml (e.g. phi35, llama8b)")
    parser.add_argument("--mode", type=str, default="baseline",
        choices=["baseline", "lora", "qlora"])
    parser.add_argument("--adapter_path", type=str, default=None,
        help="Path to LoRA adapter directory (required for lora/qlora)")
    parser.add_argument("--eval_config", type=str, default=str(_DEFAULT_EVAL_CONFIG))
    parser.add_argument("--output_dir", type=str, default=None,
        help="Defaults to outputs/{model_id}/{experiment_id}")
    parser.add_argument("--cot", action="store_true", default=False,
        help="Enable zero-shot chain-of-thought prompting")
    parser.add_argument("--experiment_id", type=str, default=None,
        help="Override experiment ID (used as folder name and in JSON metadata)")
    parser.add_argument("--training_datasets", type=str, nargs="*", default=None,
        help="Datasets used to train the adapter (e.g. folio proofwriter)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    eval_config = load_eval_config(args.eval_config)

    # Auto-derive experiment_id if not provided
    if args.experiment_id:
        exp_id = args.experiment_id
    else:
        prompting_suffix = "_cot" if args.cot else ""
        exp_id = f"{args.model_id}_{args.mode}{prompting_suffix}"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _ROOT / "outputs" / args.model_id / exp_id

    evaluate(
        model_key=args.model_id,
        mode=args.mode,
        adapter_path=args.adapter_path,
        eval_config=eval_config,
        output_dir=output_dir,
        cot=args.cot,
        experiment_id=exp_id,
        training_datasets=args.training_datasets,
    )
