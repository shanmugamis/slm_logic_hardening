"""Compute token-level attribution (saliency) using Captum LayerIntegratedGradients.

NOTE: This script loads the model in float32 (required by Captum for accurate gradients).
      Memory usage is ~2x higher than inference. Run on Colab A100 for best results.

Usage:
    # Baseline saliency
    python -m scripts.saliency --model_id phi35 --mode baseline \
        --output_dir outputs/phi35/phi35_baseline

    # LoRA saliency
    python -m scripts.saliency --model_id phi35 --mode lora \
        --adapter_path outputs/phi35/phi35_lora_r64/final_adapter \
        --output_dir outputs/phi35/phi35_lora_r64

    # Custom sample indices
    python -m scripts.saliency --model_id phi35 --mode baseline \
        --indices 0 5 10 15 20
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
from captum.attr import LayerIntegratedGradients

from data.load_data import load_folio
from data.preprocess import preprocess_folio
from models.model_loader import load_model, format_prompt, get_model_config

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SALIENCY_CONFIG = _ROOT / "configs" / "eval" / "saliency.yml"


def load_saliency_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_embed_layer(model, model_key: str):
    """Retrieve the embedding layer regardless of LoRA wrapping.

    Baseline: Phi3ForCausalLM.model.embed_tokens
    LoRA:     PeftModel.base_model.model.model.embed_tokens

    HuggingFace models have a base_model property even without PeftModel,
    so we use isinstance check to safely detect LoRA wrapping.
    """
    try:
        from peft import PeftModel
        is_peft = isinstance(model, PeftModel)
    except ImportError:
        is_peft = False

    if is_peft:
        base = model.base_model.model  # PeftModel → LoraModel → Phi3ForCausalLM
    else:
        base = model  # Phi3ForCausalLM directly

    return base.model.embed_tokens


def build_forward_func(model, target_token_id: int):
    """Build a forward function for Captum that returns a single logit score.

    Captum needs: embeddings in → one scalar out (the logit for the target label).
    This is different from model.generate() which produces token sequences.
    """
    def forward_func(input_ids):
        outputs = model(input_ids=input_ids)
        # Logits at the last position — where the model predicts the answer
        last_position_logits = outputs.logits[0, -1, :]
        return last_position_logits[target_token_id].unsqueeze(0)

    return forward_func


def compute_attribution(
    model,
    tokenizer,
    device: str,
    prompt: str,
    label: str,
    model_key: str,
    n_steps: int,
) -> list[dict]:
    """Compute token attribution scores for one example.

    Returns a list of {"token": str, "score": float} sorted by position.
    """
    formatted_prompt = format_prompt(prompt, model_key, tokenizer)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

   
    target_token_id = tokenizer.encode(label, add_special_tokens=False)[0]

    forward_func = build_forward_func(model, target_token_id)
    embed_layer = get_embed_layer(model, model_key)

    lig = LayerIntegratedGradients(forward_func, embed_layer)
    attributions = lig.attribute(input_ids, n_steps=n_steps, internal_batch_size=1)

    # Sum across embedding dimension, take absolute value → one score per token
    attr_scores = attributions.sum(dim=-1).squeeze(0).abs()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return [
        {"token": token, "score": round(score.item(), 6)}
        for token, score in zip(tokens, attr_scores)
    ]


def run_saliency(
    model_key: str,
    mode: str,
    adapter_path: str | None,
    sample_indices: list[int],
    n_steps: int,
    output_dir: Path,
    experiment_id: str | None = None,
):
    model, tokenizer, device = load_model(
        model_key, mode=mode, adapter_path=adapter_path, dtype_override="float32"
    )

    dataset = load_folio()
    preprocessed = preprocess_folio(dataset)
    val_set = preprocessed["validation"]

    exp_id = experiment_id or output_dir.name
    saliency_dir = output_dir / "saliency"
    saliency_dir.mkdir(parents=True, exist_ok=True)

    for idx in sample_indices:
        record = val_set[idx]
        prompt = record["prompt"]
        true_label = record["target_text"]

        print(f"\nExample {idx}: ground_truth = {true_label}")
        print("Computing attributions...")

        token_attributions = compute_attribution(
            model, tokenizer, device, prompt, true_label, model_key, n_steps
        )

        sorted_attrs = sorted(token_attributions, key=lambda x: x["score"], reverse=True)
        print("Top 10 tokens:")
        for ta in sorted_attrs[:10]:
            print(f"  {ta['token']:20s}  {ta['score']:.6f}")

        # Write one file per example: saliency/{index}.json
        example_path = saliency_dir / f"{idx}.json"
        with open(example_path, "w") as f:
            json.dump(
                {
                    "experiment_id": exp_id,
                    "index": idx,
                    "ground_truth": true_label,
                    "no_of_premises": record.get("no_of_premises", 0),
                    "prompt": prompt,
                    "attributions": token_attributions,
                },
                f,
                indent=2,
            )

    print(f"\nSaliency files written to {saliency_dir}/ ({len(sample_indices)} examples)")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Captum saliency maps")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model key from configs/models.yml (e.g. phi35, llama8b)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "lora"],
        help="baseline: base model | lora: base + LoRA adapter",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (required for lora mode)",
    )
    parser.add_argument(
        "--saliency_config",
        type=str,
        default=str(_DEFAULT_SALIENCY_CONFIG),
        help="Path to saliency config YAML (default: configs/eval/saliency.yml)",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Validation set indices to analyze (overrides saliency_config sample_indices)",
    )
    parser.add_argument("--output_dir", type=str, default=None,
        help="Output directory. Defaults to outputs/{model_id}/{experiment_id}")
    parser.add_argument("--experiment_id", type=str, default=None,
        help="Experiment ID — matches the experiment folder name (e.g. phi35_baseline)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    saliency_cfg = load_saliency_config(args.saliency_config)
    sample_indices = args.indices if args.indices else saliency_cfg["sample_indices"]
    n_steps = saliency_cfg["n_steps"]

    exp_id = args.experiment_id or f"{args.model_id}_{args.mode}"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _ROOT / "outputs" / args.model_id / exp_id

    run_saliency(
        model_key=args.model_id,
        mode=args.mode,
        adapter_path=args.adapter_path,
        sample_indices=sample_indices,
        experiment_id=exp_id,
        n_steps=n_steps,
        output_dir=output_dir,
    )
