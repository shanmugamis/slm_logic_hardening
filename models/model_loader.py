"""Shared model loading utility for all scripts and notebooks.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

_REGISTRY_PATH = _ROOT / "configs" / "models.yml"


def load_registry() -> dict:
    with open(_REGISTRY_PATH) as f:
        return yaml.safe_load(f)["models"]


def get_model_config(model_key: str) -> dict:
    registry = load_registry()
    if model_key not in registry:
        valid = list(registry.keys())
        raise ValueError(
            f"Unknown model_key '{model_key}'. Valid options: {valid}"
        )
    return registry[model_key]


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _str_to_dtype(dtype_str: str):
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from {list(mapping)}")
    return mapping[dtype_str]


def load_model(
    model_key: str,
    mode: str = "baseline",
    adapter_path: str | None = None,
    dtype_override: str | None = None,
):
    """Load a model and tokenizer ready for inference or saliency analysis.

    Args:
        model_key:      Key from configs/models.yml (e.g. 'phi35', 'llama8b').
        mode:           'baseline'  — load base model only.
                        'lora'      — load base model + LoRA adapter.
                        'qlora'     — load base model in 4-bit + LoRA adapter.
        adapter_path:   Path to saved PEFT adapter directory (required for lora/qlora).
        dtype_override: Override the dtype from models.yml (e.g. 'float32' for Captum).

    Returns:
        (model, tokenizer, device) — all on the same device, model in eval mode.
    """
    cfg = get_model_config(model_key)
    model_id = cfg["model_id"]
    device = get_device()

    dtype_str = dtype_override if dtype_override else cfg["torch_dtype"]
    torch_dtype = _str_to_dtype(dtype_str)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if mode == "qlora":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "QLoRA requires bitsandbytes. Install with: pip install bitsandbytes"
            ) from exc

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
        model.to(device)

    if mode in ("lora", "qlora"):
        if adapter_path is None:
            raise ValueError(
                f"mode='{mode}' requires adapter_path to be specified."
            )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer, device


def format_prompt(prompt: str, model_key: str, tokenizer) -> str:
    """Apply chat template based on models.yml use_chat_template flag.

    Centralizes model-specific prompt formatting so scripts don't need
    to know which models require a chat template and which don't.
    """
    cfg = get_model_config(model_key)
    if cfg.get("use_chat_template", False):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt
