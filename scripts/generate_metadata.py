"""Auto-generate outputs/metadata.json from experiment configs and result files.

Run after every new experiment completes:
    python -m scripts.generate_metadata

The React dashboard reads this file as its primary data source.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS_DIR = _ROOT / "configs" / "experiments"
_MODELS_CONFIG = _ROOT / "configs" / "models.yml"
_OUTPUTS_DIR = _ROOT / "outputs"


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def scan_experiment_files(model_key: str, exp_id: str) -> dict:
    """Check which output files exist for an experiment."""
    exp_dir = _OUTPUTS_DIR / model_key / exp_id

    results_path = exp_dir / "results.json"
    metrics_path = exp_dir / "metrics.json"
    performance_path = exp_dir / "performance.json"
    trainer_state_path = exp_dir / "trainer_state.json"
    saliency_dir = exp_dir / "saliency"
    adapter_dir = exp_dir / "adapter"

    return {
        "results": str(results_path.relative_to(_OUTPUTS_DIR)) if results_path.exists() else None,
        "metrics": str(metrics_path.relative_to(_OUTPUTS_DIR)) if metrics_path.exists() else None,
        "performance": str(performance_path.relative_to(_OUTPUTS_DIR)) if performance_path.exists() else None,
        "saliency_dir": str(saliency_dir.relative_to(_OUTPUTS_DIR)) if saliency_dir.exists() else None,
        "trainer_state": str(trainer_state_path.relative_to(_OUTPUTS_DIR)) if trainer_state_path.exists() else None,
        "adapter": str(adapter_dir.relative_to(_OUTPUTS_DIR)) if adapter_dir.exists() else None,
    }


def read_summary(model_key: str, exp_id: str) -> dict | None:
    """Read accuracy and counts from results.json."""
    results_path = _OUTPUTS_DIR / model_key / exp_id / "results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        d = json.load(f)
    n_total = d.get("n_total", len(d.get("results", [])))
    n_invalid = d.get("n_invalid", 0)
    return {
        "accuracy": d.get("accuracy"),
        "n_correct": d.get("n_correct"),
        "n_total": n_total,
        "n_invalid": n_invalid,
        "invalid_rate": round(n_invalid / n_total * 100, 3) if n_total else 0,
    }


def read_lora_params(model_key: str, exp_id: str) -> dict | None:
    """Read LoRA config from adapter_config.json if present."""
    adapter_config = _OUTPUTS_DIR / model_key / exp_id / "adapter" / "adapter_config.json"
    if not adapter_config.exists():
        return None
    with open(adapter_config) as f:
        cfg = json.load(f)
    return {
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "target_modules": cfg.get("target_modules"),
        "lora_dropout": cfg.get("lora_dropout"),
    }


def read_trainer_state(model_key: str, exp_id: str) -> dict | None:
    """Read best metric and training summary from trainer_state.json."""
    trainer_state = _OUTPUTS_DIR / model_key / exp_id / "trainer_state.json"
    if not trainer_state.exists():
        return None
    with open(trainer_state) as f:
        state = json.load(f)
    return {
        "best_metric": state.get("best_metric"),
        "best_global_step": state.get("best_model_checkpoint"),
        "epoch": state.get("epoch"),
    }


def build_metadata() -> dict:
    models_cfg = load_yaml(_MODELS_CONFIG)["models"]

    # Build models section with display metadata
    models = {}
    for key, cfg in models_cfg.items():
        models[key] = {
            "model_id": cfg["model_id"],
            "display_name": cfg["display_name"],
            "torch_dtype": cfg["torch_dtype"],
            "max_new_tokens": cfg.get("max_new_tokens"),
            "max_new_tokens_cot": cfg.get("max_new_tokens_cot"),
        }

    # Build experiments section from configs/experiments/*.yml
    experiments = []
    exp_configs = sorted(_EXPERIMENTS_DIR.glob("*.yml"))

    for exp_path in exp_configs:
        exp_cfg = load_yaml(exp_path)
        exp_id = exp_cfg["id"]
        model_key = exp_cfg["model_key"]

        files = scan_experiment_files(model_key, exp_id)
        summary = read_summary(model_key, exp_id)
        lora_params = read_lora_params(model_key, exp_id)
        trainer_info = read_trainer_state(model_key, exp_id)

        experiments.append({
            "id": exp_id,
            "display_name": exp_cfg.get("display_name"),
            "short_name": exp_cfg.get("short_name"),
            "description": exp_cfg.get("description", "").strip(),
            "model_key": model_key,
            "prompting": exp_cfg.get("prompting", "standard"),
            "training_datasets": exp_cfg.get("training_datasets", []),
            "lora_config": lora_params,
            "color": exp_cfg.get("color"),
            "ablation_order": exp_cfg.get("ablation_order", 99),
            "group": exp_cfg.get("group"),
            "tags": exp_cfg.get("tags", []),
            "files": files,
            "capabilities": {
                "has_results": files["results"] is not None,
                "has_metrics": files["metrics"] is not None,
                "has_performance": files["performance"] is not None,
                "has_saliency": files["saliency_dir"] is not None,
                "has_training_curves": trainer_info is not None,
                "has_cot_reasoning": exp_cfg.get("prompting") == "cot",
            },
            "summary": summary,
            "training": trainer_info,
        })

    return {
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": {
            "dataset": "folio",
            "split": "validation",
            "n_examples": 203,
            "labels": ["True", "False", "Uncertain"],
            "depth_field": "no_of_premises",
        },
        "models": models,
        "experiments": experiments,
        "presets": [
            {
                "id": "lora_wins",
                "label": "LoRA Wins",
                "description": "Fine-tuned model correct, baseline wrong",
                "filter": {
                    "type": "comparison",
                    "must_correct": ["phi35_lora_folio_cot"],
                    "must_wrong": ["phi35_baseline"],
                },
            },
            {
                "id": "all_disagree",
                "label": "All Disagree",
                "filter": {"type": "all_disagree"},
            },
            {
                "id": "scale_wins",
                "label": "Scale Wins",
                "description": "Larger model correct, smaller wrong",
                "filter": {
                    "type": "comparison",
                    "must_correct": ["llama8b_baseline"],
                    "must_wrong": ["phi35_baseline", "phi35_lora_folio_cot"],
                },
            },
            {"id": "all_correct", "label": "All Correct", "filter": {"type": "all_correct"}},
            {"id": "all_wrong", "label": "All Wrong", "filter": {"type": "all_wrong"}},
        ],
        "ui": {
            "ablation_order": [
                "phi35_baseline",
                "phi35_baseline_cot",
                "phi35_lora_folio_cot",
                "phi35_lora_folio_pw_cot",
                "phi35_lora_all_cot",
            ],
            "scale_comparison": {
                "small": ["phi35_baseline", "phi35_lora_folio_cot"],
                "large": ["llama8b_baseline", "mistral7b_baseline"],
            },
        },
    }


if __name__ == "__main__":
    metadata = build_metadata()
    output_path = _OUTPUTS_DIR / "metadata.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"metadata.json written to {output_path}")

    completed = sum(1 for e in metadata["experiments"] if e["summary"] is not None)
    print(f"Experiments: {completed}/{len(metadata['experiments'])} have results")
    for exp in metadata["experiments"]:
        status = f"{exp['summary']['accuracy']:.2f}%" if exp["summary"] else "pending"
        caps = exp["capabilities"]
        flags = " ".join(k.replace("has_", "") for k, v in caps.items() if v)
        print(f"  {exp['id']:<35} {status:<10} [{flags}]")
