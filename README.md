# Logic-Hardening: Bridging the Reasoning Gap in Small Language Models

## 📌 Overview

Small Language Models (SLMs) are efficient and deployable on edge devices, but they often fail at **multi-step logical reasoning**. Instead of following logical constraints in prompts, they tend to rely on memorized patterns.

This project investigates whether **Parameter-Efficient Fine-Tuning (PEFT)**—specifically **LoRA (Low-Rank Adaptation)**—can improve logical reasoning in SLMs.

We focus on improving the deductive reasoning ability of a 3.8B parameter model and compare its performance against larger models.

---

## 🎯 Objectives

* Benchmark baseline reasoning performance of small models
* Identify failure modes in logical reasoning tasks
* Apply LoRA fine-tuning to improve reasoning
* Evaluate improvements using:

  * Exact Match (EM) Accuracy
  * Deductive Consistency
* Analyze attention behavior using interpretability tools

---

## 🧠 Models

* Phi-3.5 (3.8B) — primary model for fine-tuning
* Llama-3 8B — baseline comparison

---

## 📊 Datasets

* FOLIO: First-order logic reasoning dataset (primary)
* LogiQA 2.0: Secondary validation dataset

---

## ⚙️ Methodology

### 1. Baseline Evaluation

* Evaluate pre-trained models on logical reasoning tasks
* Measure failure cases (incorrect logical inference)

### 2. LoRA Fine-Tuning

* Inject low-rank adapters into attention layers
* Train only a small subset of parameters
* Preserve base model knowledge while adapting reasoning

### 3. Evaluation Metrics

* **Exact Match Accuracy (EM)**
  Percentage of predictions matching ground truth labels

* **Deductive Consistency**
  Performance as reasoning depth increases

### 4. Interpretability

* Use saliency maps to analyze attention shifts
* Compare pre- and post-fine-tuning behavior

---

## 📁 Repository Structure

```
configs/    # Training and LoRA configurations
data/       # Dataset loading and preprocessing
models/     # Model loading and LoRA integration
scripts/    # Training, evaluation, inference scripts
tests/      # Unit tests
outputs/    # Logs, checkpoints, and results
notebooks/  # Experimentation and visualization
```

---

## 🚀 Getting Started

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train with LoRA

```
python scripts/train.py
```

### 3. Evaluate model

```
python scripts/eval.py
```

---

## 📈 Experiments

Key experiments include:

* Baseline vs LoRA performance comparison
* Effect of LoRA rank (r) on reasoning accuracy
* Performance vs reasoning depth
* Attention visualization before vs after tuning

---

## 🧪 Results (Planned)

* Improved EM accuracy on FOLIO
* Increased robustness on multi-step reasoning
* Better alignment with logical constraints in prompts

---

## 🔍 Key Insight

We hypothesize that:

> LoRA enables small models to **prioritize logical structure over memorized knowledge**, improving deductive reasoning without large-scale retraining.

---

## 📚 References

* FOLIO: Natural Language Reasoning with First-Order Logic
* LoRA: Low-Rank Adaptation of Large Language Models
* Orca 2: Teaching Small Language Models How to Reason
* Chain-of-Thought Prompting

---

## 🧑‍💻 Author

Shanmugam Subramanian

---

## 📌 Future Work

* Extend to other reasoning benchmarks
* Compare with full fine-tuning
* Explore chain-of-thought supervision
* Evaluate on on-device deployment constraints
