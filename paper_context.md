# Paper Writing Task: EMNLP 2026 Submission

## Project context

I'm preparing a paper for EMNLP 2026 via ARR May 2026 cycle. Submission deadline is May 25 (anywhere on Earth). I have 5 days. I'm a first-time author extending a Georgia Tech DL course project. Paper will be submitted to EMNLP Findings track, with TMLR or workshop as realistic fallback.

## Paper title (committed, do not change)

**Class-Asymmetric Effects of Label-Space Mismatch in LoRA Fine-Tuning of Small Language Models**

## Authors

(Anonymized for double-blind review. Real authorship handled separately.)

## Core findings (what the paper reports)

### Finding 1 — Class-asymmetric degradation (HEADLINE)

When fine-tuning Phi-3.5-mini (3.8B) with LoRA on FOLIO (target task with 3 classes: True/False/Uncertain) while adding ProofWriter as auxiliary data (which contains only True/False — the Uncertain class is categorically missing), the per-class effects across auxiliary scale are asymmetric:
- The missing class (Uncertain) does NOT suffer — its recall is stable or improves slightly across all auxiliary scales
- A represented class (False) shows monotonic recall degradation as auxiliary scale grows (0.597 at FOLIO-only → 0.484 at PW-20k)
- This contradicts what classical class imbalance theory would predict (which would predict the missing class suffers most)

### Finding 2 — Reweighting redistributes errors (SUPPORTING)

Classical inverse-frequency class reweighting at PW-2k (the standard remedy for class imbalance) does not resolve the asymmetric effect — it redistributes errors:
- Uncertain recall improves dramatically (0.730 → 0.860, +13pp)
- False recall drops further (0.559 → 0.473, -9pp)
- Overall accuracy declines modestly (70.11% → 68.64%, -1.5pp)
- Predicted-Uncertain count over-shoots from ~72 to ~96 against ground-truth count of ~70

### Finding 3 — Modest non-monotonic accuracy effect across auxiliary scale

Across the scaling curve:
- FOLIO-only: 67.65% (3 seeds)
- FOLIO+PW-2k: 70.11% (3 seeds)
- FOLIO+PW-5k: 70.11% (3 seeds)
- FOLIO+PW-10k: 67.99% (3 seeds)
- FOLIO+PW-20k: 66.01% (1 seed — acknowledge as single-seed limitation)

Plateau peak at modest scale (2k-5k), decline at larger scale. Effect sizes are modest but directionally consistent.

## Exact results table (multi-seed, use these numbers throughout the paper)

| Condition | Seeds | Accuracy mean | True R mean | False R mean | Unc R mean |
|---|---|---|---|---|---|
| FOLIO-only | 3 (42,123,7) | 67.65 | 0.745 | 0.597 | 0.676 |
| FOLIO+PW-2k | 3 | 70.11 | 0.796 | 0.559 | 0.763 |
| FOLIO+PW-2k+rebal | 3 | 68.64 | 0.737 | 0.473 | 0.860 |
| FOLIO+PW-5k | 3 | 70.11 | 0.773 | 0.591 | 0.725 |
| FOLIO+PW-10k | 3 | 67.99 | 0.782 | 0.554 | 0.686 |
| FOLIO+PW-20k | 1 | 66.01 | 0.750 | 0.484 | 0.725 |

Full per-seed data is in `outputs/all_results.csv` (or wherever the analyst CSV lives — check the repo).

Always include bootstrapped 95% CIs (1000 resamples) on accuracy numbers and per-class recall in the final tables. If CI computation code doesn't exist yet, write it.

## Setup and methodology

- Model: Phi-3.5-mini-instruct, 3.8B params, bf16
- Method: LoRA, r=32, alpha=64, dropout=0.30, targets={q,k,v,o,gate_up,down}_proj
- Training: LR=1e-5, effective batch 8, max 10 epochs, early stopping on FOLIO val loss, AdamW, cosine schedule, 50 warmup steps, max seq 384
- Datasets: FOLIO (1,001 train / 203 val from tasksource/folio or yale-nlp/FOLIO). ProofWriter OWA variant, filtered to True/False only (Unknown excluded — this filter is what creates the label-space mismatch). PW samples are nested deterministic across scales.
- Reweighting implementation: vocab-sized weight tensor passed to CrossEntropyLoss(weight=...), weights computed from actual combined-dataset class frequencies (sklearn "balanced" formula: weight[c] = n_total / (n_classes * count[c])). Verified single-token tokenization of label words.
- Seeds: 42, 123, 7. Identical hyperparameters across all conditions (this is critical — earlier version had hyperparameter tuning at 10k which was a confound).
- Hardware: NVIDIA A100 80GB on Thunder Compute, PyTorch + HuggingFace Transformers + PEFT, bf16 precision. Note: numbers may differ slightly across hardware due to floating-point determinism.
- Evaluation: 203-example FOLIO validation set, zero-shot, two-step label extraction (first line check, then rightmost valid label).

## What's in scope, what's out

In scope:
- Phi-3.5-mini + LoRA + FOLIO + ProofWriter as auxiliary data
- Multi-seed (3 seeds) at FOLIO, PW-2k, PW-5k, PW-10k, PW-2k-rebal
- Single-seed PW-20k (acknowledge limitation)
- The reweighting baseline at PW-2k

Out of scope (do not write claims about):
- LLaMA-3 8B comparison — was in original draft, removed in reframed paper
- Cross-model generalization — single model family
- Multiple task pairs — single benchmark only
- Mechanistic explanation beyond empirical observation
- Oversampling baseline (mention as future work but don't claim we tested it)
- Theoretical analysis

## Paper structure (target: 8 pages excluding references and limitations)

1. **Abstract** (~200-250 words)
2. **Introduction** (~1 page)
3. **Related Work** (~1 page, 4 subsections):
   - 2.1 Class imbalance: Cao 2019, Cui 2019, Lin 2017 focal loss, Buda 2018
   - 2.2 Mismatched auxiliary data: Oliver 2018 (SSL foundational), Chen 2020 UASD, Guo 2020 DS3L, Huang 2023 Fix-A-Step, AuxMix 2022, plus LLM SFT mismatch: TAIA Jiang 2024, GRAPE Zhang 2025, SDFT Yang 2024
   - 2.3 Scaling and data mixing in LLM fine-tuning: Zhang 2024 ICLR, Li 2025 SFT mixing, Shukor 2025, BiMix 2024
   - 2.4 Parameter-efficient fine-tuning for reasoning: Hu 2021 LoRA, Bilgin 2024 FLAIRS, Wang 2024 Tina
4. **Method** (~1-1.5 pages): datasets, prompt format, model and LoRA config, training protocol, reweighting implementation, evaluation
5. **Results** (~2-2.5 pages):
   - 4.1 Scaling effects (Finding 3)
   - 4.2 Class-asymmetric pattern (Finding 1 - headline)
   - 4.3 Reweighting redistributes errors (Finding 2 - supporting evidence)
6. **Discussion** (~0.5-1 page): mechanism speculation, why classical theory doesn't predict this, practitioner implications
7. **Conclusion** (~3-4 sentences)
8. **Limitations** (required by ACL): single benchmark, single model family, modest effect sizes, PW-20k single-seed, only one form of class-imbalance remedy tested (oversampling as future work), English only

## Required figures

1. **Figure 1 — Scaling curve**: Accuracy mean ± 95% CI across PW scale (x-axis: 0/2k/5k/10k/20k, y-axis: accuracy). Single line with error bars. The PW-20k point should be marked as single-seed or have no error bar with a note in caption.

2. **Figure 2 — Per-class recall across scaling**: Same x-axis (PW scale), three lines for True/False/Uncertain recall, with error bars across 3 seeds. This figure shows the class-asymmetric pattern visually — False line declining, Uncertain line stable/increasing.

3. **Figure 3 — Confusion matrices side-by-side**: Unweighted PW-2k vs Rebalanced PW-2k, both at seed 42 (or aggregated across seeds if cleaner). Shows where False examples go after reweighting — the prediction distribution shift. This is the money figure for Finding 2.

4. **Figure 4 (optional, if space)**: Prediction distribution bar chart across conditions. X-axis is condition (FOLIO, 2k, 5k, 10k, 20k, 2k-rebal). Three bars per condition (predicted-True, predicted-False, predicted-Uncertain counts). Visually shows over-prediction of Uncertain under reweighting.

## Required tables

1. **Table 1 — Main results table**: All conditions × all metrics. Mean ± std (or 95% CI) across seeds. Include accuracy, per-class precision/recall/F1, prediction distribution. Use this as the primary results reference.

2. **Table 2 — Hyperparameters**: All LoRA and training hyperparameters. Standard methodology table.

3. **Table 3 (optional) — Class counts and reweighting weights**: Shows the actual class distribution in PW-2k combined training set and the computed inverse-frequency weights. Important for reproducibility.

## Tone and writing style

- Honest about modest effect sizes. Don't overclaim.
- Precise about what's robust (per-class patterns, reweighting trade-off direction) vs variable (overall accuracy magnitudes).
- Anonymous (no "our prior work", no author names, no Georgia Tech mention)
- Third-person where self-reference is needed
- Standard ACL academic tone, no hyperbole
- Don't use words like "novel," "first," "discover," "breakthrough"
- Use precise language: "we observe," "we report," "we characterize," "consistent with," "contradicting the prediction of"
- Cite Zhang 2024, Li 2025, Oliver 2018, Cao 2019 prominently — these are the load-bearing prior-work citations

## Anonymization checklist (apply before submission)

- No author names
- No affiliations  
- No mention of Georgia Tech, course name, instructor
- GitHub link replaced with anonymized version (anonymous.4open.science) or removed
- "Our prior work" reframed in third person
- Acknowledgments removed (added back to camera-ready)

## Constraints

- Use ACL 2026 LaTeX template (downloaded to Overleaf project)
- 8 pages main paper, 1 page limitations, unlimited references and appendix
- Figures must be readable in B&W
- All numbers must come from the actual results CSV (no fabricated numbers)
- Don't update or change training code
- Don't fabricate citations — every cited paper must be real; verify any paper you don't recognize

## What I want from this session

[INSERT YOUR SPECIFIC SESSION GOAL HERE BEFORE PASTING]

Possible session goals:
- Day 1: "Draft the Methods section completely. Use the exact hyperparameters and protocol from this context. Show me the draft before saving to the LaTeX file."
- Day 2: "Generate all 4 figures using the results CSV. Save as PDFs to paper/figures/. Show me the matplotlib code before running."
- Day 2: "Draft the Results section, structuring it around the three findings as outlined above. Use real numbers from the results table."
- Day 3: "Draft the Related Work section with the 4 subsections specified. Use the exact citations listed. Each subsection should end with one sentence distinguishing our work."
- Day 3: "Draft the Introduction. Lead with the class-asymmetric finding as the puzzle, use reweighting as supporting evidence."
- Day 4: "Draft Discussion, Limitations, and Conclusion. Be honest about scope."
- Day 5: "Full anonymization pass and ACL formatting check."