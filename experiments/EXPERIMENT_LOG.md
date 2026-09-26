# Amazon ML Challenge 2026 — Experiment Log

| Exp ID | Version | Description | Blocking Strategy | Features | Model | Optimal Threshold | Candidate Recall | Macro F0.5 | Micro Precision | Micro Recall | Singleton Acc | Status / Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | **V1 (Baseline)** | Multi-channel inverted index + LightGBM + Macro F0.5 threshold sweep | Base name, clean name, name tokens, prefix (6), addr number+street token | 24 pairwise lexical, phonetic, numeric & cross-field features | LightGBM (63 leaves, 350 trees, lr=0.06) | 0.820 | 82.10% | **0.8234** | 97.68% | 69.32% | 93.02% | **Baseline Candidate V1** (Inference running) |

---

## Experiment Details

### EXP-01 (Baseline V1)
- **Validation Split**: 25,000 S1 entities stratified by country (`train_source1.tsv`).
- **Target Index**: All training target records (`train_source2.tsv` + `train_source3.tsv` = 10,320,219 records).
- **Training Samples**: 60,000 S1 queries generating 496,056 training pairs (160,455 positive, 335,601 hard negative).
- **Threshold Optimization**: Tested 31 thresholds from 0.30 to 0.90 (step 0.02). Peak Macro F0.5 at $\tau^* = 0.820$.
- **Validation Metrics**:
  - Macro $F_{0.5}$: **0.8234**
  - Micro Precision: **97.68%**
  - Micro Recall: **69.32%**
  - Singleton / Zero-match Accuracy: **93.02%**
  - Candidate Blocking Recall: **82.10%**
- **Test Inference**: Country-by-country streaming (`US` $\to$ `France` $\to$ `India`).
