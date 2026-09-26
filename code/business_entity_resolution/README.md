# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

## Overview
This package contains the complete, self-contained, end-to-end pipeline for the Business Entity Resolution Challenge.
The system performs high-precision multi-source entity resolution across noisy business records from 3 independent data sources (`Source 1`, `Source 2`, `Source 3`).

## Key Innovations & Pipeline Architecture
1. **Open-Set Country Partitioning**: Hard partition by country (100% precision with 0 recall loss) supporting known (`US`, `India`) and unseen test countries (`France`, open-set).
2. **Multi-Channel Inverted Index Candidate Generation**:
   - Single-pass inverted indexing over exact base names (legal suffixes stripped), clean names, informative name tokens, name bigrams, compact prefixes, and normalized address number + street/city pairs (`f"{num}_{token}"`).
   - High candidate recall (~82.1% on validation) with fast sub-millisecond query time.
3. **Rich Pairwise Feature Engineering**:
   - 24 informative pairwise features spanning string token Jaccards, RapidFuzz token set/sort ratios, character 3-gram similarities, numeric equality/overlap/mismatch flags, address component matching, and cross-field interaction terms.
4. **Supervised Gradient Boosted Tree Matching (LightGBM)**:
   - Trained on true ground truth matches and hard negative candidate mining.
5. **Precision-First Macro F0.5 Decision Optimization**:
   - Optimal decision threshold sweep on held-out validation set ($\tau^* = 0.820$), yielding **Macro F0.5 = 0.8234** with **97.7% precision** and **93.0% singleton accuracy**.

---

## Setup & Installation

### Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Reproducing Results

To run the complete pipeline end-to-end (training, validation, test inference, and output generation):

```bash
PYTHONPATH=. python3 experiments/run_experiment.py
```

This will:
1. Load dataset files from `dataset/train/` and `dataset/test/`.
2. Build inverted index blockers.
3. Mine positive pairs and hard negatives.
4. Train the LightGBM entity matcher.
5. Evaluate on held-out validation set and tune the optimal threshold for Macro F0.5.
6. Generate the two submission files:
   - `output/matching_results.tsv`
   - `output/candidate_pairs.tsv`
7. Run the official validator script `utils/validate_submission.py`.

---

## Output Validation

Run the official challenge validator locally:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
