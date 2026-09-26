# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Team Members:** Lead ML Engineer & Submission Manager  
**Submission Date:** September 25, 2026  

---

## 1. Executive Summary
We developed an ultra-scalable, precision-first multi-source entity resolution pipeline for linking noisy business records across three independent data sources. Our architecture employs open-set country partitioning, single-pass inverted index blocking with address number-token pairs, 24 comprehensive lexical/phonetic/numeric pairwise features, and a gradient-boosted decision tree (LightGBM) trained on mined hard negatives. Optimizing specifically for the challenge metric (Macro $F_{0.5}$) on a 25,000 held-out validation set yielded an optimal decision threshold of $\tau^* = 0.820$, achieving a **Macro $F_{0.5}$ score of 0.8234** with **97.68% precision** and **93.02% singleton accuracy**.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across the 26.4 million records revealed critical properties:
- **Hard Country Boundary**: 100.0000% of all 7,638,365 ground truth links are strictly within the same country. While the training data spans `US` (60.0%) and `India` (40.0%), the test set introduces `France` (~15.0%). Our pipeline treats country as a dynamic open set rather than hardcoded categories.
- **Match Multiplicity**: Ground truth exhibits a heavy multi-match distribution: 5.58% singletons (0 matches), 5.40% 1-to-1 matches, 17.00% 2 matches, and 72.01% 3+ matches (up to 11 matches per Source 1 entity).
- **Name Noise Patterns**: Includes legal suffix variations (`LLC`, `Corp`, `Pvt Ltd`, `SARL`, `SAS`), typos (`Offie` vs `Office`), character mutations/leet-speak (`5cheer` vs `scheer`, `tttil` vs `tactical`), domain name conversions (`teamair.com` vs `Team Air`), and transliterated Hindi/French scripts.
- **Address Inconsistencies**: Reordered address components, street/road abbreviations (`Ave`, `Dr`, `Blvd`, `St`), punctuation noise, and landmark variations. In 90.44% of true matching pairs with numbers in addresses, at least one normalized numeric token was shared.

### 2.2 Solution Strategy
**Approach Type:** Multi-Channel Inverted Index Blocking + Pairwise Gradient Boosted Tree Matching + Macro $F_{0.5}$ Threshold Optimization.  
**Core Innovation:** Single-pass inverted index blocking integrating normalized address number-street token tuples (`f"{num}_{token}"`) alongside legal-suffix-stripped base names, ensuring high candidate recall (>82%) with sub-millisecond query latency and strict RAM bounds.

---

## 3. Candidate Generation (Blocking)

To avoid $O(N \times M)$ comparisons across millions of records, candidate generation runs per country partition:
- **Blocking Keys Used**:
  1. *Exact Base Name*: Business name with international legal suffixes stripped.
  2. *Exact Clean Name*: Full normalized name without punctuation/accents.
  3. *Informative Name Tokens*: Distinctive tokens (length $\ge 3$) filtered by posting list thresholds.
  4. *Name Token Bigrams*: Consecutive word bigrams (e.g., `pediatric_dental`).
  5. *Address Number + Token Pairs*: Normalized house/building numbers paired with road/city tokens (e.g., `12003_winding`, `180_soper`).
  6. *Compact Prefix*: First 6 alphanumeric characters for domain/compressed names.
- **Candidate Filtering**: Bounded top-$K$ candidates ($K=40$–$45$) ranked by multi-channel score accumulation.
- **Recall Assurance**: Combining lexical name indexing with numeric address key indexing captures matches with heavy name mutations or domain abbreviations without sacrificing candidate compactness.

---

## 4. Matching Model

### 4.1 Feature Engineering (24 Pairwise Features)
1. **Name Features**:
   - `name_exact_clean`, `name_exact_base`, `name_exact_compact`
   - `name_token_jaccard`, `name_base_jaccard`, `name_token_overlap`
   - `name_len_ratio`, `name_char_3gram_jaccard`
   - `name_fuzz_ratio`, `name_fuzz_partial_ratio`, `name_fuzz_token_sort_ratio`
2. **Address Features**:
   - `addr_exact_clean`, `addr_token_jaccard`, `addr_token_overlap`
   - `addr_num_exact`, `addr_num_jaccard`, `addr_num_overlap`, `addr_num_mismatch`
   - `addr_empty_s1`, `addr_empty_tgt`, `addr_fuzz_token_set_ratio`
3. **Cross-Field Interaction Features**:
   - `name_addr_joint_score` ($0.6 \times \text{NameJaccard} + 0.4 \times \text{AddrJaccard}$)
   - `high_name_high_addr` (Boolean flag for high name + address confidence)
   - `exact_name_same_num` (Exact base name match with verified shared address number)

### 4.2 Model Architecture & Training
- **Model Type**: LightGBM Binary Classifier (`num_leaves=63`, `learning_rate=0.06`, `colsample_bytree=0.8`, `subsample=0.8`).
- **Hard Negative Mining**: Trained on true positive links paired with hard negatives mined directly from the candidate generator (entities sharing tokens/numbers but referring to different businesses).
- **Threshold Selection**: Fine-grained grid sweep on held-out validation data directly optimizing the challenge Macro $F_{0.5}$ metric. Because $F_{0.5}$ weights precision $2\times$ over recall, the optimal threshold shifts to $\tau^* = 0.820$, drastically cutting false merges while preserving high recall on strong matches.

---

## 5. Experimental Results & Evaluation

- **Macro $F_{0.5}$ Score:** **0.8234** (on 25,000 held-out S1 validation entities)
- **Micro Precision:** **97.68%**
- **Micro Recall:** **69.32%**
- **Zero-Match Entity Accuracy:** **93.02%** (correctly predicts empty set for unlinked entities)
- **Singleton Match (1 match) Macro $F_{0.5}$:** **0.7845**
- **Multi-Match (2+ matches) Macro $F_{0.5}$:** **0.8291**
- **Candidate Blocking Recall:** **82.10%**

### Error Analysis:
- **Common False Positives (Wrong Merges):** Franchise businesses sharing near-identical brand names and occupying adjacent unit numbers within the same commercial shopping center or strip mall.
- **Common False Negatives (Missed Matches):** Entities where both name and address were heavily corrupted simultaneously (e.g., severe OCR errors plus incomplete/missing address details).

---

## 6. Conclusion
By leveraging open-set country partitioning, high-throughput inverted index blocking, rich pairwise feature interactions, and precision-calibrated LightGBM ranking, our system delivers high Macro $F_{0.5}$ performance while maintaining memory scalability and sub-second per-record inference throughput.

---

## Appendix

### A. Code Artefacts
- `code/business_entity_resolution/src/`:
  - `normalization.py`: Ultra-fast string cleaner and token/number extractor (>150k strings/sec).
  - `blocking.py`: `FastCountryBlocker` inverted index engine.
  - `features.py`: 24-dimensional pairwise feature extractor.
  - `matcher.py`: LightGBM supervised entity classifier.
  - `evaluation.py`: Exact Macro $F_{0.5}$ and diagnostic metric calculator.
  - `threshold.py`: Macro $F_{0.5}$ decision threshold optimizer.
  - `pipeline.py`: End-to-end country-partitioned streamed execution pipeline.
- `output/matching_results.tsv`: Final predictions for all test Source 1 entities.
- `output/candidate_pairs.tsv`: Final candidate set fed to the matching model.

### B. Reproducibility
All files can be regenerated from scratch by running:
```bash
PYTHONPATH=. python3 experiments/run_experiment.py
```
