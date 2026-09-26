"""
End-to-End Entity Resolution Pipeline.
Coordinates data loading, country partitioning, inverted index blocking,
feature generation, LightGBM model training, dynamic sibling threshold tuning, and submission generation.
"""

import os
import sys
import time
import gc
from typing import Dict, Set, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data_loader import load_ground_truth, load_source_tsv, create_validation_split
from src.normalization import normalize_name_fast, normalize_address_fast
from src.features import extract_pairwise_features, FEATURE_NAMES
from src.blocking import FastCountryBlocker
from src.matcher import EntityMatcher
from src.threshold import sweep_dynamic_sibling_thresholds
from src.evaluation import evaluate_predictions

class Pipeline:
    def __init__(
        self,
        max_posting_len: int = 1500,
        max_candidates: int = 75,
        model_path: str = "experiments/entity_matcher.pkl"
    ):
        self.max_posting_len = max_posting_len
        self.max_candidates = max_candidates
        self.model_path = model_path
        self.matcher = EntityMatcher()
        self.primary_threshold = 0.82
        self.sibling_threshold = 0.58

    def build_country_blockers(
        self,
        targets_df: pd.DataFrame,
        verbose: bool = True
    ) -> Dict[str, FastCountryBlocker]:
        """Build country blockers for all targets in targets_df."""
        blockers = {}
        for country, group in targets_df.groupby('country'):
            if verbose:
                print(f"Building blocker for country: {country} ({len(group)} records)...")
            b = FastCountryBlocker(
                country=country,
                max_posting_len=self.max_posting_len,
                min_token_len=3,
                max_candidates=self.max_candidates
            )
            b.build_index(
                list(group['entity_id']),
                list(group['business_name']),
                list(group['business_address']),
                verbose=verbose
            )
            blockers[country] = b
        return blockers

    def build_training_dataset(
        self,
        s1_df: pd.DataFrame,
        blockers: Dict[str, FastCountryBlocker],
        ground_truth: Dict[str, Set[str]],
        n_sample_s1: int = 80000,
        negatives_per_positive: int = 2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build feature matrix X and binary labels y with true positives and hard negatives.
        """
        t0 = time.time()
        print(f"\n--- Generating Supervised Training Pairs (from {n_sample_s1} S1 samples) ---")
        
        sampled_s1 = s1_df.sample(n=min(n_sample_s1, len(s1_df)), random_state=random_state)
        s1_records = sampled_s1.to_dict('records')
        
        X_list = []
        y_list = []
        pos_count = 0
        neg_count = 0
        
        for s1_rec in tqdm(s1_records, desc="Mining pairs"):
            s1_id = s1_rec['entity_id']
            country = s1_rec['country']
            blocker = blockers.get(country)
            if not blocker:
                continue
                
            s1_n_norm = normalize_name_fast(s1_rec['business_name'])
            s1_a_norm = normalize_address_fast(s1_rec['business_address'])
            
            cand_indices = blocker.query(s1_n_norm, s1_a_norm)
            true_targets = ground_truth.get(s1_id, set())
            
            s1_pos_added = 0
            s1_neg_added = 0
            
            for idx in cand_indices:
                tid = blocker.target_ids[idx]
                tgt_name = blocker.target_names[idx]
                tgt_addr = blocker.target_addrs[idx]
                
                tgt_n_norm = normalize_name_fast(tgt_name)
                tgt_a_norm = normalize_address_fast(tgt_addr)
                
                feat = extract_pairwise_features(
                    (s1_n_norm, s1_a_norm),
                    (tgt_n_norm, tgt_a_norm)
                )
                
                is_match = 1 if tid in true_targets else 0
                
                if is_match == 1:
                    X_list.append(feat)
                    y_list.append(1)
                    pos_count += 1
                    s1_pos_added += 1
                else:
                    if s1_neg_added < negatives_per_positive * max(1, s1_pos_added):
                        X_list.append(feat)
                        y_list.append(0)
                        neg_count += 1
                        s1_neg_added += 1
                        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        print(f"Generated dataset in {time.time()-t0:.2f}s: X shape = {X.shape}, Positives = {pos_count}, Negatives = {neg_count}")
        return X, y

    def evaluate_validation(
        self,
        val_s1_df: pd.DataFrame,
        blockers: Dict[str, FastCountryBlocker],
        ground_truth: Dict[str, Set[str]],
        verbose: bool = True
    ) -> float:
        """
        Run end-to-end blocking, scoring, dynamic sibling threshold tuning, and evaluation on validation set.
        """
        print(f"\n=== RUNNING VALIDATION ON {len(val_s1_df)} S1 ENTITIES ===")
        t0 = time.time()
        
        val_items = []
        val_gt = {s1_id: ground_truth.get(s1_id, set()) for s1_id in val_s1_df['entity_id']}
        s1_records = val_s1_df.to_dict('records')
        
        recalled_cands = 0
        total_eval_true = 0
        
        for s1_rec in tqdm(s1_records, desc="Scoring Val S1"):
            s1_id = s1_rec['entity_id']
            country = s1_rec['country']
            blocker = blockers.get(country)
            if not blocker:
                continue
                
            s1_n_norm = normalize_name_fast(s1_rec['business_name'])
            s1_a_norm = normalize_address_fast(s1_rec['business_address'])
            
            cand_indices = blocker.query(s1_n_norm, s1_a_norm)
            
            true_m = val_gt.get(s1_id, set())
            total_eval_true += len(true_m)
            recalled_cands += sum(1 for idx in cand_indices if blocker.target_ids[idx] in true_m)
            
            if not cand_indices:
                val_items.append({
                    's1_id': s1_id,
                    'cand_tids': [],
                    'probs': np.array([]),
                    'cand_bases': [],
                    'cand_compact_bases': [],
                    'cand_nums': []
                })
                continue
                
            feats = []
            cand_tids = []
            cand_bases = []
            cand_compact_bases = []
            cand_nums = []
            
            for idx in cand_indices:
                tid = blocker.target_ids[idx]
                tgt_name = blocker.target_names[idx]
                tgt_addr = blocker.target_addrs[idx]
                
                tgt_n_norm = normalize_name_fast(tgt_name)
                tgt_a_norm = normalize_address_fast(tgt_addr)
                
                f = extract_pairwise_features(
                    (s1_n_norm, s1_a_norm),
                    (tgt_n_norm, tgt_a_norm)
                )
                feats.append(f)
                cand_tids.append(tid)
                cand_bases.append(tgt_n_norm[1])
                cand_compact_bases.append(tgt_n_norm[3])
                cand_nums.append(tgt_a_norm[3])
                
            feats_arr = np.array(feats, dtype=np.float32)
            probs = self.matcher.predict_proba(feats_arr)
            
            val_items.append({
                's1_id': s1_id,
                'cand_tids': cand_tids,
                'probs': probs,
                'cand_bases': cand_bases,
                'cand_compact_bases': cand_compact_bases,
                'cand_nums': cand_nums
            })
            
        print(f"\nCandidate Recall on Val: {recalled_cands} / {total_eval_true} ({recalled_cands/(total_eval_true+1e-9)*100:.2f}%)")
        
        best_p_th, best_s_th, best_f05, metrics = sweep_dynamic_sibling_thresholds(
            val_gt,
            val_items,
            verbose=verbose
        )
        self.primary_threshold = best_p_th
        self.sibling_threshold = best_s_th
        return best_f05

    def generate_submission(
        self,
        test_dir: str = "dataset/test",
        output_dir: str = "output",
        primary_threshold: Optional[float] = None,
        sibling_threshold: Optional[float] = None
    ):
        """
        Generate both output/candidate_pairs.tsv and output/matching_results.tsv for test set.
        Processes country by country to minimize peak memory usage.
        """
        p_th = primary_threshold if primary_threshold is not None else self.primary_threshold
        s_th = sibling_threshold if sibling_threshold is not None else self.sibling_threshold
        
        print(f"\n=== GENERATING TEST SUBMISSION (Primary Th = {p_th:.3f}, Sibling Th = {s_th:.3f}) ===")
        t0 = time.time()
        os.makedirs(output_dir, exist_ok=True)
        
        matching_path = os.path.join(output_dir, "matching_results.tsv")
        candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
        
        print("Loading test files...")
        test_s1 = load_source_tsv(os.path.join(test_dir, "test_source1.tsv"))
        test_s2 = load_source_tsv(os.path.join(test_dir, "test_source2.tsv"))
        test_s3 = load_source_tsv(os.path.join(test_dir, "test_source3.tsv"))
        
        print(f"Test S1: {len(test_s1)} | Test S2: {len(test_s2)} | Test S3: {len(test_s3)}")
        
        # Combine S2 + S3
        test_targets = pd.concat([test_s2, test_s3], ignore_index=True)
        del test_s2, test_s3
        gc.collect()
        
        # Open output files
        with open(matching_path, 'w', encoding='utf-8') as f_match, \
             open(candidate_path, 'w', encoding='utf-8') as f_cand:
             
            f_match.write("source1_entity_id\tmatched_entity_ids\n")
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            
            total_matches_count = 0
            total_cands_count = 0
            total_s1_processed = 0
            
            # Process country by country
            unique_countries = list(test_s1['country'].unique())
            print(f"Processing countries: {unique_countries}")
            
            for country in unique_countries:
                country_s1 = test_s1[test_s1['country'] == country]
                country_targets = test_targets[test_targets['country'] == country]
                
                print(f"\n--- Processing Country: {country} (S1: {len(country_s1)}, Targets: {len(country_targets)}) ---")
                
                blocker = FastCountryBlocker(
                    country=country,
                    max_posting_len=self.max_posting_len,
                    min_token_len=3,
                    max_candidates=self.max_candidates
                )
                blocker.build_index(
                    list(country_targets['entity_id']),
                    list(country_targets['business_name']),
                    list(country_targets['business_address']),
                    verbose=True
                )
                
                s1_records = country_s1.to_dict('records')
                
                for s1_rec in tqdm(s1_records, desc=f"Inference [{country}]"):
                    s1_id = s1_rec['entity_id']
                    
                    s1_n_norm = normalize_name_fast(s1_rec['business_name'])
                    s1_a_norm = normalize_address_fast(s1_rec['business_address'])
                    
                    cand_indices = blocker.query(s1_n_norm, s1_a_norm)
                    
                    if not cand_indices:
                        f_match.write(f"{s1_id}\t\n")
                        f_cand.write(f"{s1_id}\t\n")
                        total_s1_processed += 1
                        continue
                        
                    feats = []
                    cand_tids = []
                    cand_bases = []
                    cand_compact_bases = []
                    cand_nums = []
                    
                    for idx in cand_indices:
                        tid = blocker.target_ids[idx]
                        tgt_name = blocker.target_names[idx]
                        tgt_addr = blocker.target_addrs[idx]
                        
                        tgt_n_norm = normalize_name_fast(tgt_name)
                        tgt_a_norm = normalize_address_fast(tgt_addr)
                        
                        f = extract_pairwise_features(
                            (s1_n_norm, s1_a_norm),
                            (tgt_n_norm, tgt_a_norm)
                        )
                        feats.append(f)
                        cand_tids.append(tid)
                        cand_bases.append(tgt_n_norm[1])
                        cand_compact_bases.append(tgt_n_norm[3])
                        cand_nums.append(tgt_a_norm[3])
                        
                    feats_arr = np.array(feats, dtype=np.float32)
                    probs = self.matcher.predict_proba(feats_arr)
                    
                    # Anchor-Conditioned Dynamic Sibling Matching
                    primary_indices = [i for i in range(len(cand_tids)) if probs[i] >= p_th]
                    if primary_indices:
                        anchor_bases = {cand_bases[i] for i in primary_indices if cand_bases[i]}
                        anchor_cbases = {cand_compact_bases[i] for i in primary_indices if cand_compact_bases[i]}
                        anchor_nums = set()
                        for i in primary_indices:
                            anchor_nums.update(cand_nums[i])
                            
                        matched_ids = [cand_tids[i] for i in primary_indices]
                        for j in range(len(cand_tids)):
                            if j in primary_indices:
                                continue
                            if probs[j] >= s_th:
                                cb = cand_bases[j]
                                ccb = cand_compact_bases[j]
                                cn = cand_nums[j]
                                name_match = (cb and cb in anchor_bases) or (ccb and ccb in anchor_cbases)
                                num_match = bool(cn and (cn & anchor_nums))
                                if name_match or num_match:
                                    matched_ids.append(cand_tids[j])
                    else:
                        matched_ids = [cand_tids[i] for i in range(len(cand_tids)) if probs[i] >= (p_th + 0.04)]
                        
                    # Deduplicate while preserving order
                    cand_tids_unique = list(dict.fromkeys(cand_tids))
                    matched_ids_unique = list(dict.fromkeys(matched_ids))
                    
                    cand_str = ",".join(cand_tids_unique)
                    match_str = ",".join(matched_ids_unique)
                    
                    f_cand.write(f"{s1_id}\t{cand_str}\n")
                    f_match.write(f"{s1_id}\t{match_str}\n")
                    
                    total_cands_count += len(cand_tids_unique)
                    total_matches_count += len(matched_ids_unique)
                    total_s1_processed += 1
                    
                # Free country blocker memory
                del blocker, country_targets, country_s1
                gc.collect()
                
        print(f"\n=========================================================")
        print(f"Test Generation Complete in {time.time()-t0:.2f}s.")
        print(f"Total S1 processed: {total_s1_processed} (Expected: {len(test_s1)})")
        print(f"Total candidate pairs: {total_cands_count} (Avg: {total_cands_count/len(test_s1):.1f} per S1)")
        print(f"Total matched pairs: {total_matches_count} (Avg: {total_matches_count/len(test_s1):.2f} per S1)")
        print(f"Outputs written to:\n  {matching_path}\n  {candidate_path}")
        print("=========================================================")
