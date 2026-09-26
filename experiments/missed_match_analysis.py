"""
Detailed Failure Mode Analysis for Missed Candidate Matches.
Categorizes missed pairs and computes recovery potential for candidate recall improvement.
"""

import time
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

from src.data_loader import load_ground_truth, load_source_tsv
from src.normalization import normalize_name_fast, normalize_address_fast
from src.blocking import FastCountryBlocker

def analyze_missed_matches():
    print("=== MISSED MATCH ROOT CAUSE ANALYSIS ===")
    
    gt = load_ground_truth("dataset/train/train_ground_truth.tsv")
    s1_df = load_source_tsv("dataset/train/train_source1.tsv")
    s2_df = load_source_tsv("dataset/train/train_source2.tsv")
    s3_df = load_source_tsv("dataset/train/train_source3.tsv")
    
    # Evaluate across a sample of 10,000 S1 records (US and India)
    sample_s1 = s1_df.sample(n=10000, random_state=42)
    sample_s1_dict = sample_s1.set_index('entity_id').to_dict('index')
    
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    target_id_to_idx = {tid: i for i, tid in enumerate(targets_df['entity_id'])}
    
    # Build country blockers
    blockers = {}
    for country, group in targets_df.groupby('country'):
        print(f"Building index for {country} ({len(group)} records)...")
        b = FastCountryBlocker(country=country, max_posting_len=500, min_token_len=3, max_candidates=45)
        b.build_index(list(group['entity_id']), list(group['business_name']), list(group['business_address']), verbose=False)
        blockers[country] = b
        
    print("\nQuerying sample S1 records and identifying missed positive pairs...")
    
    total_true_pairs = 0
    recalled_pairs = 0
    missed_pairs = []
    
    for s1_id, s1_rec in sample_s1_dict.items():
        country = s1_rec['country']
        blocker = blockers.get(country)
        if not blocker:
            continue
            
        s1_n_norm = normalize_name_fast(s1_rec['business_name'])
        s1_a_norm = normalize_address_fast(s1_rec['business_address'])
        
        cand_indices = blocker.query(s1_n_norm, s1_a_norm)
        cand_ids = {blocker.target_ids[idx] for idx in cand_indices}
        
        true_m = gt.get(s1_id, set())
        for tid in true_m:
            if tid in target_id_to_idx:
                total_true_pairs += 1
                if tid in cand_ids:
                    recalled_pairs += 1
                else:
                    t_idx = target_id_to_idx[tid]
                    missed_pairs.append({
                        's1_id': s1_id,
                        's1_name': s1_rec['business_name'],
                        's1_addr': s1_rec['business_address'],
                        's1_n_norm': s1_n_norm,
                        's1_a_norm': s1_a_norm,
                        'tgt_id': tid,
                        'tgt_name': targets_df['business_name'].iloc[t_idx],
                        'tgt_addr': targets_df['business_address'].iloc[t_idx],
                        'country': country
                    })
                    
    print(f"\nTotal Evaluated True Pairs: {total_true_pairs}")
    print(f"Recalled True Pairs: {recalled_pairs} ({recalled_pairs/total_true_pairs*100:.2f}%)")
    print(f"Missed True Pairs: {len(missed_pairs)} ({len(missed_pairs)/total_true_pairs*100:.2f}%)")
    
    # Categorize root causes of missed pairs
    categories = Counter()
    for p in missed_pairs:
        s1_name_tokens = set(p['s1_n_norm'][4])
        tgt_n_norm = normalize_name_fast(p['tgt_name'])
        tgt_name_tokens = set(tgt_n_norm[4])
        
        s1_nums = p['s1_a_norm'][3]
        tgt_a_norm = normalize_address_fast(p['tgt_addr'])
        tgt_nums = tgt_a_norm[3]
        
        has_shared_token = bool(s1_name_tokens & tgt_name_tokens)
        has_shared_num = bool(s1_nums & tgt_nums)
        
        if not p['tgt_name'] or not p['s1_name']:
            categories['missing_name'] += 1
        elif not p['tgt_addr'] or not p['s1_addr']:
            categories['missing_address'] += 1
        elif has_shared_num and not has_shared_token:
            categories['shared_addr_num_only'] += 1
        elif has_shared_token and not has_shared_num:
            categories['shared_token_low_rank'] += 1
        elif not has_shared_token and not has_shared_num:
            categories['heavy_corruption_both'] += 1
        else:
            categories['other_ranking_cutoff'] += 1
            
    print("\n--- Failure Mode Distribution ---")
    for cat, cnt in categories.most_common():
        pct = cnt / len(missed_pairs) * 100
        print(f"  - {cat:25s}: {cnt:5d} ({pct:5.1f}%)")

if __name__ == "__main__":
    analyze_missed_matches()
