import pandas as pd
import numpy as np
from collections import Counter
from src.blocking import FastCountryBlocker
from src.data_loader import load_ground_truth
from src.normalization import normalize_name_fast, normalize_address_fast

def run_fast_diagnosis():
    print("=== FAST MISSED MATCH ANALYSIS ===")
    
    # Load ground truth
    print("Loading ground truth...")
    gt = load_ground_truth("dataset/train/train_ground_truth.tsv")
    
    # Load S1 sample (5000 S1 records)
    print("Loading train datasets...")
    s1_df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
    s2_df = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
    s3_df = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
    
    s1_us = s1_df[s1_df['country'] == 'US'].sample(n=3000, random_state=42)
    s2_us = s2_df[s2_df['country'] == 'US']
    s3_us = s3_df[s3_df['country'] == 'US']
    
    # Index all US targets
    print("Indexing US targets...")
    tgt_df = pd.concat([s2_us, s3_us], ignore_index=True)
    target_ids = list(tgt_df['entity_id'])
    target_names = list(tgt_df['business_name'])
    target_addrs = list(tgt_df['business_address'])
    
    blocker = FastCountryBlocker(country="US", max_posting_len=500, min_token_len=3, max_candidates=50)
    blocker.build_index(target_ids, target_names, target_addrs, verbose=True)
    
    target_id_to_idx = {tid: i for i, tid in enumerate(target_ids)}
    target_ids_set = set(target_ids)
    
    s1_names = list(s1_us['business_name'])
    s1_addrs = list(s1_us['business_address'])
    s1_ids = list(s1_us['entity_id'])
    
    total_true = 0
    recalled = 0
    missed_examples = []
    failure_reasons = Counter()
    
    for i in range(len(s1_us)):
        s1_id = s1_ids[i]
        n_norm = normalize_name_fast(s1_names[i])
        a_norm = normalize_address_fast(s1_addrs[i])
        
        cand_indices = blocker.query(n_norm, a_norm)
        cand_ids = {blocker.target_ids[idx] for idx in cand_indices}
        
        true_tgts = gt.get(s1_id, set()) & target_ids_set
        total_true += len(true_tgts)
        
        for tid in true_tgts:
            if tid in cand_ids:
                recalled += 1
            else:
                t_idx = target_id_to_idx[tid]
                t_n_norm = normalize_name_fast(target_names[t_idx])
                t_a_norm = normalize_address_fast(target_addrs[t_idx])
                
                # Analyze why it was missed
                # Check 1: Shared base name?
                shared_base = (n_norm[1] == t_n_norm[1])
                # Check 2: Shared tokens?
                shared_tokens = set(n_norm[4]) & set(t_n_norm[4])
                # Check 3: Shared address numbers?
                shared_nums = set(n_norm[3] if len(n_norm) > 3 else []) & set(t_n_norm[3] if len(t_n_norm) > 3 else [])
                shared_addr_nums = set(a_norm[3]) & set(t_a_norm[3])
                shared_addr_tokens = set(a_norm[1]) & set(t_a_norm[1])
                
                reason = "other"
                if shared_base:
                    reason = "posting_list_truncated (high frequency base name)"
                elif shared_tokens:
                    reason = "token_filtered_or_low_score"
                elif shared_addr_nums and shared_addr_tokens:
                    reason = "address_pair_not_indexed_or_different_token_length"
                elif shared_addr_nums and not shared_tokens:
                    reason = "name_heavy_typo_with_shared_number"
                elif not shared_addr_nums and not shared_tokens:
                    reason = "severe_name_and_addr_mismatch"
                
                failure_reasons[reason] += 1
                
                if len(missed_examples) < 25:
                    missed_examples.append({
                        's1_name': s1_names[i],
                        's1_addr': s1_addrs[i],
                        's1_base': n_norm[1],
                        's1_tokens': n_norm[4],
                        's1_addr_nums': a_norm[3],
                        's1_addr_toks': a_norm[1],
                        'tgt_name': target_names[t_idx],
                        'tgt_addr': target_addrs[t_idx],
                        'tgt_base': t_n_norm[1],
                        'tgt_tokens': t_n_norm[4],
                        'tgt_addr_nums': t_a_norm[3],
                        'tgt_addr_toks': t_a_norm[1],
                        'reason': reason
                    })
                    
    print(f"\nResults over {len(s1_us)} S1 queries:")
    print(f"  Total true links: {total_true}")
    print(f"  Recalled by blocker: {recalled} ({recalled/total_true*100:.2f}%)")
    print(f"  Missed by blocker: {total_true - recalled} ({(total_true-recalled)/total_true*100:.2f}%)")
    
    print("\nFailure breakdown:")
    for reason, count in failure_reasons.most_common():
        print(f"  - {reason}: {count} ({count/(total_true-recalled)*100:.1f}%)")
        
    print("\nSample Missed Matches:")
    for j, ex in enumerate(missed_examples[:15], 1):
        print(f"\n[{j}] Reason: {ex['reason']}")
        print(f"  S1 : {ex['s1_name']} || {ex['s1_addr']}")
        print(f"  TGT: {ex['tgt_name']} || {ex['tgt_addr']}")
        print(f"  S1 tokens: {ex['s1_tokens']} | TGT tokens: {ex['tgt_tokens']}")
        print(f"  S1 nums: {ex['s1_addr_nums']} | TGT nums: {ex['tgt_addr_nums']}")

if __name__ == "__main__":
    run_fast_diagnosis()
