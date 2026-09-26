import pandas as pd
import numpy as np
from src.blocking import FastCountryBlocker
from src.data_loader import load_ground_truth, load_source_tsv
from src.normalization import normalize_name_fast, normalize_address_fast

def diagnose_misses():
    print("=== DIAGNOSING BLOCKING MISSES ===")
    gt = load_ground_truth("dataset/train/train_ground_truth.tsv")
    s1_df = load_source_tsv("dataset/train/train_source1.tsv")
    s2_df = load_source_tsv("dataset/train/train_source2.tsv")
    s3_df = load_source_tsv("dataset/train/train_source3.tsv")
    
    val_s1 = s1_df[s1_df['country'] == 'US'].sample(n=2000, random_state=42)
    us_s2 = s2_df[s2_df['country'] == 'US']
    us_s3 = s3_df[s3_df['country'] == 'US']
    
    targets_df = pd.concat([us_s2, us_s3], ignore_index=True)
    target_ids = list(targets_df['entity_id'])
    target_names = list(targets_df['business_name'])
    target_addrs = list(targets_df['business_address'])
    
    blocker = FastCountryBlocker(country="US", max_posting_len=500, min_token_len=3, max_candidates=40)
    blocker.build_index(target_ids, target_names, target_addrs, verbose=False)
    
    target_id_to_idx = {tid: i for i, tid in enumerate(target_ids)}
    target_ids_set = set(target_ids)
    
    missed_examples = []
    
    s1_names = list(val_s1['business_name'])
    s1_addrs = list(val_s1['business_address'])
    s1_ids = list(val_s1['entity_id'])
    
    total_true_pairs = 0
    recalled_pairs = 0
    
    for i in range(len(val_s1)):
        s1_id = s1_ids[i]
        n_norm = normalize_name_fast(s1_names[i])
        a_norm = normalize_address_fast(s1_addrs[i])
        
        cand_indices = set(blocker.query(n_norm, a_norm))
        cand_ids = {blocker.target_ids[idx] for idx in cand_indices}
        
        true_targets = gt.get(s1_id, set()) & target_ids_set
        total_true_pairs += len(true_targets)
        for tid in true_targets:
            if tid in cand_ids:
                recalled_pairs += 1
            else:
                t_idx = target_id_to_idx[tid]
                t_n_norm = normalize_name_fast(target_names[t_idx])
                t_a_norm = normalize_address_fast(target_addrs[t_idx])
                missed_examples.append({
                    's1_id': s1_id,
                    's1_name': s1_names[i],
                    's1_addr': s1_addrs[i],
                    's1_n_norm': n_norm,
                    's1_a_norm': a_norm,
                    'tgt_id': tid,
                    'tgt_name': target_names[t_idx],
                    'tgt_addr': target_addrs[t_idx],
                    'tgt_n_norm': t_n_norm,
                    'tgt_a_norm': t_a_norm,
                })
            
    print(f"\nFound {len(missed_examples)} missed true match examples. Printing 20:")
    for j, ex in enumerate(missed_examples[:20], 1):
        print(f"\n[{j}] S1: {ex['s1_name']} || {ex['s1_addr']}")
        print(f"     Tgt: {ex['tgt_name']} || {ex['tgt_addr']}")
        print(f"     S1 base tokens: {ex['s1_n_norm'][4]} | Tgt base tokens: {ex['tgt_n_norm'][4]}")
        print(f"     S1 addr numbers: {ex['s1_a_norm'][3]} | Tgt addr numbers: {ex['tgt_a_norm'][3]}")
        print(f"     S1 addr tokens: {ex['s1_a_norm'][1][:4]} | Tgt addr tokens: {ex['tgt_a_norm'][1][:4]}")

if __name__ == "__main__":
    diagnose_misses()
