import time
import pandas as pd
import numpy as np
from src.blocking import FastCountryBlocker
from src.data_loader import load_ground_truth, load_source_tsv
from src.normalization import normalize_name_fast, normalize_address_fast

def benchmark_fast_blocking():
    print("=== BENCHMARKING FAST BLOCKING ===")
    
    # 1. Load GT
    print("Loading GT...")
    gt = load_ground_truth("dataset/train/train_ground_truth.tsv")
    
    # 2. Sample 20,000 S1 records
    s1_df = load_source_tsv("dataset/train/train_source1.tsv")
    val_s1 = s1_df.sample(n=20000, random_state=42)
    
    # Load S2 and S3 for US
    print("Loading S2 and S3...")
    s2_df = load_source_tsv("dataset/train/train_source2.tsv")
    s3_df = load_source_tsv("dataset/train/train_source3.tsv")
    
    us_val_s1 = val_s1[val_s1['country'] == 'US'].copy()
    us_s2 = s2_df[s2_df['country'] == 'US']
    us_s3 = s3_df[s3_df['country'] == 'US']
    
    targets_df = pd.concat([us_s2, us_s3], ignore_index=True)
    target_ids = list(targets_df['entity_id'])
    target_names = list(targets_df['business_name'])
    target_addrs = list(targets_df['business_address'])
    
    print(f"Building US index for {len(target_ids)} target records...")
    t0 = time.time()
    blocker = FastCountryBlocker(country="US", max_posting_len=500, min_token_len=3, max_candidates=40)
    blocker.build_index(target_ids, target_names, target_addrs, verbose=True)
    
    # Query for US val S1 records
    print(f"\nQuerying {len(us_val_s1)} US S1 records...")
    t_query = time.time()
    
    recalled = 0
    total_eval_true = 0
    target_id_set = set(target_ids)
    
    cand_lens = []
    
    s1_names = list(us_val_s1['business_name'])
    s1_addrs = list(us_val_s1['business_address'])
    s1_ids = list(us_val_s1['entity_id'])
    
    for i in range(len(us_val_s1)):
        s1_id = s1_ids[i]
        n_norm = normalize_name_fast(s1_names[i])
        a_norm = normalize_address_fast(s1_addrs[i])
        
        cand_indices = blocker.query(n_norm, a_norm)
        cand_ids = {blocker.target_ids[idx] for idx in cand_indices}
        cand_lens.append(len(cand_ids))
        
        true_m = gt.get(s1_id, set()) & target_id_set
        total_eval_true += len(true_m)
        recalled += len(true_m & cand_ids)
        
    t_elapsed = time.time() - t_query
    print(f"\nQuery time: {t_elapsed:.2f}s ({len(us_val_s1)/t_elapsed:.1f} queries/sec)")
    print(f"Evaluated true matches: {total_eval_true}")
    print(f"Recalled matches: {recalled} ({recalled/(total_eval_true+1e-9)*100:.2f}%)")
    print(f"Candidates per query: mean={np.mean(cand_lens):.1f}, median={np.median(cand_lens)}, p95={np.percentile(cand_lens, 95):.1f}, max={max(cand_lens)}")

if __name__ == "__main__":
    benchmark_fast_blocking()
