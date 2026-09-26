"""
Benchmark script to measure candidate generation recall, candidate counts, and runtime.
"""

import time
import numpy as np
import pandas as pd
from src.blocking import BlockingEngine
from src.data_loader import load_ground_truth, load_source_tsv

def run_blocking_benchmark():
    print("=== BLOCKING / CANDIDATE GENERATION BENCHMARK ===")
    
    # 1. Load ground truth
    print("Loading ground truth...")
    gt = load_ground_truth("dataset/train/train_ground_truth.tsv")
    
    # 2. Sample 10,000 S1 records for validation
    print("Loading S1...")
    s1_df = load_source_tsv("dataset/train/train_source1.tsv")
    val_s1 = s1_df.sample(n=10000, random_state=42)
    val_s1_dict = val_s1.set_index('entity_id').to_dict('index')
    
    # Check true matches for these 10,000 S1 records
    val_gt = {s1_id: gt.get(s1_id, set()) for s1_id in val_s1['entity_id']}
    total_true_pairs = sum(len(v) for v in val_gt.values())
    print(f"Sampled 10,000 S1 entities with {total_true_pairs} true matching targets.")
    
    # Load S2 and S3 (sample or full subset)
    # Let's load full S2 and S3 for US and India to evaluate realistic blocking
    print("Loading S2 and S3 (first 500,000 records of each for quick test)...")
    s2_df = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", nrows=500000, keep_default_na=False)
    s3_df = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", nrows=500000, keep_default_na=False)
    
    # Ensure all true targets for our val set are in targets_df to test recall
    all_needed_target_ids = set()
    for targets in val_gt.values():
        all_needed_target_ids |= targets
        
    print(f"Needed target IDs in GT: {len(all_needed_target_ids)}")
    
    # Load all targets that are in ground truth for this sample to accurately assess recall
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    # Build blocking engine
    engine = BlockingEngine(max_token_df=5000, min_token_len=3, max_candidates_per_s1=50)
    engine.index_targets(targets_df, verbose=True)
    
    # Generate candidates for val_s1
    print(f"Generating candidates for {len(val_s1)} S1 entities...")
    t0 = time.time()
    
    cand_counts = []
    recalled_pairs = 0
    eval_true_pairs = 0
    
    available_target_ids = set(targets_df['entity_id'])
    
    for _, row in val_s1.iterrows():
        s1_id = row['entity_id']
        cands = engine.generate_candidates_for_s1(row.to_dict())
        cand_set = set(cands)
        cand_counts.append(len(cands))
        
        # Check recall against true matches that are present in the target index
        true_targets = val_gt.get(s1_id, set()) & available_target_ids
        eval_true_pairs += len(true_targets)
        recalled_pairs += len(true_targets & cand_set)
        
    t_gen = time.time() - t0
    cand_counts = np.array(cand_counts)
    
    print("\n--- Benchmark Results ---")
    print(f"Time for {len(val_s1)} S1 entities: {t_gen:.2f}s ({len(val_s1)/t_gen:.1f} S1/sec)")
    print(f"Evaluated true pairs present in targets: {eval_true_pairs}")
    print(f"Recalled true pairs: {recalled_pairs}")
    print(f"Candidate Recall Ceiling: {recalled_pairs / (eval_true_pairs + 1e-9) * 100:.2f}%")
    print(f"Candidates per S1: mean={cand_counts.mean():.1f}, median={np.median(cand_counts):.1f}, p95={np.percentile(cand_counts, 95):.1f}, max={cand_counts.max()}")

if __name__ == "__main__":
    run_blocking_benchmark()
