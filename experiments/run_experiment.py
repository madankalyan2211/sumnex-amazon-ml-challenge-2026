"""
Main Experiment Runner:
1. Splits training data into Train and Held-Out Validation.
2. Builds training target blockers.
3. Trains Supervised LightGBM Matcher on positives + hard negatives.
4. Sweeps decision thresholds and evaluates on Held-Out Validation.
5. Generates test output files (candidate_pairs.tsv and matching_results.tsv).
6. Runs the official challenge submission validator.
7. Packages and delivers submission files to Downloads.
"""

import os
import sys
import time
import gc
import pandas as pd
import numpy as np

from src.data_loader import load_ground_truth, load_source_tsv, create_validation_split
from src.pipeline import Pipeline
from src.evaluation import evaluate_predictions

def run_experiment():
    print("=========================================================")
    print("AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION")
    print("=========================================================")
    
    t_start = time.time()
    
    # 1. Load Ground Truth and Train Sources
    print("\n[Step 1/5] Loading Ground Truth and Training Data...")
    gt = load_ground_truth("dataset/train/train_ground_truth.tsv")
    s1_df = load_source_tsv("dataset/train/train_source1.tsv")
    s2_df = load_source_tsv("dataset/train/train_source2.tsv")
    s3_df = load_source_tsv("dataset/train/train_source3.tsv")
    
    print(f"Total S1: {len(s1_df)} | Total S2: {len(s2_df)} | Total S3: {len(s3_df)} | Total GT entries: {len(gt)}")
    
    # 2. Held-Out Validation Split (25,000 S1 records for validation)
    print("\n[Step 2/5] Creating Held-Out Validation Split...")
    val_s1 = s1_df.sample(n=25000, random_state=42)
    val_s1_ids = set(val_s1['entity_id'])
    train_s1 = s1_df[~s1_df['entity_id'].isin(val_s1_ids)].copy()
    
    print(f"Training S1 Pool: {len(train_s1)} records | Validation S1: {len(val_s1)} records")
    
    # 3. Initialize Pipeline
    pipeline = Pipeline(
        max_posting_len=1500,
        max_candidates=75,
        model_path="experiments/entity_matcher.pkl"
    )
    
    # 4. Build Country Blockers on Train Targets
    print("\n[Step 3/5] Building Inverted Index Blockers on Training Targets...")
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    blockers = pipeline.build_country_blockers(targets_df, verbose=True)
    
    # 5. Mining Training Pairs and Fitting LightGBM
    print("\n[Step 4/5] Mining Training Pairs and Training LightGBM Matcher...")
    X_train, y_train = pipeline.build_training_dataset(
        train_s1, blockers, gt,
        n_sample_s1=100000,
        negatives_per_positive=2,
        random_state=42
    )
    
    # Train LightGBM model
    pipeline.matcher.fit(X_train, y_train)
    pipeline.matcher.save("experiments/entity_matcher.pkl")
    
    # 6. Held-Out Validation & Threshold Optimization
    print("\n[Step 5/5] Evaluating on Held-Out Validation Set & Optimizing Macro F0.5...")
    best_f05 = pipeline.evaluate_validation(
        val_s1, blockers, gt,
        verbose=True
    )
    
    # Free memory before test inference
    del targets_df, blockers, s1_df, train_s1, val_s1, X_train, y_train, gt
    gc.collect()
    
    # 7. Generate Test Submission
    print("\n[Inference] Generating Final Test Predictions...")
    pipeline.generate_submission(
        test_dir="dataset/test",
        output_dir="output",
        primary_threshold=pipeline.primary_threshold,
        sibling_threshold=pipeline.sibling_threshold
    )
    
    print(f"\nTotal Pipeline Execution Time: {time.time()-t_start:.2f}s")
    print("\nRunning Official Validator...")
    exit_code = os.system("python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test")
    if exit_code == 0:
        print("\nSUCCESS: Official validator PASSED with 0 errors!")
        print("\nPackaging and Delivering to Downloads...")
        os.system("python3 deliver_to_downloads.py")
    else:
        print("\nWARNING: Official validator returned exit code:", exit_code)

if __name__ == "__main__":
    run_experiment()
