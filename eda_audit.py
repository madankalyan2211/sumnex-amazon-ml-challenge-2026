import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
import time

def audit_sources():
    print("=== EDA: DATASET AUDIT ===")
    
    # 1. Ground Truth Analysis
    print("\n--- Analyzing train_ground_truth.tsv ---")
    t0 = time.time()
    gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str)
    print(f"Loaded ground truth in {time.time()-t0:.2f}s, shape: {gt_df.shape}")
    print(f"Columns: {list(gt_df.columns)}")
    print(f"Unique S1 IDs in GT: {gt_df['source1_entity_id'].nunique()}")
    
    # Check nulls / empty matches
    gt_df['matched_entity_ids'] = gt_df['matched_entity_ids'].fillna('')
    match_lists = gt_df['matched_entity_ids'].apply(lambda x: [m.strip() for m in x.split(',') if m.strip()])
    match_counts = match_lists.apply(len)
    
    print("\nMatch Count Distribution:")
    print(f"Total S1 entities: {len(gt_df)}")
    print(f"0 matches (singletons/unmatched): {(match_counts == 0).sum()} ({(match_counts == 0).mean()*100:.2f}%)")
    print(f"1 match: {(match_counts == 1).sum()} ({(match_counts == 1).mean()*100:.2f}%)")
    print(f"2 matches: {(match_counts == 2).sum()} ({(match_counts == 2).mean()*100:.2f}%)")
    print(f"3+ matches: {(match_counts >= 3).sum()} ({(match_counts >= 3).mean()*100:.2f}%)")
    print(f"Max matches for an S1: {match_counts.max()}")
    print(f"Mean matches per S1: {match_counts.mean():.4f}")
    print(f"Total ground truth pair links: {match_counts.sum()}")
    
    # Breakdown of match IDs: S2 vs S3
    s2_count = 0
    s3_count = 0
    other_prefix = 0
    for lst in match_lists:
        for mid in lst:
            if mid.startswith('S2-'):
                s2_count += 1
            elif mid.startswith('S3-'):
                s3_count += 1
            else:
                other_prefix += 1
    print(f"Matched target IDs: S2 = {s2_count} ({s2_count/match_counts.sum()*100:.2f}%), S3 = {s3_count} ({s3_count/match_counts.sum()*100:.2f}%), other = {other_prefix}")

    # Inspect train files
    train_files = {
        'train_source1': 'dataset/train/train_source1.tsv',
        'train_source2': 'dataset/train/train_source2.tsv',
        'train_source3': 'dataset/train/train_source3.tsv',
    }
    
    country_stats = {}
    for name, path in train_files.items():
        print(f"\n--- Analyzing {name} ---")
        t0 = time.time()
        df = pd.read_csv(path, sep="\t", dtype=str)
        print(f"Loaded {name} in {time.time()-t0:.2f}s, shape: {df.shape}")
        print(f"Null counts:\n{df.isnull().sum()}")
        print(f"Empty string counts:\n{(df == '').sum()}")
        print(f"Country value counts:\n{df['country'].value_counts(dropna=False)}")
        print(f"Unique IDs: {df['entity_id'].nunique()} / {len(df)}")
        # Length distributions
        df['name_len'] = df['business_name'].fillna('').apply(len)
        df['addr_len'] = df['business_address'].fillna('').apply(len)
        print(f"Name length: min={df['name_len'].min()}, median={df['name_len'].median()}, mean={df['name_len'].mean():.1f}, max={df['name_len'].max()}")
        print(f"Addr length: min={df['addr_len'].min()}, median={df['addr_len'].median()}, mean={df['addr_len'].mean():.1f}, max={df['addr_len'].max()}")
        country_stats[name] = df.set_index('entity_id')['country'].to_dict()

    # Cross-country match check: do S1 and S2/S3 matches have the same country?
    print("\n--- Cross-Country Match Consistency Check ---")
    s1_countries = country_stats['train_source1']
    s2_countries = country_stats['train_source2']
    s3_countries = country_stats['train_source3']
    
    same_country = 0
    diff_country = 0
    missing_country_lookup = 0
    
    for s1_id, lst in zip(gt_df['source1_entity_id'], match_lists):
        c1 = s1_countries.get(s1_id)
        for mid in lst:
            if mid.startswith('S2-'):
                c2 = s2_countries.get(mid)
            else:
                c2 = s3_countries.get(mid)
            if c1 is None or c2 is None:
                missing_country_lookup += 1
            elif c1 == c2:
                same_country += 1
            else:
                diff_country += 1
                
    print(f"True matches same country: {same_country} ({same_country/(same_country+diff_country+1e-9)*100:.4f}%)")
    print(f"True matches diff country: {diff_country}")
    print(f"Missing country lookups: {missing_country_lookup}")

    # Inspect test files
    test_files = {
        'test_source1': 'dataset/test/test_source1.tsv',
        'test_source2': 'dataset/test/test_source2.tsv',
        'test_source3': 'dataset/test/test_source3.tsv',
    }
    print("\n=== TEST DATASET AUDIT ===")
    for name, path in test_files.items():
        print(f"\n--- Analyzing {name} ---")
        t0 = time.time()
        df = pd.read_csv(path, sep="\t", dtype=str)
        print(f"Loaded {name} in {time.time()-t0:.2f}s, shape: {df.shape}")
        print(f"Null counts:\n{df.isnull().sum()}")
        print(f"Country value counts:\n{df['country'].value_counts(dropna=False)}")
        print(f"Unique IDs: {df['entity_id'].nunique()} / {len(df)}")

if __name__ == "__main__":
    audit_sources()
