import pandas as pd
import numpy as np
import re
import unicodedata
from collections import Counter

def run_deep_dive():
    print("=== DEEP DIVE: SAMPLE PAIR INVESTIGATION ===")
    
    # Load ground truth
    gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str)
    gt_df['matched_entity_ids'] = gt_df['matched_entity_ids'].fillna('')
    
    # Filter to S1s that have matches
    matched_gt = gt_df[gt_df['matched_entity_ids'] != ''].copy()
    
    # Sample 50k S1s for fast statistics
    sample_gt = matched_gt.sample(n=50000, random_state=42)
    
    # Load S1
    s1_df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str).set_index('entity_id')
    
    # Expand positive pairs
    pairs = []
    for s1_id, m_str in zip(sample_gt['source1_entity_id'], sample_gt['matched_entity_ids']):
        m_list = [m.strip() for m in m_str.split(',') if m.strip()]
        for mid in m_list:
            pairs.append((s1_id, mid))
            
    print(f"Total positive pairs in sample: {len(pairs)}")
    pair_df = pd.DataFrame(pairs, columns=['s1_id', 'target_id'])
    
    # Separate S2 and S3 targets
    s2_ids = set(pair_df[pair_df['target_id'].str.startswith('S2-')]['target_id'])
    s3_ids = set(pair_df[pair_df['target_id'].str.startswith('S3-')]['target_id'])
    
    # Load required rows from S2 and S3
    print("Loading S2...")
    s2_df = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str)
    s2_map = s2_df[s2_df['entity_id'].isin(s2_ids)].set_index('entity_id').to_dict('index')
    del s2_df
    
    print("Loading S3...")
    s3_df = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str)
    s3_map = s3_df[s3_df['entity_id'].isin(s3_ids)].set_index('entity_id').to_dict('index')
    del s3_df
    
    print("Analyzing positive pairs...")
    exact_raw_name = 0
    exact_lower_name = 0
    exact_clean_name = 0
    token_jaccard_gt_50 = 0
    token_jaccard_gt_80 = 0
    has_common_number = 0
    has_numbers_in_both = 0
    
    # Normalization helper
    def clean_str(s):
        if not s or pd.isna(s):
            return ""
        # NFKD normalize
        s = unicodedata.normalize('NFKD', str(s))
        s = s.lower()
        s = re.sub(r'[^\w\s]', ' ', s)
        return " ".join(s.split())
    
    samples_to_print = []
    
    for _, row in pair_df.iterrows():
        s1_id = row['s1_id']
        mid = row['target_id']
        
        s1_rec = s1_df.loc[s1_id]
        tgt_rec = s2_map.get(mid) if mid.startswith('S2-') else s3_map.get(mid)
        if not tgt_rec:
            continue
            
        s1_name = str(s1_rec['business_name']) if pd.notna(s1_rec['business_name']) else ""
        s1_addr = str(s1_rec['business_address']) if pd.notna(s1_rec['business_address']) else ""
        tgt_name = str(tgt_rec['business_name']) if pd.notna(tgt_rec['business_name']) else ""
        tgt_addr = str(tgt_rec['business_address']) if pd.notna(tgt_rec['business_address']) else ""
        country = s1_rec['country']
        
        if s1_name == tgt_name:
            exact_raw_name += 1
        if s1_name.lower() == tgt_name.lower():
            exact_lower_name += 1
            
        c1 = clean_str(s1_name)
        c2 = clean_str(tgt_name)
        if c1 == c2:
            exact_clean_name += 1
            
        t1 = set(c1.split())
        t2 = set(c2.split())
        jacc = len(t1 & t2) / len(t1 | t2) if (t1 or t2) else 0.0
        if jacc >= 0.5:
            token_jaccard_gt_50 += 1
        if jacc >= 0.8:
            token_jaccard_gt_80 += 1
            
        # Numbers in address
        n1 = set(re.findall(r'\b\d+\b', s1_addr))
        n2 = set(re.findall(r'\b\d+\b', tgt_addr))
        if n1 and n2:
            has_numbers_in_both += 1
            if n1 & n2:
                has_common_number += 1
                
        if len(samples_to_print) < 25:
            samples_to_print.append({
                's1_id': s1_id,
                'target_id': mid,
                'country': country,
                's1_name': s1_name,
                'tgt_name': tgt_name,
                's1_addr': s1_addr,
                'tgt_addr': tgt_addr,
                'name_jacc': round(jacc, 2)
            })
            
    total = len(pair_df)
    print(f"\n--- Positive Pair Statistics (out of {total} true matches) ---")
    print(f"Exact raw name match: {exact_raw_name} ({exact_raw_name/total*100:.2f}%)")
    print(f"Exact lowercase name match: {exact_lower_name} ({exact_lower_name/total*100:.2f}%)")
    print(f"Exact cleaned name match: {exact_clean_name} ({exact_clean_name/total*100:.2f}%)")
    print(f"Token Jaccard >= 0.5: {token_jaccard_gt_50} ({token_jaccard_gt_50/total*100:.2f}%)")
    print(f"Token Jaccard >= 0.8: {token_jaccard_gt_80} ({token_jaccard_gt_80/total*100:.2f}%)")
    print(f"Addresses with numbers in both: {has_numbers_in_both} ({has_numbers_in_both/total*100:.2f}%)")
    print(f"Shared address numbers when both have numbers: {has_common_number} ({has_common_number/(has_numbers_in_both+1e-9)*100:.2f}%)")

    print("\n--- 20 Sample True Matches ---")
    for i, s in enumerate(samples_to_print[:20], 1):
        print(f"\n[{i}] Country: {s['country']} | Jaccard: {s['name_jacc']}")
        print(f"  S1 ({s['s1_id']}): {s['s1_name']} || {s['s1_addr']}")
        print(f"  Tgt ({s['target_id']}): {s['tgt_name']} || {s['tgt_addr']}")

if __name__ == "__main__":
    run_deep_dive()
