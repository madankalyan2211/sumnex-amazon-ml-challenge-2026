import pandas as pd
import numpy as np
import time
from collections import defaultdict
from src.data_loader import load_ground_truth, load_source_tsv
from src.normalization import normalize_name_fast, normalize_address_fast
from src.features import extract_pairwise_features
from src.blocking import FastCountryBlocker
from src.matcher import EntityMatcher
from src.evaluation import evaluate_predictions

def evaluate_champion_strategies():
    print("=== EVALUATING CHAMPION STRATEGIES ON VALIDATION ===")
    
    # 1. Load data
    gt = load_ground_truth("dataset/train/train_ground_truth.tsv")
    s1_df = load_source_tsv("dataset/train/train_source1.tsv")
    s2_df = load_source_tsv("dataset/train/train_source2.tsv")
    s3_df = load_source_tsv("dataset/train/train_source3.tsv")
    
    # Load 15,000 validation sample across US and India
    val_s1 = s1_df.sample(n=15000, random_state=42)
    val_gt = {s1_id: gt.get(s1_id, set()) for s1_id in val_s1['entity_id']}
    
    # Build blockers on training targets
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    blockers = {}
    for country, grp in targets_df.groupby('country'):
        b = FastCountryBlocker(country=country, max_posting_len=800, max_candidates=60)
        b.build_index(list(grp['entity_id']), list(grp['business_name']), list(grp['business_address']), verbose=False)
        blockers[country] = b
        
    matcher = EntityMatcher()
    matcher.load("experiments/entity_matcher.pkl")
    
    print("\n--- Scoring Validation Candidates ---")
    s1_records = val_s1.to_dict('records')
    
    val_cand_data = {}
    
    for s1_rec in s1_records:
        s1_id = s1_rec['entity_id']
        country = s1_rec['country']
        blocker = blockers.get(country)
        if not blocker:
            continue
            
        s1_n = normalize_name_fast(s1_rec['business_name'])
        s1_a = normalize_address_fast(s1_rec['business_address'])
        
        cand_indices = blocker.query(s1_n, s1_a)
        if not cand_indices:
            val_cand_data[s1_id] = []
            continue
            
        feats = []
        cand_tids = []
        cand_n_list = []
        cand_a_list = []
        
        for idx in cand_indices:
            tid = blocker.target_ids[idx]
            t_n = normalize_name_fast(blocker.target_names[idx])
            t_a = normalize_address_fast(blocker.target_addrs[idx])
            
            f = extract_pairwise_features((s1_n, s1_a), (t_n, t_a))
            feats.append(f)
            cand_tids.append(tid)
            cand_n_list.append(t_n)
            cand_a_list.append(t_a)
            
        probs = matcher.predict_proba(np.array(feats, dtype=np.float32))
        
        cands_with_info = []
        for i in range(len(cand_tids)):
            cands_with_info.append({
                'tid': cand_tids[i],
                'prob': float(probs[i]),
                'n_norm': cand_n_list[i],
                'a_norm': cand_a_list[i]
            })
        val_cand_data[s1_id] = cands_with_info
        
    print(f"Scored {len(val_cand_data)} validation entities.")
    
    # 1. Baseline Fixed Threshold (tau = 0.820)
    print("\n[Strategy 1] Standard Global Threshold (tau = 0.820):")
    preds_v1 = {}
    for s1_id, cands in val_cand_data.items():
        preds_v1[s1_id] = set(c['tid'] for c in cands if c['prob'] >= 0.820)
    m1 = evaluate_predictions(val_gt, preds_v1)
    mean_matches1 = np.mean([len(v) for v in preds_v1.values()])
    print(f"  Macro F0.5 = {m1['macro_f05']:.4f} | Precision = {m1['micro_precision']:.4f} | Recall = {m1['micro_recall']:.4f} | Mean Matches = {mean_matches1:.2f}")
    
    # 2. Strategy 2: Dynamic Cluster-Aware Secondary Thresholding
    print("\n[Strategy 2] Cluster-Aware Dynamic Evidence Thresholding:")
    for th_primary in [0.80, 0.82, 0.85]:
        for th_secondary in [0.45, 0.50, 0.55, 0.60, 0.65]:
            preds_dyn = {}
            for s1_id, cands in val_cand_data.items():
                primary_matches = [c for c in cands if c['prob'] >= th_primary]
                if primary_matches:
                    anchor_bases = {c['n_norm'][1] for c in primary_matches if c['n_norm'][1]}
                    anchor_nums = set()
                    for c in primary_matches:
                        anchor_nums.update(c['a_norm'][3])
                        
                    matched = set(c['tid'] for c in primary_matches)
                    for c in cands:
                        if c['tid'] in matched:
                            continue
                        if c['prob'] >= th_secondary:
                            cand_base = c['n_norm'][1]
                            cand_nums = c['a_norm'][3]
                            if (cand_base and cand_base in anchor_bases) or (cand_nums and bool(cand_nums & anchor_nums)):
                                matched.add(c['tid'])
                    preds_dyn[s1_id] = matched
                else:
                    preds_dyn[s1_id] = set(c['tid'] for c in cands if c['prob'] >= 0.86)
                    
            m = evaluate_predictions(val_gt, preds_dyn)
            mean_m = np.mean([len(v) for v in preds_dyn.values()])
            print(f"  Primary Th={th_primary:.2f}, Sec Th={th_secondary:.2f} -> Macro F0.5 = {m['macro_f05']:.4f} | Precision = {m['micro_precision']:.4f} | Recall = {m['micro_recall']:.4f} | Mean Matches = {mean_m:.2f}")

if __name__ == "__main__":
    evaluate_champion_strategies()
