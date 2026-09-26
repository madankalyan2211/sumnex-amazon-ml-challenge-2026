"""
Threshold optimization and decision policy engine.
Optimizes primary and sibling decision thresholds on validation set to maximize Macro F_0.5.
"""

from typing import Dict, Set, List, Tuple, Any, Optional
import numpy as np

from src.evaluation import evaluate_predictions

def sweep_thresholds(
    ground_truth: Dict[str, Set[str]],
    candidate_predictions: Dict[str, List[Tuple[str, float]]],
    threshold_min: float = 0.30,
    threshold_max: float = 0.95,
    step: float = 0.02,
    verbose: bool = True
) -> Tuple[float, float, List[Dict]]:
    """
    Search for the optimal decision threshold that maximizes Macro F_0.5.
    """
    thresholds = np.arange(threshold_min, threshold_max + 1e-5, step)
    results = []
    
    best_threshold = 0.80
    best_macro_f05 = -1.0
    
    for th in thresholds:
        preds = {}
        for s1_id, cand_list in candidate_predictions.items():
            matched = {tid for tid, prob in cand_list if prob >= th}
            preds[s1_id] = matched
            
        metrics = evaluate_predictions(ground_truth, preds)
        macro_f05 = metrics['macro_f05']
        
        metrics['threshold'] = round(float(th), 3)
        results.append(metrics)
        
        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_threshold = round(float(th), 3)
            
    if verbose:
        print("\n--- Threshold Optimization Sweep Results ---")
        print(f"Tested {len(thresholds)} thresholds from {threshold_min:.2f} to {threshold_max:.2f}")
        print(f"Optimal Threshold: {best_threshold:.3f} -> Best Macro F_0.5: {best_macro_f05:.4f}")
        sorted_res = sorted(results, key=lambda x: x['macro_f05'], reverse=True)[:5]
        for r in sorted_res:
            print(f"  Th: {r['threshold']:.3f} | Macro F0.5: {r['macro_f05']:.4f} | Micro P: {r['micro_precision']:.4f} | Micro R: {r['micro_recall']:.4f}")

    return best_threshold, best_macro_f05, results

def sweep_dynamic_sibling_thresholds(
    ground_truth: Dict[str, Set[str]],
    val_items: List[Dict[str, Any]],
    primary_th_range: List[float] = [0.76, 0.78, 0.80, 0.82, 0.84, 0.86],
    sibling_th_range: List[float] = [0.50, 0.55, 0.58, 0.60, 0.65],
    verbose: bool = True
) -> Tuple[float, float, float, Dict]:
    """
    Joint 2D grid search over Primary Threshold (anchor confirmation) and Sibling Threshold.
    """
    best_primary_th = 0.82
    best_sibling_th = 0.58
    best_macro_f05 = -1.0
    best_metrics = {}
    
    for p_th in primary_th_range:
        for s_th in sibling_th_range:
            preds = {}
            for item in val_items:
                s1_id = item['s1_id']
                cand_tids = item['cand_tids']
                probs = item['probs']
                cand_bases = item['cand_bases']
                cand_compact_bases = item['cand_compact_bases']
                cand_nums = item['cand_nums']
                
                primary_indices = [i for i in range(len(cand_tids)) if probs[i] >= p_th]
                if primary_indices:
                    anchor_bases = {cand_bases[i] for i in primary_indices if cand_bases[i]}
                    anchor_cbases = {cand_compact_bases[i] for i in primary_indices if cand_compact_bases[i]}
                    anchor_nums = set()
                    for i in primary_indices:
                        anchor_nums.update(cand_nums[i])
                        
                    matched = set(cand_tids[i] for i in primary_indices)
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
                                matched.add(cand_tids[j])
                    preds[s1_id] = matched
                else:
                    preds[s1_id] = {cand_tids[i] for i in range(len(cand_tids)) if probs[i] >= (p_th + 0.04)}
                    
            metrics = evaluate_predictions(ground_truth, preds)
            f05 = metrics['macro_f05']
            if f05 > best_macro_f05:
                best_macro_f05 = f05
                best_primary_th = p_th
                best_sibling_th = s_th
                best_metrics = metrics
                
    if verbose:
        print(f"\nOptimal Dynamic Settings -> Primary Th: {best_primary_th:.2f} | Sibling Th: {best_sibling_th:.2f}")
        print(f"  Macro F0.5: {best_macro_f05:.4f} | Micro Precision: {best_metrics.get('micro_precision', 0):.4f} | Micro Recall: {best_metrics.get('micro_recall', 0):.4f} | Singleton Acc: {best_metrics.get('singleton_accuracy', 0):.4f}")
        
    return best_primary_th, best_sibling_th, best_macro_f05, best_metrics
