"""
Evaluation metrics for Entity Resolution.
Computes exact Macro F_0.5 across all Source 1 entities including singletons,
plus component metrics: precision, recall, singleton accuracy, multi-match performance.
"""

from typing import Dict, Set, List, Tuple
import numpy as np

def compute_entity_f05(true_set: Set[str], pred_set: Set[str]) -> float:
    """
    Compute F_0.5 score for a single Source 1 entity.
    - True empty, Pred empty -> 1.0 (Correct singleton)
    - True empty, Pred non-empty -> 0.0 (False positive merge)
    - True non-empty, Pred empty -> 0.0 (Missed match)
    - True non-empty, Pred non-empty:
        Precision = |True & Pred| / |Pred|
        Recall = |True & Pred| / |True|
        F_0.5 = (1.25 * P * R) / (0.25 * P + R) if P, R > 0 else 0.0
    """
    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0
    
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
        
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom

def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> Dict[str, float]:
    """
    Evaluate predictions against ground truth for all S1 entities in ground_truth.
    Returns:
        macro_f05: Overall Macro F_0.5 score (official challenge metric)
        precision_micro: Global micro precision over all predicted links
        recall_micro: Global micro recall over all true links
        singleton_accuracy: Accuracy on true singleton entities (T = empty)
        matched_macro_f05: Macro F_0.5 strictly on non-singleton entities
        candidate_coverage / recall
    """
    s1_ids = list(ground_truth.keys())
    total_entities = len(s1_ids)
    
    if total_entities == 0:
        return {"macro_f05": 0.0}
        
    scores = []
    singleton_scores = []
    non_singleton_scores = []
    
    total_tp = 0
    total_pred = 0
    total_true = 0
    
    for s1_id in s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())
        
        f05 = compute_entity_f05(true_set, pred_set)
        scores.append(f05)
        
        if not true_set:
            singleton_scores.append(f05)
        else:
            non_singleton_scores.append(f05)
            
        tp = len(true_set & pred_set)
        total_tp += tp
        total_pred += len(pred_set)
        total_true += len(true_set)
        
    micro_prec = total_tp / total_pred if total_pred > 0 else 1.0 if total_true == 0 else 0.0
    micro_rec = total_tp / total_true if total_true > 0 else 1.0
    micro_f05 = (1.25 * micro_prec * micro_rec) / (0.25 * micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    
    return {
        "macro_f05": float(np.mean(scores)),
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f05": micro_f05,
        "singleton_accuracy": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        "non_singleton_macro_f05": float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
        "total_s1": total_entities,
        "total_singletons": len(singleton_scores),
        "total_non_singletons": len(non_singleton_scores),
        "total_true_pairs": total_true,
        "total_pred_pairs": total_pred,
        "total_tp_pairs": total_tp,
    }
