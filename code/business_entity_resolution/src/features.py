"""
Pairwise feature extraction engine for Entity Resolution.
Computes highly informative lexical, token, character-ngram, numeric, and cross-field features.
"""

from typing import Dict, Tuple, Set, List, Any
import numpy as np
from rapidfuzz import fuzz

from src.normalization import GENERIC_ADDR_TOKENS

FEATURE_NAMES = [
    'name_exact_clean',
    'name_exact_base',
    'name_exact_compact',
    'name_exact_compact_base',
    'name_compact_base_contain',
    'name_token_jaccard',
    'name_base_jaccard',
    'name_token_overlap',
    'name_len_ratio',
    'name_char_3gram_jaccard',
    'name_fuzz_ratio',
    'name_fuzz_partial_ratio',
    'name_fuzz_token_sort_ratio',
    'name_fuzz_token_set_ratio',
    'name_fuzz_wratio',
    'name_first_token_match',
    'name_prefix4_match',
    'addr_exact_clean',
    'addr_token_jaccard',
    'addr_token_overlap',
    'addr_rare_token_overlap',
    'addr_num_exact',
    'addr_num_jaccard',
    'addr_num_overlap',
    'addr_num_mismatch',
    'addr_empty_s1',
    'addr_empty_tgt',
    'addr_fuzz_token_set_ratio',
    'addr_fuzz_token_sort_ratio',
    'name_addr_joint_score',
    'high_name_high_addr',
    'exact_name_same_num',
    'num_exact_name_fuzz_high',
    'name_exact_diff_num',
    'token_count_diff',
    'name_lcs_ratio',
]

def jaccard_similarity(set1: Set[Any], set2: Set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def extract_pairwise_features(
    s1_norm: Tuple[Tuple, Tuple],  # (name_norm_tuple, addr_norm_tuple)
    tgt_norm: Tuple[Tuple, Tuple],
) -> List[float]:
    """
    Extract feature vector for an (S1, Target) pair from pre-normalized tuples.
    s1_norm: (
        (cleaned, base, compact, compact_base, tokens, base_tokens, char_3grams),
        (cleaned, tokens, token_set, numbers)
    )
    """
    (s1_name_clean, s1_name_base, s1_name_compact, s1_name_compact_base, s1_tokens, s1_base_tokens, s1_3grams), \
    (s1_addr_clean, s1_addr_tokens, s1_addr_token_set, s1_addr_numbers) = s1_norm

    (tgt_name_clean, tgt_name_base, tgt_name_compact, tgt_name_compact_base, tgt_tokens, tgt_base_tokens, tgt_3grams), \
    (tgt_addr_clean, tgt_addr_tokens, tgt_addr_token_set, tgt_addr_numbers) = tgt_norm

    # 1. Name exact & compact features
    name_exact_clean = 1.0 if s1_name_clean == tgt_name_clean and s1_name_clean else 0.0
    name_exact_base = 1.0 if s1_name_base == tgt_name_base and s1_name_base else 0.0
    name_exact_compact = 1.0 if s1_name_compact == tgt_name_compact and s1_name_compact else 0.0
    name_exact_compact_base = 1.0 if s1_name_compact_base == tgt_name_compact_base and s1_name_compact_base else 0.0
    
    name_compact_base_contain = 0.0
    if s1_name_compact_base and tgt_name_compact_base and min(len(s1_name_compact_base), len(tgt_name_compact_base)) >= 5:
        if s1_name_compact_base in tgt_name_compact_base or tgt_name_compact_base in s1_name_compact_base:
            name_compact_base_contain = 1.0
            
    # 2. Name token features
    s1_tok_set = set(s1_tokens)
    tgt_tok_set = set(tgt_tokens)
    name_token_jaccard = jaccard_similarity(s1_tok_set, tgt_tok_set)
    name_token_overlap = float(len(s1_tok_set & tgt_tok_set))
    
    s1_base_set = set(s1_base_tokens)
    tgt_base_set = set(tgt_base_tokens)
    name_base_jaccard = jaccard_similarity(s1_base_set, tgt_base_set)
    
    # Length ratio & token count diff
    l1 = len(s1_name_clean)
    l2 = len(tgt_name_clean)
    name_len_ratio = (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 1.0
    token_count_diff = float(abs(len(s1_tokens) - len(tgt_tokens)))
    
    # Prefix & first token match
    name_first_token_match = 1.0 if (s1_base_tokens and tgt_base_tokens and s1_base_tokens[0] == tgt_base_tokens[0]) else 0.0
    name_prefix4_match = 1.0 if (len(s1_name_compact) >= 4 and len(tgt_name_compact) >= 4 and s1_name_compact[:4] == tgt_name_compact[:4]) else 0.0
    
    # 3. Name character 3-grams
    name_char_3gram_jaccard = jaccard_similarity(s1_3grams, tgt_3grams)
    
    # 4. RapidFuzz similarities
    if s1_name_clean and tgt_name_clean:
        name_fuzz_ratio = fuzz.ratio(s1_name_clean, tgt_name_clean) / 100.0
        name_fuzz_partial_ratio = fuzz.partial_ratio(s1_name_clean, tgt_name_clean) / 100.0
        name_fuzz_token_sort_ratio = fuzz.token_sort_ratio(s1_name_clean, tgt_name_clean) / 100.0
        name_fuzz_token_set_ratio = fuzz.token_set_ratio(s1_name_clean, tgt_name_clean) / 100.0
        name_fuzz_wratio = fuzz.WRatio(s1_name_clean, tgt_name_clean) / 100.0
    else:
        name_fuzz_ratio = 0.0
        name_fuzz_partial_ratio = 0.0
        name_fuzz_token_sort_ratio = 0.0
        name_fuzz_token_set_ratio = 0.0
        name_fuzz_wratio = 0.0

    # 5. Address exact & token features
    addr_exact_clean = 1.0 if s1_addr_clean == tgt_addr_clean and s1_addr_clean else 0.0
    addr_token_jaccard = jaccard_similarity(s1_addr_token_set, tgt_addr_token_set)
    addr_token_overlap = float(len(s1_addr_token_set & tgt_addr_token_set))
    
    # Rare address token overlap (excluding generic street words)
    s1_rare_addr = s1_addr_token_set - GENERIC_ADDR_TOKENS
    tgt_rare_addr = tgt_addr_token_set - GENERIC_ADDR_TOKENS
    addr_rare_token_overlap = float(len(s1_rare_addr & tgt_rare_addr))
    
    # 6. Address numeric features
    addr_num_exact = 1.0 if s1_addr_numbers == tgt_addr_numbers and s1_addr_numbers else 0.0
    addr_num_jaccard = jaccard_similarity(s1_addr_numbers, tgt_addr_numbers)
    addr_num_overlap = float(len(s1_addr_numbers & tgt_addr_numbers))
    addr_num_mismatch = 1.0 if (s1_addr_numbers and tgt_addr_numbers and not (s1_addr_numbers & tgt_addr_numbers)) else 0.0
    
    addr_empty_s1 = 1.0 if not s1_addr_clean else 0.0
    addr_empty_tgt = 1.0 if not tgt_addr_clean else 0.0
    
    if s1_addr_clean and tgt_addr_clean:
        addr_fuzz_token_set_ratio = fuzz.token_set_ratio(s1_addr_clean, tgt_addr_clean) / 100.0
        addr_fuzz_token_sort_ratio = fuzz.token_sort_ratio(s1_addr_clean, tgt_addr_clean) / 100.0
    else:
        addr_fuzz_token_set_ratio = 0.0
        addr_fuzz_token_sort_ratio = 0.0

    # 7. Cross-field interaction features
    name_addr_joint_score = name_base_jaccard * 0.6 + addr_token_jaccard * 0.4
    high_name_high_addr = 1.0 if (name_base_jaccard >= 0.7 and addr_token_jaccard >= 0.3) else 0.0
    exact_name_same_num = 1.0 if (name_exact_base == 1.0 and addr_num_overlap > 0) else 0.0
    num_exact_name_fuzz_high = 1.0 if (addr_num_overlap > 0 and name_fuzz_token_set_ratio >= 0.85) else 0.0
    name_exact_diff_num = 1.0 if (name_exact_base == 1.0 and addr_num_mismatch == 1.0) else 0.0
    name_lcs_ratio = (name_fuzz_partial_ratio + name_fuzz_ratio) / 2.0

    return [
        name_exact_clean,
        name_exact_base,
        name_exact_compact,
        name_exact_compact_base,
        name_compact_base_contain,
        name_token_jaccard,
        name_base_jaccard,
        name_token_overlap,
        name_len_ratio,
        name_char_3gram_jaccard,
        name_fuzz_ratio,
        name_fuzz_partial_ratio,
        name_fuzz_token_sort_ratio,
        name_fuzz_token_set_ratio,
        name_fuzz_wratio,
        name_first_token_match,
        name_prefix4_match,
        addr_exact_clean,
        addr_token_jaccard,
        addr_token_overlap,
        addr_rare_token_overlap,
        addr_num_exact,
        addr_num_jaccard,
        addr_num_overlap,
        addr_num_mismatch,
        addr_empty_s1,
        addr_empty_tgt,
        addr_fuzz_token_set_ratio,
        addr_fuzz_token_sort_ratio,
        name_addr_joint_score,
        high_name_high_addr,
        exact_name_same_num,
        num_exact_name_fuzz_high,
        name_exact_diff_num,
        token_count_diff,
        name_lcs_ratio,
    ]
