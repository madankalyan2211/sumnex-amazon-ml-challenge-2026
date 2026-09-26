"""
Ultra-scalable, high-recall inverted index blocking engine.
Combines:
1. Exact base business name
2. Compact base business name (stripped spaces & legal suffixes)
3. Exact clean business name
4. Compact prefix (4-char and 6-char)
5. Distinctive name tokens & bigrams
6. Address number + street/city token pairs
7. Address number + base name token pairs
"""

import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np

from src.normalization import normalize_name_fast, normalize_address_fast

class FastCountryBlocker:
    """
    Inverted index candidate generator for a single country partition.
    """
    def __init__(
        self,
        country: str,
        max_posting_len: int = 1500,
        min_token_len: int = 3,
        max_candidates: int = 75
    ):
        self.country = country
        self.max_posting_len = max_posting_len
        self.min_token_len = min_token_len
        self.max_candidates = max_candidates
        
        self.target_ids: List[str] = []
        self.target_names: List[str] = []
        self.target_addrs: List[str] = []
        
        # Inverted index tables (str key -> list of target integer indices)
        self.idx_base: Dict[str, List[int]] = defaultdict(list)
        self.idx_compact_base: Dict[str, List[int]] = defaultdict(list)
        self.idx_clean: Dict[str, List[int]] = defaultdict(list)
        self.idx_token: Dict[str, List[int]] = defaultdict(list)
        self.idx_name_bigram: Dict[str, List[int]] = defaultdict(list)
        self.idx_addr_num_tok: Dict[str, List[int]] = defaultdict(list)
        self.idx_addr_num_name: Dict[str, List[int]] = defaultdict(list)
        self.idx_compact_pref: Dict[str, List[int]] = defaultdict(list)

    def build_index(
        self,
        target_ids: List[str],
        names: List[str],
        addresses: List[str],
        verbose: bool = True
    ):
        t0 = time.time()
        n = len(target_ids)
        self.target_ids = target_ids
        self.target_names = names
        self.target_addrs = addresses
        
        # Single-pass indexing
        for i in range(n):
            name = names[i]
            addr = addresses[i]
            
            cleaned_n, base_n, compact_n, compact_base_n, tokens_n, base_tokens_n, char_3grams_n = normalize_name_fast(name)
            cleaned_a, tokens_a, token_set_a, numbers_a = normalize_address_fast(addr)
            
            # 1. Base name (cap at 2500)
            if base_n:
                lst = self.idx_base[base_n]
                if len(lst) < 2500:
                    lst.append(i)
                
            # 2. Compact base name (e.g. secunderabadengineers)
            if compact_base_n and compact_base_n != base_n and len(compact_base_n) >= 4:
                lst = self.idx_compact_base[compact_base_n]
                if len(lst) < 2500:
                    lst.append(i)
                    
            # 3. Clean name
            if cleaned_n and cleaned_n != base_n:
                lst = self.idx_clean[cleaned_n]
                if len(lst) < 1500:
                    lst.append(i)
                
            # 4. Compact prefix (4-char and 6-char prefixes)
            if len(compact_base_n) >= 4:
                p4 = compact_base_n[:4]
                lst = self.idx_compact_pref[p4]
                if len(lst) < 800:
                    lst.append(i)
                if len(compact_base_n) >= 6:
                    p6 = compact_base_n[:6]
                    lst = self.idx_compact_pref[p6]
                    if len(lst) < 1000:
                        lst.append(i)
                    
            # 5. Informative base tokens
            for tok in base_tokens_n:
                if len(tok) >= self.min_token_len:
                    lst = self.idx_token[tok]
                    if len(lst) < self.max_posting_len:
                        lst.append(i)
                        
            # 5b. Consecutive name token bigrams
            if len(base_tokens_n) >= 2:
                for b_i in range(len(base_tokens_n) - 1):
                    bg = f"{base_tokens_n[b_i]}_{base_tokens_n[b_i+1]}"
                    lst = self.idx_name_bigram[bg]
                    if len(lst) < self.max_posting_len:
                        lst.append(i)
                        
            # 6. Address number + street/city token
            long_addr_tokens = [t for t in tokens_a if len(t) >= 3 and not t.isdigit()][:4]
            for num in numbers_a:
                for at in long_addr_tokens:
                    key = f"{num}_{at}"
                    lst = self.idx_addr_num_tok[key]
                    if len(lst) < self.max_posting_len:
                        lst.append(i)
                        
            # 7. Address number + Base Name Token (High precision & recall)
            for num in numbers_a:
                for nt in base_tokens_n[:3]:
                    if len(nt) >= 3:
                        key = f"{num}_{nt}"
                        lst = self.idx_addr_num_name[key]
                        if len(lst) < self.max_posting_len:
                            lst.append(i)
                        
        if verbose:
            print(f"[{self.country}] Indexed {n} records in {time.time()-t0:.2f}s.")
            print(f"  Unique bases: {len(self.idx_base)}, Compact bases: {len(self.idx_compact_base)}, Addr pairs: {len(self.idx_addr_num_tok)}, Num-Name pairs: {len(self.idx_addr_num_name)}")

    def query(
        self,
        s1_name_norm: Tuple,
        s1_addr_norm: Tuple
    ) -> List[int]:
        """
        Query candidate target indices for an S1 entity.
        """
        cleaned_n, base_n, compact_n, compact_base_n, tokens_n, base_tokens_n, char_3grams_n = s1_name_norm
        cleaned_a, tokens_a, token_set_a, numbers_a = s1_addr_norm
        
        scores: Dict[int, float] = defaultdict(float)
        
        # 1. Exact base match
        if base_n and base_n in self.idx_base:
            matches = self.idx_base[base_n]
            w = 14.0 / (1.0 + 0.05 * np.log1p(len(matches)))
            for idx in matches:
                scores[idx] += w
                
        # 2. Compact base match
        if compact_base_n and compact_base_n in self.idx_compact_base:
            matches = self.idx_compact_base[compact_base_n]
            w = 13.0 / (1.0 + 0.05 * np.log1p(len(matches)))
            for idx in matches:
                scores[idx] += w
                
        # 3. Exact clean match
        if cleaned_n and cleaned_n in self.idx_clean:
            matches = self.idx_clean[cleaned_n]
            w = 11.0 / (1.0 + 0.05 * np.log1p(len(matches)))
            for idx in matches:
                scores[idx] += w
                
        # 4. Compact prefix match
        if len(compact_base_n) >= 4:
            p4 = compact_base_n[:4]
            if p4 in self.idx_compact_pref:
                matches = self.idx_compact_pref[p4]
                w = 3.5 / (1.0 + np.log1p(len(matches)))
                for idx in matches:
                    scores[idx] += w
            if len(compact_base_n) >= 6:
                p6 = compact_base_n[:6]
                if p6 in self.idx_compact_pref:
                    matches = self.idx_compact_pref[p6]
                    w = 6.0 / (1.0 + np.log1p(len(matches)))
                    for idx in matches:
                        scores[idx] += w
                
        # 5. Informative name tokens
        for tok in base_tokens_n:
            if len(tok) >= self.min_token_len and tok in self.idx_token:
                matches = self.idx_token[tok]
                w = 7.0 / (1.0 + np.log1p(len(matches)))
                for idx in matches:
                    scores[idx] += w
                    
        # 5b. Name token bigrams
        if len(base_tokens_n) >= 2:
            for b_i in range(len(base_tokens_n) - 1):
                bg = f"{base_tokens_n[b_i]}_{base_tokens_n[b_i+1]}"
                if bg in self.idx_name_bigram:
                    matches = self.idx_name_bigram[bg]
                    w = 8.0 / (1.0 + np.log1p(len(matches)))
                    for idx in matches:
                        scores[idx] += w
                        
        # 6. Address number + base name token
        for num in numbers_a:
            for nt in base_tokens_n[:3]:
                if len(nt) >= 3:
                    key = f"{num}_{nt}"
                    if key in self.idx_addr_num_name:
                        matches = self.idx_addr_num_name[key]
                        w = 9.5 / (1.0 + np.log1p(len(matches)))
                        for idx in matches:
                            scores[idx] += w
                            
        # 7. Address number + street/city token
        long_addr_tokens = [t for t in tokens_a if len(t) >= 3 and not t.isdigit()][:4]
        for num in numbers_a:
            for at in long_addr_tokens:
                key = f"{num}_{at}"
                if key in self.idx_addr_num_tok:
                    matches = self.idx_addr_num_tok[key]
                    w = 7.5 / (1.0 + np.log1p(len(matches)))
                    for idx in matches:
                        scores[idx] += w
                        
        if not scores:
            return []
            
        # Top-K candidate extraction
        if len(scores) <= self.max_candidates:
            return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
            
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.max_candidates]
        return [idx for idx, _ in top_items]
