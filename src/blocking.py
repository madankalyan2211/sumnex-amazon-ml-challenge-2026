"""
High-Recall Multi-Channel BM25 Inverted Index Blocking Engine.
Combines:
1. Exact base business name & cleaned name
2. Recursively cleaned compact base name
3. Compact prefix (4-char and 6-char)
4. BM25 / IDF-weighted distinctive name tokens & bigrams
5. Address number + street/city token pairs with IDF
6. Address number + base name token pairs with IDF
7. Rare distinctive address tokens for numberless entities
"""

import math
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np

from src.normalization import normalize_name_fast, normalize_address_fast, GENERIC_ADDR_TOKENS

class FastCountryBlocker:
    """
    Inverted index candidate generator for a single country partition.
    """
    def __init__(
        self,
        country: str,
        max_posting_len: int = 2500,
        min_token_len: int = 3,
        max_candidates: int = 90
    ):
        self.country = country
        self.max_posting_len = max_posting_len
        self.min_token_len = min_token_len
        self.max_candidates = max_candidates
        self.total_targets = 0
        
        self.target_ids: List[str] = []
        self.target_names: List[str] = []
        self.target_addrs: List[str] = []
        
        # Inverted index tables (str key -> list of target integer indices)
        self.idx_base: Dict[str, List[int]] = defaultdict(list)
        self.idx_compact_base: Dict[str, List[int]] = defaultdict(list)
        self.idx_clean: Dict[str, List[int]] = defaultdict(list)
        self.idx_token: Dict[str, List[int]] = defaultdict(list)
        self.idx_name_bigram: Dict[str, List[int]] = defaultdict(list)
        self.idx_num_tok: Dict[str, List[int]] = defaultdict(list)
        self.idx_num_name: Dict[str, List[int]] = defaultdict(list)
        self.idx_compact_pref: Dict[str, List[int]] = defaultdict(list)
        self.idx_rare_addr_tok: Dict[str, List[int]] = defaultdict(list)

    def build_index(
        self,
        target_ids: List[str],
        names: List[str],
        addresses: List[str],
        verbose: bool = True
    ):
        t0 = time.time()
        n = len(target_ids)
        self.total_targets = n
        self.target_ids = target_ids
        self.target_names = names
        self.target_addrs = addresses
        
        # Single-pass indexing
        for i in range(n):
            name = names[i]
            addr = addresses[i]
            
            cleaned_n, base_n, compact_n, compact_base_n, tokens_n, base_tokens_n, char_3grams_n = normalize_name_fast(name)
            cleaned_a, tokens_a, token_set_a, numbers_a = normalize_address_fast(addr)
            
            # 1. Base name (cap at 3500)
            if base_n:
                lst = self.idx_base[base_n]
                if len(lst) < 3500:
                    lst.append(i)
                
            # 2. Compact base name (e.g. secunderabadengineers, cpmeduserve)
            if compact_base_n and compact_base_n != base_n and len(compact_base_n) >= 4:
                lst = self.idx_compact_base[compact_base_n]
                if len(lst) < 3500:
                    lst.append(i)
                    
            # 3. Clean name
            if cleaned_n and cleaned_n != base_n:
                lst = self.idx_clean[cleaned_n]
                if len(lst) < 2000:
                    lst.append(i)
                
            # 4. Compact prefix (4-char and 6-char prefixes)
            if len(compact_base_n) >= 4:
                p4 = compact_base_n[:4]
                lst = self.idx_compact_pref[p4]
                if len(lst) < 1200:
                    lst.append(i)
                if len(compact_base_n) >= 6:
                    p6 = compact_base_n[:6]
                    lst = self.idx_compact_pref[p6]
                    if len(lst) < 1500:
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
                        
            # 6. Address number + street/city token (first 3 numbers & first 3 tokens)
            long_addr_tokens = [t for t in tokens_a if len(t) >= 3 and not t.isdigit()][:3]
            for num in list(numbers_a)[:3]:
                for at in long_addr_tokens:
                    key = f"{num}_{at}"
                    lst = self.idx_num_tok[key]
                    if len(lst) < self.max_posting_len:
                        lst.append(i)
                        
            # 7. Address number + Base Name Token
            for num in list(numbers_a)[:3]:
                for nt in base_tokens_n[:2]:
                    if len(nt) >= 3:
                        key = f"{num}_{nt}"
                        lst = self.idx_num_name[key]
                        if len(lst) < self.max_posting_len:
                            lst.append(i)
                            
            # 8. Rare address tokens (for numberless addresses)
            for at in tokens_a:
                if len(at) >= 4 and not at.isdigit() and at not in GENERIC_ADDR_TOKENS:
                    lst = self.idx_rare_addr_tok[at]
                    if len(lst) < 2000:
                        lst.append(i)
                        
        if verbose:
            print(f"[{self.country}] Indexed {n} records in {time.time()-t0:.2f}s.")
            print(f"  Bases: {len(self.idx_base)}, Compact bases: {len(self.idx_compact_base)}, Addr pairs: {len(self.idx_num_tok)}, Num-Name: {len(self.idx_num_name)}, Rare addr: {len(self.idx_rare_addr_tok)}")

    def query(
        self,
        s1_name_norm: Tuple,
        s1_addr_norm: Tuple
    ) -> List[int]:
        """
        Query candidate target indices for an S1 entity using Multi-Channel Quotas & BM25 IDF weighting.
        """
        cleaned_n, base_n, compact_n, compact_base_n, tokens_n, base_tokens_n, char_3grams_n = s1_name_norm
        cleaned_a, tokens_a, token_set_a, numbers_a = s1_addr_norm
        n_total = self.total_targets
        
        cands_set = set()
        
        # Channel 1: Exact Base & Compact Base & Clean (Top priority, up to 35)
        ch1_cands = []
        if base_n and base_n in self.idx_base:
            ch1_cands.extend(self.idx_base[base_n][:35])
        if compact_base_n and compact_base_n in self.idx_compact_base:
            ch1_cands.extend(self.idx_compact_base[compact_base_n][:35])
        if cleaned_n and cleaned_n in self.idx_clean:
            ch1_cands.extend(self.idx_clean[cleaned_n][:25])
        cands_set.update(ch1_cands[:35])
        
        # Channel 2: Cross-Field Address Num + Base Name Token (up to 35)
        ch2_scores: Dict[int, float] = defaultdict(float)
        for num in list(numbers_a)[:3]:
            for nt in base_tokens_n[:3]:
                if len(nt) >= 3:
                    k = f"{num}_{nt}"
                    if k in self.idx_num_name:
                        post = self.idx_num_name[k]
                        idf = math.log((n_total - len(post) + 0.5) / (len(post) + 0.5) + 1.0)
                        for idx in post[:60]:
                            ch2_scores[idx] += idf
        if ch2_scores:
            top_ch2 = sorted(ch2_scores.keys(), key=lambda k: ch2_scores[k], reverse=True)[:35]
            cands_set.update(top_ch2)
            
        # Channel 3: Address Num + Street/City Token (up to 35)
        ch3_scores: Dict[int, float] = defaultdict(float)
        long_addr_tokens = [t for t in tokens_a if len(t) >= 3 and not t.isdigit()][:4]
        for num in list(numbers_a)[:3]:
            for at in long_addr_tokens:
                k = f"{num}_{at}"
                if k in self.idx_num_tok:
                    post = self.idx_num_tok[k]
                    idf = math.log((n_total - len(post) + 0.5) / (len(post) + 0.5) + 1.0)
                    for idx in post[:60]:
                        ch3_scores[idx] += idf
        if ch3_scores:
            top_ch3 = sorted(ch3_scores.keys(), key=lambda k: ch3_scores[k], reverse=True)[:35]
            cands_set.update(top_ch3)
            
        # Channel 4: Rare Base Tokens (up to 40)
        ch4_scores: Dict[int, float] = defaultdict(float)
        for tok in base_tokens_n:
            if len(tok) >= self.min_token_len and tok in self.idx_token:
                post = self.idx_token[tok]
                if len(post) < 10000:
                    idf = math.log((n_total - len(post) + 0.5) / (len(post) + 0.5) + 1.0)
                    for idx in post[:100]:
                        ch4_scores[idx] += idf
        if ch4_scores:
            top_ch4 = sorted(ch4_scores.keys(), key=lambda k: ch4_scores[k], reverse=True)[:40]
            cands_set.update(top_ch4)
            
        # Channel 5: Rare Address Tokens (for numberless entities or cross-validation, up to 30)
        ch5_scores: Dict[int, float] = defaultdict(float)
        for at in tokens_a:
            if len(at) >= 4 and not at.isdigit() and at not in GENERIC_ADDR_TOKENS and at in self.idx_rare_addr_tok:
                post = self.idx_rare_addr_tok[at]
                if len(post) < 2000:
                    idf = math.log((n_total - len(post) + 0.5) / (len(post) + 0.5) + 1.0)
                    for idx in post[:50]:
                        ch5_scores[idx] += idf
        if ch5_scores:
            top_ch5 = sorted(ch5_scores.keys(), key=lambda k: ch5_scores[k], reverse=True)[:30]
            cands_set.update(top_ch5)
            
        # Channel 6: Compact Prefix (4 & 6 char) and Name Bigrams (up to 30)
        ch6_scores: Dict[int, float] = defaultdict(float)
        if len(compact_base_n) >= 4:
            p4 = compact_base_n[:4]
            if p4 in self.idx_compact_pref:
                post = self.idx_compact_pref[p4]
                idf = math.log((n_total - len(post) + 0.5) / (len(post) + 0.5) + 1.0)
                for idx in post[:50]:
                    ch6_scores[idx] += idf * 0.5
            if len(compact_base_n) >= 6:
                p6 = compact_base_n[:6]
                if p6 in self.idx_compact_pref:
                    post = self.idx_compact_pref[p6]
                    idf = math.log((n_total - len(post) + 0.5) / (len(post) + 0.5) + 1.0)
                    for idx in post[:60]:
                        ch6_scores[idx] += idf * 0.8
        if len(base_tokens_n) >= 2:
            for b_i in range(len(base_tokens_n) - 1):
                bg = f"{base_tokens_n[b_i]}_{base_tokens_n[b_i+1]}"
                if bg in self.idx_name_bigram:
                    post = self.idx_name_bigram[bg]
                    idf = math.log((n_total - len(post) + 0.5) / (len(post) + 0.5) + 1.0)
                    for idx in post[:60]:
                        ch6_scores[idx] += idf
        if ch6_scores:
            top_ch6 = sorted(ch6_scores.keys(), key=lambda k: ch6_scores[k], reverse=True)[:30]
            cands_set.update(top_ch6)
            
        if not cands_set:
            return []
            
        if len(cands_set) <= self.max_candidates:
            return list(cands_set)
            
        return list(cands_set)[:self.max_candidates]
