import re
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import time
from src.data_loader import load_ground_truth

# Enhanced Normalization
DOMAIN_SUFFIXES = re.compile(r'\.(com|net|org|biz|co|in|fr|us|io|gov|edu|info|xyz|me|online|store|shop|tech|app|dev)\b', re.IGNORECASE)
NON_ALNUM = re.compile(r'[^a-z0-9\s]')
WHITESPACE = re.compile(r'\s+')
DIGITS = re.compile(r'\d+')

LEGAL_SUFFIXES = {
    'llc', 'inc', 'corp', 'corporation', 'ltd', 'limited', 'co', 'company',
    'pvt ltd', 'private limited', 'gmbh', 'sa', 'sarl', 'sas', 'spa',
    'plc', 'lp', 'llp', 'pllc', 'dba', 'group', 'services', 'enterprises',
    'holdings', 'assoc', 'associates'
}

def clean_text_v2(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    t = text.lower()
    t = DOMAIN_SUFFIXES.sub(' ', t)
    t = NON_ALNUM.sub(' ', t)
    return WHITESPACE.sub(' ', t).strip()

def extract_numbers_v2(text: str) -> set:
    if not isinstance(text, str) or not text:
        return set()
    nums = set()
    for match in DIGITS.finditer(text):
        n = match.group().lstrip('0')
        if n and len(n) <= 8:
            nums.add(n)
    return nums

def get_base_name_v2(clean_n: str) -> str:
    tokens = clean_n.split()
    if not tokens:
        return ""
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    if not tokens:
        return clean_n
    return " ".join(tokens)

class HighRecallBlocker:
    def __init__(self, max_candidates=60):
        self.max_candidates = max_candidates
        self.target_ids = []
        self.target_names = []
        self.target_addrs = []
        
        self.idx_base = defaultdict(list)
        self.idx_clean = defaultdict(list)
        self.idx_token = defaultdict(list)
        self.idx_name_pref = defaultdict(list)
        self.idx_name_bigram = defaultdict(list)
        self.idx_num_tok = defaultdict(list)
        self.idx_num_name = defaultdict(list)
        self.idx_compact_pref = defaultdict(list)

    def build_index(self, target_ids, names, addresses):
        t0 = time.time()
        self.target_ids = target_ids
        self.target_names = names
        self.target_addrs = addresses
        n = len(target_ids)
        
        for i in range(n):
            c_name = clean_text_v2(names[i])
            b_name = get_base_name_v2(c_name)
            c_addr = clean_text_v2(addresses[i])
            
            n_tokens = b_name.split() if b_name else c_name.split()
            a_tokens = [t for t in c_addr.split() if len(t) >= 4 and not t.isdigit()][:4]
            a_nums = extract_numbers_v2(addresses[i])
            compact = c_name.replace(" ", "")
            
            # 1. Base name (cap at 2000)
            if b_name:
                lst = self.idx_base[b_name]
                if len(lst) < 2000:
                    lst.append(i)
                    
            # 2. Clean name
            if c_name and c_name != b_name:
                lst = self.idx_clean[c_name]
                if len(lst) < 1500:
                    lst.append(i)
                    
            # 3. Compact prefixes (first 5 and 7 chars)
            if len(compact) >= 5:
                p5 = compact[:5]
                lst = self.idx_compact_pref[p5]
                if len(lst) < 800:
                    lst.append(i)
                if len(compact) >= 7:
                    p7 = compact[:7]
                    lst = self.idx_compact_pref[p7]
                    if len(lst) < 800:
                        lst.append(i)
                        
            # 4. Name tokens (length >= 3)
            for tok in n_tokens:
                if len(tok) >= 3:
                    lst = self.idx_token[tok]
                    if len(lst) < 800:
                        lst.append(i)
                        
            # 4b. Name bigrams
            for b_i in range(len(n_tokens) - 1):
                bg = f"{n_tokens[b_i]}_{n_tokens[b_i+1]}"
                lst = self.idx_name_bigram[bg]
                if len(lst) < 800:
                    lst.append(i)
                    
            # 5. Address Number + Address Token pairs
            for num in a_nums:
                for at in a_tokens:
                    key = f"{num}_{at}"
                    lst = self.idx_num_tok[key]
                    if len(lst) < 800:
                        lst.append(i)
                        
            # 6. Address Number + Base Name Token pairs (High Precision & Recall!)
            for num in a_nums:
                for nt in n_tokens[:2]:
                    if len(nt) >= 3:
                        key = f"{num}_{nt}"
                        lst = self.idx_num_name[key]
                        if len(lst) < 800:
                            lst.append(i)
                            
        print(f"Indexed {n} targets in {time.time()-t0:.2f}s.")

    def query(self, s1_name, s1_addr):
        c_name = clean_text_v2(s1_name)
        b_name = get_base_name_v2(c_name)
        c_addr = clean_text_v2(s1_addr)
        
        n_tokens = b_name.split() if b_name else c_name.split()
        a_tokens = [t for t in c_addr.split() if len(t) >= 4 and not t.isdigit()][:4]
        a_nums = extract_numbers_v2(s1_addr)
        compact = c_name.replace(" ", "")
        
        scores = defaultdict(float)
        
        # 1. Base name
        if b_name and b_name in self.idx_base:
            matches = self.idx_base[b_name]
            w = 14.0 / (1.0 + 0.1 * np.log1p(len(matches)))
            for idx in matches:
                scores[idx] += w
                
        # 2. Clean name
        if c_name and c_name in self.idx_clean:
            matches = self.idx_clean[c_name]
            w = 12.0 / (1.0 + 0.1 * np.log1p(len(matches)))
            for idx in matches:
                scores[idx] += w
                
        # 3. Compact prefix
        if len(compact) >= 5:
            p5 = compact[:5]
            if p5 in self.idx_compact_pref:
                matches = self.idx_compact_pref[p5]
                w = 5.0 / (1.0 + np.log1p(len(matches)))
                for idx in matches:
                    scores[idx] += w
            if len(compact) >= 7:
                p7 = compact[:7]
                if p7 in self.idx_compact_pref:
                    matches = self.idx_compact_pref[p7]
                    w = 6.0 / (1.0 + np.log1p(len(matches)))
                    for idx in matches:
                        scores[idx] += w
                        
        # 4. Name tokens
        for tok in n_tokens:
            if len(tok) >= 3 and tok in self.idx_token:
                matches = self.idx_token[tok]
                w = 7.0 / (1.0 + np.log1p(len(matches)))
                for idx in matches:
                    scores[idx] += w
                    
        # 4b. Name bigrams
        for b_i in range(len(n_tokens) - 1):
            bg = f"{n_tokens[b_i]}_{n_tokens[b_i+1]}"
            if bg in self.idx_name_bigram:
                matches = self.idx_name_bigram[bg]
                w = 8.0 / (1.0 + np.log1p(len(matches)))
                for idx in matches:
                    scores[idx] += w
                    
        # 5. Address Number + Address Token
        for num in a_nums:
            for at in a_tokens:
                key = f"{num}_{at}"
                if key in self.idx_num_tok:
                    matches = self.idx_num_tok[key]
                    w = 6.0 / (1.0 + np.log1p(len(matches)))
                    for idx in matches:
                        scores[idx] += w
                        
        # 6. Address Number + Name Token
        for num in a_nums:
            for nt in n_tokens[:2]:
                if len(nt) >= 3:
                    key = f"{num}_{nt}"
                    if key in self.idx_num_name:
                        matches = self.idx_num_name[key]
                        w = 9.0 / (1.0 + np.log1p(len(matches)))
                        for idx in matches:
                            scores[idx] += w
                            
        if not scores:
            return []
            
        if len(scores) <= self.max_candidates:
            return list(scores.keys())
            
        return sorted(scores, key=scores.get, reverse=True)[:self.max_candidates]

def test_high_recall():
    print("=== TESTING HIGH RECALL BLOCKER ===")
    gt = load_ground_truth("dataset/train/train_ground_truth.tsv")
    s1_df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
    s2_df = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
    s3_df = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
    
    s1_us = s1_df[s1_df['country'] == 'US'].sample(n=3000, random_state=42)
    s2_us = s2_df[s2_df['country'] == 'US']
    s3_us = s3_df[s3_df['country'] == 'US']
    
    tgt_df = pd.concat([s2_us, s3_us], ignore_index=True)
    target_ids = list(tgt_df['entity_id'])
    target_names = list(tgt_df['business_name'])
    target_addrs = list(tgt_df['business_address'])
    
    blocker = HighRecallBlocker(max_candidates=60)
    blocker.build_index(target_ids, target_names, target_addrs)
    
    target_ids_set = set(target_ids)
    s1_names = list(s1_us['business_name'])
    s1_addrs = list(s1_us['business_address'])
    s1_ids = list(s1_us['entity_id'])
    
    total_true = 0
    recalled = 0
    total_candidates = 0
    
    t0 = time.time()
    for i in range(len(s1_us)):
        cand_indices = blocker.query(s1_names[i], s1_addrs[i])
        cand_ids = {blocker.target_ids[idx] for idx in cand_indices}
        total_candidates += len(cand_ids)
        
        true_tgts = gt.get(s1_ids[i], set()) & target_ids_set
        total_true += len(true_tgts)
        recalled += len(true_tgts & cand_ids)
        
    dt = time.time() - t0
    print(f"\n--- RESULTS WITH HIGH RECALL BLOCKER ---")
    print(f"Total True Links: {total_true}")
    print(f"Recalled Links  : {recalled} ({recalled/total_true*100:.2f}%)  [Previous Baseline was 85.12%]")
    print(f"Missed Links    : {total_true - recalled} ({(total_true-recalled)/total_true*100:.2f}%)")
    print(f"Avg Candidates/S1: {total_candidates/len(s1_us):.1f}")
    print(f"Query Speed     : {len(s1_us)/dt:.1f} queries/sec")

if __name__ == "__main__":
    test_high_recall()
