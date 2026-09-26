"""
Data loader and validation split manager.
Provides memory-efficient chunked/streamed loading and reproducible train/val splits.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Set, Tuple, List, Optional

def load_ground_truth(path: str) -> Dict[str, Set[str]]:
    """Load ground truth mapping {s1_id: set(matched_s2_s3_ids)}."""
    gt = {}
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t')
            s1_id = parts[0]
            if len(parts) > 1 and parts[1].strip():
                gt[s1_id] = set(m.strip() for m in parts[1].split(',') if m.strip())
            else:
                gt[s1_id] = set()
    return gt

def load_source_tsv(path: str) -> pd.DataFrame:
    """Load a source TSV file."""
    return pd.read_csv(
        path,
        sep='\t',
        dtype={'entity_id': str, 'business_name': str, 'business_address': str, 'country': str},
        keep_default_na=False
    )

def create_validation_split(
    s1_df: pd.DataFrame,
    val_size: float = 0.05,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split Source 1 entities into train and validation sets,
    stratified by country to ensure balanced representation.
    """
    np.random.seed(random_state)
    val_indices = []
    train_indices = []
    
    for country, group in s1_df.groupby('country'):
        n_val = int(len(group) * val_size)
        shuffled = group.sample(frac=1.0, random_state=random_state)
        val_indices.extend(shuffled.index[:n_val])
        train_indices.extend(shuffled.index[n_val:])
        
    return s1_df.loc[train_indices].copy(), s1_df.loc[val_indices].copy()
