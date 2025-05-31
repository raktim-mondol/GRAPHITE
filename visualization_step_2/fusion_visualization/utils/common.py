# utils/common.py

import numpy as np
from scipy.ndimage import gaussian_filter

def normalize_attention(attention_map):
    """Normalize attention map to [0,1] range"""
    if attention_map.max() == attention_map.min():
        return np.zeros_like(attention_map)
    return (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

def apply_gaussian_smoothing(attention_map, sigma=150):
    """Apply Gaussian smoothing to attention map"""
    return gaussian_filter(attention_map, sigma=sigma)