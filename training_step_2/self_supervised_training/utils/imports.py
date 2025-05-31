# utils/imports.py

# Standard library imports
import os
import sys
import json
import math
import logging
import traceback
import glob
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime

# Scientific and numerical computing
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata, interp2d
from sklearn.metrics import (
    roc_curve, 
    auc, 
    accuracy_score, 
    f1_score, 
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

# Deep learning and PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch

from torch_scatter import scatter_mean, scatter_add

try:
    from torchsummary import summary
except ImportError:
    print("Warning: torchsummary not available. Model summary will be skipped.")
    summary = None

# Image processing
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filters, morphology
import dask.array as da
import dask_image.imread

# Machine learning and model utilities
import timm
import networkx as nx

# Configuration and utilities
import yaml
from yaml import SafeLoader
from tqdm import tqdm

# Type aliases
PathLike = Union[str, Path]
Tensor = torch.Tensor
Array = np.ndarray

# Utility functions
def set_seed(seed: int = 78):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Only set deterministic if supported
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except:
            pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Only set CUBLAS config if supported
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except:
        pass

# Don't set default seed during import to avoid hanging
# Users should call set_seed() explicitly when needed

def get_device() -> torch.device:
    """Get the best available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# Constants
DEFAULT_CONFIG = {
    'PATCH_SIZE': 224,
    'NUM_CLASSES': 2,
    'BATCH_SIZE': 4,
    'NUM_WORKERS': 4,
    'LEARNING_RATE': 1e-3,
    'WEIGHT_DECAY': 1e-5,
}