# utils/imports.py

# Standard library imports
import os
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
from pathlib import Path
import traceback
from utils.imports import *
import os
import cv2
import json
from pathlib import Path
import numpy as np
import cv2
from typing import Tuple, List, Dict
from tqdm import tqdm
import random

import json
import math
import logging
import traceback
import glob
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
# Scientific and numerical computing
import numpy as np
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
import numpy as np
import cv2
from typing import Tuple, List, Dict
from tqdm import tqdm
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
from torchsummary import summary

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

import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
import json
import dask.array as da
from tqdm import tqdm
from typing import Tuple, List, Dict
# Custom imports (add your custom modules here)
#from .macenko_color_normalizer_v2 import MacenkoColorNormalization, NormalizationError

# Type aliases (add your custom type aliases here)
PathLike = Union[str, Path]
Tensor = torch.Tensor
Array = np.ndarray

# Utility functions for imports
def verify_imports():
    """Verify that all required packages are installed and accessible."""
    required_packages = {
        'torch': torch.__version__,
        'numpy': np.__version__,
        'PIL': Image.__version__,
        'sklearn': sklearn.__version__,
        'timm': timm.__version__,
        'yaml': yaml.__version__,
        'matplotlib': plt.__version__,
        'cv2': cv2.__version__,
        'networkx': nx.__version__
    }
    
    print("Installed package versions:")
    for package, version in required_packages.items():
        print(f"{package}: {version}")

#def set_random_seeds(seed: int = 42):
#    """Set random seeds for reproducibility."""
#    np.random.seed(seed)
#    torch.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False
    
    
#def set_seed(seed):
#    random.seed(seed)
#    np.random.seed(seed)
#    torch.manual_seed(seed)
#    if torch.cuda.is_available():
#        torch.cuda.manual_seed(seed)
#        torch.cuda.manual_seed_all(seed)
#        torch.backends.cudnn.deterministic = True
#        torch.backends.cudnn.benchmark = False
#
#set_seed(78)  # You can choose any integer as the seed
##torch.use_deterministic_algorithms(True)


def get_device() -> torch.device:
    """Get the best available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# Constants (add your constants here)
DEFAULT_CONFIG = {
    'PATCH_SIZE': 224,
    'NUM_CLASSES': 2,
    'BATCH_SIZE': 2,
    'NUM_WORKERS': 4,
    'LEARNING_RATE': 1e-4,
    'WEIGHT_DECAY': 1e-5,
}

# Then in other files, you can import everything you need like this:
# from utils.imports import *

# Or import specific items:
# from utils.imports import (
#     torch, 
#     np, 
#     Path, 
#     verify_imports, 
#     set_random_seeds, 
#     get_device,
#     DEFAULT_CONFIG
# )