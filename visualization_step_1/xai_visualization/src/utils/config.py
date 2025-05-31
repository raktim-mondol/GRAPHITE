"""
Configuration management for the XAI visualization tool.

This module provides utilities for managing configuration parameters
from files and command line arguments.
"""

import os
import yaml
import random
import numpy as np
import torch


class Config:
    """Configuration manager for the XAI visualization tool."""
    
    def __init__(self, config_path=None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file (optional)
        """
        # Default configuration
        self.patch_size = 224
        self.stride = 224
        self.target_class = 1
        self.use_color_normalization = False
        self.feat_dim = 512
        self.proj_dim = 128
        self.num_classes = 2
        
        # SHAP-specific parameters
        self.background_samples = 10
        
        # LIME-specific parameters
        self.lime_num_samples = 100
        self.lime_batch_size = 10
        self.lime_num_features = 10
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update attributes from config file
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def update_from_args(self, args):
        """
        Update configuration from command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        # Update from args if they exist
        if hasattr(args, 'patch_size'):
            self.patch_size = args.patch_size
        if hasattr(args, 'stride'):
            self.stride = args.stride
        if hasattr(args, 'target_class'):
            self.target_class = args.target_class
    
    def set_seed(self, seed):
        """
        Set random seeds for reproducibility.
        
        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def to_dict(self):
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    def save_to_file(self, config_path):
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration file
        """
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False) 