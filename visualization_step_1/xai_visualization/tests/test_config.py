"""
Tests for configuration management.
"""

import pytest
import tempfile
import os
import yaml
from src.utils.config import Config


def test_default_config():
    """Test default configuration values."""
    config = Config()
    
    assert config.patch_size == 224
    assert config.stride == 224
    assert config.target_class == 1
    assert config.use_color_normalization == False
    assert config.feat_dim == 512
    assert config.proj_dim == 128
    assert config.num_classes == 2


def test_config_from_file():
    """Test loading configuration from YAML file."""
    # Create temporary config file
    config_data = {
        'patch_size': 512,
        'stride': 256,
        'target_class': 0,
        'use_color_normalization': True
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_config_path = f.name
    
    try:
        config = Config(temp_config_path)
        
        assert config.patch_size == 512
        assert config.stride == 256
        assert config.target_class == 0
        assert config.use_color_normalization == True
        
    finally:
        os.unlink(temp_config_path)


def test_config_to_dict():
    """Test converting configuration to dictionary."""
    config = Config()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert 'patch_size' in config_dict
    assert 'stride' in config_dict
    assert config_dict['patch_size'] == 224


def test_config_save_to_file():
    """Test saving configuration to file."""
    config = Config()
    config.patch_size = 512
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_config_path = f.name
    
    try:
        config.save_to_file(temp_config_path)
        
        # Load and verify
        with open(temp_config_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config['patch_size'] == 512
        
    finally:
        os.unlink(temp_config_path) 