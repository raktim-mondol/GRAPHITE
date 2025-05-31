"""
Logging utilities for the XAI visualization tool.

This module provides logging functionality with different levels and formatting.
"""

import logging
import sys
from datetime import datetime


def setup_logger(verbose=False, log_file=None):
    """
    Setup the main logger for the application.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
        log_file: Optional file to write logs to
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('xai_visualization')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger():
    """
    Get the existing logger instance.
    
    Returns:
        Logger instance
    """
    return logging.getLogger('xai_visualization') 