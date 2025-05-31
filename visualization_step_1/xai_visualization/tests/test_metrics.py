"""
Tests for evaluation metrics.
"""

import pytest
import numpy as np
from src.utils.metrics import (
    calculate_iou, calculate_precision, calculate_recall, 
    calculate_f1_score, calculate_metrics
)


def test_calculate_iou():
    """Test IoU calculation."""
    # Perfect match
    pred = np.array([[1, 1], [0, 0]])
    gt = np.array([[1, 1], [0, 0]])
    assert calculate_iou(pred, gt) == 1.0
    
    # No overlap
    pred = np.array([[1, 1], [0, 0]])
    gt = np.array([[0, 0], [1, 1]])
    assert calculate_iou(pred, gt) == 0.0
    
    # Partial overlap
    pred = np.array([[1, 1], [1, 0]])
    gt = np.array([[1, 0], [1, 1]])
    iou = calculate_iou(pred, gt)
    assert 0 < iou < 1


def test_calculate_precision():
    """Test precision calculation."""
    # Perfect precision
    pred = np.array([[1, 0], [0, 0]])
    gt = np.array([[1, 0], [0, 0]])
    assert calculate_precision(pred, gt) == 1.0
    
    # Zero precision (all false positives)
    pred = np.array([[1, 1], [1, 1]])
    gt = np.array([[0, 0], [0, 0]])
    assert calculate_precision(pred, gt) == 0.0


def test_calculate_recall():
    """Test recall calculation."""
    # Perfect recall
    pred = np.array([[1, 0], [0, 0]])
    gt = np.array([[1, 0], [0, 0]])
    assert calculate_recall(pred, gt) == 1.0
    
    # Zero recall (all false negatives)
    pred = np.array([[0, 0], [0, 0]])
    gt = np.array([[1, 1], [1, 1]])
    assert calculate_recall(pred, gt) == 0.0


def test_calculate_f1_score():
    """Test F1 score calculation."""
    # Perfect F1
    assert calculate_f1_score(1.0, 1.0) == 1.0
    
    # Zero F1
    assert calculate_f1_score(0.0, 1.0) == 0.0
    assert calculate_f1_score(1.0, 0.0) == 0.0
    
    # Balanced F1
    f1 = calculate_f1_score(0.5, 0.5)
    assert f1 == 0.5


def test_calculate_metrics():
    """Test comprehensive metrics calculation."""
    pred = np.array([[1, 1], [0, 0]])
    gt = np.array([[1, 0], [0, 1]])
    
    metrics = calculate_metrics(pred, gt)
    
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'iou' in metrics
    assert 'TP' in metrics
    assert 'FP' in metrics
    assert 'FN' in metrics
    assert 'TN' in metrics
    
    # Check that TP + FP + FN + TN equals total pixels
    total_pixels = pred.size
    assert metrics['TP'] + metrics['FP'] + metrics['FN'] + metrics['TN'] == total_pixels 