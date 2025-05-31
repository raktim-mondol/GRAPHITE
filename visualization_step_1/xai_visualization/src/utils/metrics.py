"""
Evaluation metrics for binary classification tasks.

This module provides various metrics for evaluating the performance of binary
classification models, particularly for segmentation tasks.
"""

import numpy as np


def calculate_iou(pred_mask, true_mask):
    """
    Calculate Intersection over Union between two binary masks.
    
    Args:
        pred_mask: Predicted binary mask
        true_mask: Ground truth binary mask
        
    Returns:
        IoU score
    """
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    return np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0


def calculate_precision(pred_mask, true_mask):
    """
    Calculate precision between two binary masks.
    
    Args:
        pred_mask: Predicted binary mask
        true_mask: Ground truth binary mask
        
    Returns:
        Precision score
    """
    true_positive = np.sum(np.logical_and(pred_mask, true_mask))
    false_positive = np.sum(np.logical_and(pred_mask, np.logical_not(true_mask)))
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0


def calculate_recall(pred_mask, true_mask):
    """
    Calculate recall between two binary masks.
    
    Args:
        pred_mask: Predicted binary mask
        true_mask: Ground truth binary mask
        
    Returns:
        Recall score
    """
    true_positive = np.sum(np.logical_and(pred_mask, true_mask))
    false_negative = np.sum(np.logical_and(np.logical_not(pred_mask), true_mask))
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0


def calculate_f1_score(precision, recall):
    """
    Calculate F1 score from precision and recall.
    
    Args:
        precision: Precision score
        recall: Recall score
        
    Returns:
        F1 score
    """
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


def calculate_metrics(pred, gt):
    """
    Calculate comprehensive metrics between prediction and ground truth masks.
    
    Args:
        pred: Predicted binary mask
        gt: Ground truth binary mask
        
    Returns:
        Dictionary containing various metrics
    """
    # Calculate confusion matrix elements
    TP = np.sum((pred == 1) & (gt == 1))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    
    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'iou': iou,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN
    } 