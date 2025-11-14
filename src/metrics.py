"""
Metrics for evaluating causal inference methods.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def compute_ate_rmse(
    estimated_ate: float,
    true_ate: float
) -> float:
    """Compute RMSE for ATE estimate."""
    return np.sqrt((estimated_ate - true_ate) ** 2)


def compute_pehe(
    estimated_ite: np.ndarray,
    true_ite: np.ndarray
) -> float:
    """
    Compute Precision in Estimation of Heterogeneous Effect (PEHE).
    
    PEHE = sqrt(mean((ITE_estimated - ITE_true)^2))
    """
    if len(estimated_ite) != len(true_ite):
        raise ValueError("Estimated and true ITE arrays must have same length")
    
    return np.sqrt(np.mean((estimated_ite - true_ite) ** 2))


def compute_mae(
    estimated: np.ndarray,
    true: np.ndarray
) -> float:
    """Compute Mean Absolute Error."""
    if len(estimated) != len(true):
        raise ValueError("Estimated and true arrays must have same length")
    
    return np.mean(np.abs(estimated - true))


def compute_precision_recall(
    predicted_edges: set,
    true_edges: set
) -> Dict[str, float]:
    """
    Compute precision and recall for edge prediction.
    
    Args:
        predicted_edges: Set of (source, target) tuples
        true_edges: Set of (source, target) tuples
    
    Returns:
        Dictionary with 'precision', 'recall', 'f1' keys
    """
    if len(predicted_edges) == 0 and len(true_edges) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    if len(predicted_edges) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if len(true_edges) == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    
    tp = len(predicted_edges & true_edges)
    fp = len(predicted_edges - true_edges)
    fn = len(true_edges - predicted_edges)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def evaluate_estimator(
    estimated_ate: float,
    true_ate: Optional[float] = None,
    estimated_ite: Optional[np.ndarray] = None,
    true_ite: Optional[np.ndarray] = None,
    runtime_seconds: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluate an estimator and return metrics dictionary.
    
    Returns:
        Dictionary with computed metrics
    """
    metrics = {
        "estimated_ate": estimated_ate,
        "runtime_seconds": runtime_seconds
    }
    
    if true_ate is not None:
        metrics["true_ate"] = true_ate
        metrics["ate_rmse"] = compute_ate_rmse(estimated_ate, true_ate)
        metrics["ate_error"] = abs(estimated_ate - true_ate)
    
    if estimated_ite is not None and true_ite is not None:
        metrics["pehe"] = compute_pehe(estimated_ite, true_ite)
        metrics["mae_ite"] = compute_mae(estimated_ite, true_ite)
    
    return metrics
