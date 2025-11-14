"""
Visualization utilities for causal inference results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_ate_comparison(
    estimator_results: Dict[str, float],
    true_ate: Optional[float] = None,
    output_path: Path = Path("results/sample_plots/ate_comparison.png")
) -> None:
    """
    Plot comparison of ATE estimates across estimators.
    
    Args:
        estimator_results: Dictionary mapping estimator names to ATE estimates
        true_ate: Optional true ATE value for reference line
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    estimators = list(estimator_results.keys())
    ate_values = list(estimator_results.values())
    
    bars = ax.bar(estimators, ate_values, alpha=0.7)
    
    if true_ate is not None:
        ax.axhline(y=true_ate, color='r', linestyle='--', label=f'True ATE = {true_ate:.4f}')
        ax.legend()
    
    ax.set_xlabel('Estimator', fontsize=12)
    ax.set_ylabel('ATE Estimate', fontsize=12)
    ax.set_title('Average Treatment Effect (ATE) Comparison', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ATE comparison plot to {output_path}")


def plot_ite_scatter(
    estimated_ite: np.ndarray,
    true_ite: np.ndarray,
    output_path: Path = Path("results/sample_plots/ite_scatter.png")
) -> None:
    """
    Plot scatter plot of estimated vs true ITE.
    
    Args:
        estimated_ite: Array of estimated individual treatment effects
        true_ite: Array of true individual treatment effects
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(true_ite, estimated_ite, alpha=0.5)
    
    # Add diagonal line
    min_val = min(np.min(true_ite), np.min(estimated_ite))
    max_val = max(np.max(true_ite), np.max(estimated_ite))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    ax.set_xlabel('True ITE', fontsize=12)
    ax.set_ylabel('Estimated ITE', fontsize=12)
    ax.set_title('Individual Treatment Effect (ITE) Scatter Plot', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ITE scatter plot to {output_path}")


def plot_estimator_runtimes(
    runtimes: Dict[str, float],
    output_path: Path = Path("results/sample_plots/runtimes.png")
) -> None:
    """Plot estimator runtimes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    estimators = list(runtimes.keys())
    times = list(runtimes.values())
    
    bars = ax.bar(estimators, times, alpha=0.7, color='steelblue')
    ax.set_xlabel('Estimator', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Estimator Runtimes', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved runtime plot to {output_path}")


def plot_discovery_metrics(
    precision_recall_results: Dict[str, Dict[str, float]],
    output_path: Path = Path("results/sample_plots/discovery_metrics.png")
) -> None:
    """
    Plot precision-recall metrics for causal discovery methods.
    
    Args:
        precision_recall_results: Dict mapping method names to metrics dicts
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(precision_recall_results.keys())
    precisions = [precision_recall_results[m]["precision"] for m in methods]
    recalls = [precision_recall_results[m]["recall"] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, precisions, width, label='Precision', alpha=0.7)
    bars2 = ax.bar(x + width/2, recalls, width, label='Recall', alpha=0.7)
    
    ax.set_xlabel('Discovery Method', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Causal Discovery: Precision and Recall', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45)
    ax.legend()
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved discovery metrics plot to {output_path}")
