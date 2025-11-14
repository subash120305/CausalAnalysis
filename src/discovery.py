"""
Causal discovery methods: PC, FCI, NOTEARS.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def run_pc_algorithm(
    data: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, object]:
    """
    Run PC algorithm for causal discovery.
    
    Args:
        data: Data matrix (n_samples, n_features)
        alpha: Significance level
    
    Returns:
        (adjacency_matrix, graph_object)
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc
        
        cg = pc(data, alpha=alpha, stable=True)
        adj_matrix = cg.G.graph
        
        logger.info(f"PC algorithm completed: {adj_matrix.shape}")
        return adj_matrix, cg
        
    except ImportError:
        logger.error("causal-learn not installed. Install with: pip install causal-learn")
        raise
    except Exception as e:
        logger.error(f"PC algorithm failed: {e}")
        raise


def run_fci_algorithm(
    data: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, object]:
    """
    Run FCI algorithm for causal discovery.
    
    Args:
        data: Data matrix (n_samples, n_features)
        alpha: Significance level
    
    Returns:
        (adjacency_matrix, graph_object)
    """
    try:
        from causallearn.search.ConstraintBased.FCI import fci
        
        cg = fci(data, alpha=alpha)
        adj_matrix = cg.G.graph
        
        logger.info(f"FCI algorithm completed: {adj_matrix.shape}")
        return adj_matrix, cg
        
    except ImportError:
        logger.error("causal-learn not installed. Install with: pip install causal-learn")
        raise
    except Exception as e:
        logger.error(f"FCI algorithm failed: {e}")
        raise


def run_notears(
    data: np.ndarray,
    lambda1: float = 0.1,
    loss_type: str = "l2"
) -> Tuple[np.ndarray, object]:
    """
    Run NOTEARS algorithm for causal discovery.
    
    Args:
        data: Data matrix (n_samples, n_features)
        lambda1: Regularization parameter
        loss_type: Loss function type
    
    Returns:
        (adjacency_matrix, model_object)
    """
    try:
        from notears import notears_linear
        
        W_est = notears_linear(data, lambda1=lambda1, loss_type=loss_type)
        
        logger.info(f"NOTEARS completed: {W_est.shape}")
        return W_est, None
        
    except ImportError:
        error_msg = "notears not installed. Not available for Python 3.13+. Install with: pip install notears (Python <3.13 only)"
        logger.warning(error_msg)
        raise ImportError(error_msg)
    except Exception as e:
        logger.error(f"NOTEARS algorithm failed: {e}")
        raise


def compare_discovery_methods(
    data: pd.DataFrame,
    true_adjacency: Optional[np.ndarray] = None,
    methods: List[str] = ["pc", "fci", "notears"],
    output_dir: Path = Path("results"),
    node_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Run multiple discovery methods and compare results.
    
    Args:
        data: DataFrame with data
        true_adjacency: Optional ground-truth adjacency matrix
        methods: List of methods to run
        output_dir: Output directory
        node_names: Optional list of node names
    
    Returns:
        DataFrame with precision/recall metrics
    """
    if node_names is None:
        node_names = list(data.columns)
    
    data_matrix = data.values
    
    results = []
    
    for method in methods:
        try:
            if method.lower() == "pc":
                adj_matrix, _ = run_pc_algorithm(data_matrix)
            elif method.lower() == "fci":
                adj_matrix, _ = run_fci_algorithm(data_matrix)
            elif method.lower() == "notears":
                try:
                    adj_matrix, _ = run_notears(data_matrix)
                except ImportError as e:
                    logger.warning(f"Skipping NOTEARS: {e}")
                    results.append({
                        "method": method,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "error": "notears not available for Python 3.13+"
                    })
                    continue
            else:
                logger.warning(f"Unknown method: {method}")
                continue
            
            # Convert to binary (NOTEARS may have continuous values)
            adj_binary = (np.abs(adj_matrix) > 1e-6).astype(int)
            
            # Evaluate if true adjacency provided
            if true_adjacency is not None:
                from .metrics import compute_precision_recall
                from .dag_builder import adjacency_to_edges
                
                pred_edges = set(adjacency_to_edges(adj_binary, node_names))
                true_edges = set(adjacency_to_edges(true_adjacency, node_names))
                
                metrics = compute_precision_recall(pred_edges, true_edges)
                metrics["method"] = method
                results.append(metrics)
                
                logger.info(f"{method}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
            else:
                results.append({
                    "method": method,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan
                })
            
            # Save discovered DAG
            from .dag_builder import save_dag_graphviz, adjacency_to_edges
            edges = adjacency_to_edges(adj_binary, node_names)
            dag_path = output_dir / "sample_plots" / f"{method}_dag.png"
            save_dag_graphviz(edges, dag_path, title=f"{method.upper()} Discovered DAG")
            
        except Exception as e:
            logger.error(f"Method {method} failed: {e}")
            results.append({
                "method": method,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "error": str(e)
            })
    
    results_df = pd.DataFrame(results)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "discovery_precision_recall.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved discovery results to {results_path}")
    
    return results_df


def get_sachs_ground_truth() -> Tuple[np.ndarray, List[str]]:
    """
    Get ground-truth adjacency matrix for Sachs dataset.
    
    Returns:
        (adjacency_matrix, node_names)
    """
    # Sachs protein signaling network (11 nodes)
    node_names = [
        "Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", 
        "Akt", "PKA", "PKC", "P38", "Jnk"
    ]
    
    # Ground-truth edges (from Sachs et al. 2005)
    edges = [
        # Direct edges from Sachs paper
        ("Raf", "Mek"),
        ("Mek", "Erk"),
        ("Plcg", "PIP2"),
        ("Plcg", "PIP3"),
        ("PIP2", "PKC"),
        ("PIP3", "Akt"),
        ("Erk", "Akt"),
        ("PKA", "Akt"),
        ("PKA", "Erk"),
        ("PKA", "P38"),
        ("PKA", "Jnk"),
        ("PKC", "P38"),
        ("PKC", "Jnk"),
        ("PKC", "Mek"),
    ]
    
    n_nodes = len(node_names)
    adj = np.zeros((n_nodes, n_nodes))
    
    for source, target in edges:
        i = node_names.index(source)
        j = node_names.index(target)
        adj[i, j] = 1
    
    return adj, node_names
