"""
DAG builder utilities for visualizing causal graphs.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import networkx as nx

logger = logging.getLogger(__name__)

try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    logger.warning("graphviz not available. DAG visualization will be skipped.")
    # Create dummy Digraph class
    class Digraph:
        pass


def save_dag_graphviz(
    edges: List[Tuple[str, str]],
    output_path: Path,
    node_labels: Optional[Dict[str, str]] = None,
    title: str = "Causal DAG"
) -> None:
    """
    Save a DAG visualization using Graphviz.
    
    Args:
        edges: List of (source, target) tuples
        output_path: Path to save PNG file
        node_labels: Optional dictionary mapping node names to display labels
        title: Graph title
    """
    if not GRAPHVIZ_AVAILABLE:
        logger.warning(f"Skipping DAG visualization (graphviz not installed): {output_path}")
        return
    
    dot = Digraph(comment=title)
    dot.attr(rankdir='LR')
    dot.attr(size='8,5')
    dot.attr(label=title)
    
    # Collect all nodes
    nodes = set()
    for source, target in edges:
        nodes.add(source)
        nodes.add(target)
    
    # Add nodes with labels
    for node in nodes:
        label = node_labels.get(node, node) if node_labels else node
        dot.node(node, label)
    
    # Add edges
    for source, target in edges:
        dot.edge(source, target)
    
    # Save as PNG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dot.render(str(output_path.with_suffix('')), format='png', cleanup=True)
    logger.info(f"Saved DAG visualization to {output_path}")


def networkx_to_graphviz(
    graph: nx.DiGraph,
    output_path: Path,
    title: str = "Causal DAG"
) -> None:
    """
    Convert NetworkX DiGraph to Graphviz PNG.
    
    Args:
        graph: NetworkX directed graph
        output_path: Path to save PNG file
        title: Graph title
    """
    edges = list(graph.edges())
    node_labels = {n: str(n) for n in graph.nodes()}
    save_dag_graphviz(edges, output_path, node_labels, title)


def adjacency_to_edges(adj_matrix, node_names: List[str]) -> List[Tuple[str, str]]:
    """
    Convert adjacency matrix to edge list.
    
    Args:
        adj_matrix: 2D array or matrix (numpy array or list of lists)
        node_names: List of node names corresponding to matrix indices
    
    Returns:
        List of (source, target) tuples
    """
    import numpy as np
    
    adj = np.array(adj_matrix)
    edges = []
    
    for i, source in enumerate(node_names):
        for j, target in enumerate(node_names):
            if adj[i, j] != 0:
                edges.append((source, target))
    
    return edges
