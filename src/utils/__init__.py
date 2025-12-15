"""Utility functions for traffic network analysis project."""

import random
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """Get the best available device with fallback chain: CUDA -> MPS -> CPU.
    
    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable format.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_score: Current validation score
            model: Model to potentially save weights from
            
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
            
        return False
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save
        """
        self.best_weights = model.state_dict().copy()


def normalize_features(features: torch.Tensor, method: str = "standard") -> torch.Tensor:
    """Normalize node features.
    
    Args:
        features: Input features tensor
        method: Normalization method ('standard', 'minmax', 'l2')
        
    Returns:
        torch.Tensor: Normalized features
    """
    if method == "standard":
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        return (features - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = features.min(dim=0, keepdim=True)[0]
        max_val = features.max(dim=0, keepdim=True)[0]
        return (features - min_val) / (max_val - min_val + 1e-8)
    elif method == "l2":
        return torch.nn.functional.normalize(features, p=2, dim=1)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_adjacency_matrix(edge_index: torch.Tensor, num_nodes: int, 
                          edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Create adjacency matrix from edge index.
    
    Args:
        edge_index: Edge connectivity tensor of shape [2, num_edges]
        num_nodes: Number of nodes in the graph
        edge_weight: Optional edge weights
        
    Returns:
        torch.Tensor: Adjacency matrix of shape [num_nodes, num_nodes]
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1))
    
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = edge_weight
    
    return adj


def add_self_loops(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None,
                  num_nodes: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Add self-loops to edge index.
    
    Args:
        edge_index: Edge connectivity tensor
        edge_weight: Optional edge weights
        num_nodes: Number of nodes (inferred if None)
        
    Returns:
        Tuple of (edge_index, edge_weight) with self-loops added
    """
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    
    # Create self-loop edges
    self_loops = torch.arange(num_nodes, device=edge_index.device).repeat(2, 1)
    
    # Combine with existing edges
    edge_index_with_loops = torch.cat([edge_index, self_loops], dim=1)
    
    if edge_weight is not None:
        # Add self-loop weights (typically 1.0)
        self_loop_weights = torch.ones(num_nodes, device=edge_weight.device)
        edge_weight_with_loops = torch.cat([edge_weight, self_loop_weights], dim=0)
        return edge_index_with_loops, edge_weight_with_loops
    else:
        return edge_index_with_loops, None
