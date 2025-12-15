"""Data loading and preprocessing for traffic network analysis."""

import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, DataLoader as PyGDataLoader
from typing import Tuple, Optional, List, Dict, Any
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class TrafficDataset(Dataset):
    """Dataset for traffic network data."""
    
    def __init__(self, data_dir: str, sequence_length: int = 12, 
                 prediction_horizon: int = 3, split: str = "train",
                 normalize: bool = True):
        """Initialize traffic dataset.
        
        Args:
            data_dir: Directory containing data files
            sequence_length: Number of historical time steps
            prediction_horizon: Number of future time steps to predict
            split: Data split ('train', 'val', 'test')
            normalize: Whether to normalize features
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.split = split
        self.normalize = normalize
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess data."""
        # Try to load existing processed data
        processed_file = self.data_dir / f"processed_{self.split}.pkl"
        
        if processed_file.exists():
            logger.info(f"Loading processed data from {processed_file}")
            with open(processed_file, 'rb') as f:
                data = pickle.load(f)
                self.features = data['features']
                self.targets = data['targets']
                self.edge_index = data['edge_index']
                self.edge_weight = data.get('edge_weight', None)
                self.scaler = data.get('scaler', None)
        else:
            logger.info("Generating synthetic traffic data")
            self._generate_synthetic_data()
            self._save_processed_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic traffic data for demonstration."""
        # Create a synthetic traffic network
        num_nodes = 20
        num_time_steps = 1000
        
        # Generate random graph structure
        G = nx.barabasi_albert_graph(num_nodes, 3, seed=42)
        edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
        
        # Add reverse edges for directed graph
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Generate synthetic traffic features
        # Features: [traffic_volume, speed, occupancy, incident_indicator]
        np.random.seed(42)
        
        # Base traffic patterns
        base_volume = np.random.uniform(10, 100, (num_time_steps, num_nodes))
        base_speed = np.random.uniform(20, 80, (num_time_steps, num_nodes))
        base_occupancy = np.random.uniform(0.1, 0.9, (num_time_steps, num_nodes))
        base_incidents = np.random.binomial(1, 0.05, (num_time_steps, num_nodes))
        
        # Add temporal patterns (rush hours, weekends)
        time_of_day = np.arange(num_time_steps) % 24
        day_of_week = (np.arange(num_time_steps) // 24) % 7
        
        # Rush hour effect
        rush_hour_mask = ((time_of_day >= 7) & (time_of_day <= 9)) | \
                        ((time_of_day >= 17) & (time_of_day <= 19))
        
        # Weekend effect
        weekend_mask = (day_of_week >= 5)
        
        # Apply patterns
        volume_multiplier = np.ones((num_time_steps, num_nodes))
        volume_multiplier[rush_hour_mask] *= 1.5
        volume_multiplier[weekend_mask] *= 0.8
        
        speed_multiplier = np.ones((num_time_steps, num_nodes))
        speed_multiplier[rush_hour_mask] *= 0.7
        speed_multiplier[weekend_mask] *= 1.1
        
        occupancy_multiplier = np.ones((num_time_steps, num_nodes))
        occupancy_multiplier[rush_hour_mask] *= 1.3
        occupancy_multiplier[weekend_mask] *= 0.9
        
        # Apply multipliers
        traffic_volume = base_volume * volume_multiplier
        traffic_speed = base_speed * speed_multiplier
        traffic_occupancy = base_occupancy * occupancy_multiplier
        
        # Add spatial correlation (neighboring nodes have similar patterns)
        for t in range(num_time_steps):
            for node in range(num_nodes):
                neighbors = [edge[1] for edge in G.edges(node)]
                if neighbors:
                    neighbor_avg = np.mean([traffic_volume[t, n] for n in neighbors])
                    traffic_volume[t, node] = 0.7 * traffic_volume[t, node] + 0.3 * neighbor_avg
                    
                    neighbor_avg = np.mean([traffic_speed[t, n] for n in neighbors])
                    traffic_speed[t, node] = 0.7 * traffic_speed[t, node] + 0.3 * neighbor_avg
        
        # Combine features
        features = np.stack([
            traffic_volume,
            traffic_speed,
            traffic_occupancy,
            base_incidents
        ], axis=-1)  # [time_steps, num_nodes, num_features]
        
        # Create targets (next time step values)
        targets = features[1:]  # Shift by one time step
        
        # Normalize features
        if self.normalize:
            self.scaler = self._fit_scaler(features)
            features = self._normalize(features, self.scaler)
            targets = self._normalize(targets, self.scaler)
        
        # Convert to tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.edge_index = edge_index
        
        # Generate edge weights based on distance
        edge_weight = torch.ones(edge_index.size(1))
        self.edge_weight = edge_weight
        
        logger.info(f"Generated synthetic data: {self.features.shape}")
    
    def _fit_scaler(self, data: np.ndarray) -> Dict[str, Any]:
        """Fit normalization scaler.
        
        Args:
            data: Input data
            
        Returns:
            Dict containing scaler parameters
        """
        # Reshape data for scaling
        data_reshaped = data.reshape(-1, data.shape[-1])
        
        mean = np.mean(data_reshaped, axis=0)
        std = np.std(data_reshaped, axis=0)
        
        return {
            'mean': mean,
            'std': std
        }
    
    def _normalize(self, data: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
        """Normalize data using scaler.
        
        Args:
            data: Input data
            scaler: Scaler parameters
            
        Returns:
            Normalized data
        """
        mean = scaler['mean']
        std = scaler['std']
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        return (data - mean) / std
    
    def _save_processed_data(self):
        """Save processed data to disk."""
        processed_file = self.data_dir / f"processed_{self.split}.pkl"
        
        data = {
            'features': self.features,
            'targets': self.targets,
            'edge_index': self.edge_index,
            'edge_weight': self.edge_weight,
            'scaler': self.scaler
        }
        
        with open(processed_file, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved processed data to {processed_file}")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.features) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_sequence, target_sequence)
        """
        # Get input sequence
        input_start = idx
        input_end = input_start + self.sequence_length
        input_seq = self.features[input_start:input_end]  # [seq_len, num_nodes, features]
        
        # Get target sequence
        target_start = input_end
        target_end = target_start + self.prediction_horizon
        target_seq = self.targets[target_start:target_end]  # [pred_horizon, num_nodes, features]
        
        return input_seq, target_seq
    
    def get_graph_data(self) -> Data:
        """Get graph structure data.
        
        Returns:
            PyTorch Geometric Data object
        """
        return Data(
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            num_nodes=self.features.size(1)
        )


class TrafficDataModule:
    """Data module for traffic forecasting."""
    
    def __init__(self, data_dir: str, sequence_length: int = 12,
                 prediction_horizon: int = 3, batch_size: int = 32,
                 num_workers: int = 4, train_ratio: float = 0.7,
                 val_ratio: float = 0.15, test_ratio: float = 0.15):
        """Initialize data module.
        
        Args:
            data_dir: Directory containing data files
            sequence_length: Number of historical time steps
            prediction_horizon: Number of future time steps to predict
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Create data directory
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self):
        """Setup datasets and data loaders."""
        # Create datasets for each split
        self.train_dataset = TrafficDataset(
            self.data_dir, self.sequence_length, self.prediction_horizon,
            split="train", normalize=True
        )
        
        self.val_dataset = TrafficDataset(
            self.data_dir, self.sequence_length, self.prediction_horizon,
            split="val", normalize=True
        )
        
        self.test_dataset = TrafficDataset(
            self.data_dir, self.sequence_length, self.prediction_horizon,
            split="test", normalize=True
        )
        
        # Get graph data
        self.graph_data = self.train_dataset.get_graph_data()
        
        logger.info(f"Setup complete. Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_graph_data(self) -> Data:
        """Get graph structure data."""
        return self.graph_data


def create_traffic_network(num_nodes: int = 20, seed: int = 42) -> nx.DiGraph:
    """Create a synthetic traffic network.
    
    Args:
        num_nodes: Number of nodes (intersections)
        seed: Random seed
        
    Returns:
        NetworkX directed graph representing the traffic network
    """
    # Create a realistic traffic network using Barabási-Albert model
    G = nx.barabasi_albert_graph(num_nodes, 3, seed=seed)
    
    # Convert to directed graph
    G = G.to_directed()
    
    # Add edge attributes (road properties)
    for u, v in G.edges():
        # Road length (distance between intersections)
        length = np.random.uniform(0.5, 2.0)
        
        # Road capacity (maximum traffic volume)
        capacity = np.random.uniform(50, 200)
        
        # Speed limit
        speed_limit = np.random.uniform(30, 80)
        
        G[u][v]['length'] = length
        G[u][v]['capacity'] = capacity
        G[u][v]['speed_limit'] = speed_limit
    
    return G


def visualize_traffic_network(G: nx.DiGraph, pos: Optional[Dict] = None,
                            save_path: Optional[str] = None) -> None:
    """Visualize traffic network.
    
    Args:
        G: Traffic network graph
        pos: Node positions (if None, will be computed)
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.8)
    
    # Draw edges with different colors based on capacity
    edges = G.edges()
    capacities = [G[u][v]['capacity'] for u, v in edges]
    
    nx.draw_networkx_edges(G, pos, edge_color=capacities, 
                          edge_cmap=plt.cm.viridis, width=2, alpha=0.6)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Add edge labels for length
    edge_labels = {(u, v): f"{G[u][v]['length']:.1f}" 
                   for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title("Synthetic Traffic Network")
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), 
                label='Road Capacity')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
