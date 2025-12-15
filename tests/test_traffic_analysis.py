"""Tests for traffic network analysis project."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, get_device, count_parameters
from utils.config import Config, get_default_config
from data import TrafficDataset, TrafficDataModule
from models import STGCN, DCRNN, GMAN
from eval import TrafficMetrics, TrafficEvaluator


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        assert True  # If no exception is raised, test passes
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        param_count = count_parameters(model)
        assert param_count == 55  # 10*5 + 5 bias


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_config()
        assert isinstance(config, Config)
        assert config.model.name == "STGCN"
        assert config.training.epochs == 100
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = get_default_config()
        
        # Test to_yaml
        config_path = "test_config.yaml"
        config.to_yaml(config_path)
        
        # Test from_yaml
        loaded_config = Config.from_yaml(config_path)
        assert loaded_config.model.name == config.model.name
        
        # Cleanup
        Path(config_path).unlink()


class TestData:
    """Test data loading and preprocessing."""
    
    def test_traffic_dataset(self):
        """Test traffic dataset creation."""
        dataset = TrafficDataset("data", sequence_length=6, prediction_horizon=2)
        
        assert len(dataset) > 0
        assert dataset.features.size(0) > 0
        assert dataset.targets.size(0) > 0
    
    def test_traffic_data_module(self):
        """Test traffic data module."""
        data_module = TrafficDataModule("data", batch_size=16)
        data_module.setup()
        
        assert data_module.train_dataset is not None
        assert data_module.val_dataset is not None
        assert data_module.test_dataset is not None
        
        train_loader = data_module.train_dataloader()
        assert len(train_loader) > 0


class TestModels:
    """Test model implementations."""
    
    def test_stgcn(self):
        """Test STGCN model."""
        model = STGCN(num_nodes=10, in_channels=4, out_channels=4)
        
        # Test forward pass
        batch_size, seq_len = 2, 6
        x = torch.randn(batch_size, seq_len, 10, 4)
        edge_index = torch.randint(0, 10, (2, 20))
        
        output = model(x, edge_index)
        assert output.shape == (batch_size, 10, 4)
    
    def test_dcrnn(self):
        """Test DCRNN model."""
        model = DCRNN(num_nodes=10, in_channels=4, out_channels=4)
        
        # Test forward pass
        batch_size, seq_len = 2, 6
        x = torch.randn(batch_size, seq_len, 10, 4)
        edge_index = torch.randint(0, 10, (2, 20))
        
        output = model(x, edge_index)
        assert output.shape == (batch_size, 10, 4)
    
    def test_gman(self):
        """Test GMAN model."""
        model = GMAN(num_nodes=10, in_channels=4, out_channels=4)
        
        # Test forward pass
        batch_size, seq_len = 2, 6
        x = torch.randn(batch_size, seq_len, 10, 4)
        edge_index = torch.randint(0, 10, (2, 20))
        
        output = model(x, edge_index)
        assert output.shape == (batch_size, 10, 4)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_traffic_metrics(self):
        """Test traffic metrics computation."""
        metrics = TrafficMetrics()
        
        # Create dummy data
        predictions = torch.randn(2, 3, 10, 4)
        targets = torch.randn(2, 3, 10, 4)
        
        # Test individual metrics
        mae = metrics.mae(predictions, targets)
        rmse = metrics.rmse(predictions, targets)
        mape = metrics.mape(predictions, targets)
        
        assert isinstance(mae, float)
        assert isinstance(rmse, float)
        assert isinstance(mape, float)
        assert mae >= 0
        assert rmse >= 0
        assert mape >= 0
    
    def test_metrics_computation(self):
        """Test complete metrics computation."""
        metrics = TrafficMetrics()
        
        predictions = torch.randn(2, 3, 10, 4)
        targets = torch.randn(2, 3, 10, 4)
        
        results = metrics.compute_metrics(predictions, targets)
        
        assert 'mae' in results
        assert 'rmse' in results
        assert 'mape' in results
        assert 'mae_h1' in results
        assert 'rmse_h1' in results
        assert 'mape_h1' in results


if __name__ == "__main__":
    pytest.main([__file__])
