#!/usr/bin/env python3
"""Quick test script to verify the traffic network analysis project works."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np

from utils import set_seed, get_device
from utils.config import get_default_config
from data import TrafficDataModule
from models import STGCN, DCRNN, GMAN
from eval import TrafficMetrics


def test_basic_functionality():
    """Test basic functionality of the traffic analysis project."""
    print("🚦 Testing Traffic Network Analysis Project")
    print("=" * 50)
    
    # Set random seed
    set_seed(42)
    print("✅ Random seed set")
    
    # Get device
    device = get_device()
    print(f"✅ Device detected: {device}")
    
    # Load configuration
    config = get_default_config()
    print(f"✅ Configuration loaded: {config.model.name}")
    
    # Create data module
    print("\n📊 Setting up data...")
    data_module = TrafficDataModule("data", batch_size=8)
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    graph_data = data_module.get_graph_data()
    
    print(f"✅ Data loaded: {len(train_loader.dataset)} training samples")
    print(f"✅ Graph: {graph_data.num_nodes} nodes, {graph_data.edge_index.size(1)} edges")
    
    # Test models
    print("\n🧠 Testing models...")
    
    # Get sample data
    sample_input, sample_target = next(iter(train_loader))
    batch_size, seq_len, num_nodes, features = sample_input.shape
    
    print(f"✅ Sample data shape: {sample_input.shape}")
    
    # Test STGCN
    stgcn = STGCN(num_nodes, features, features, hidden_dim=32, num_layers=2)
    stgcn_output = stgcn(sample_input, graph_data.edge_index)
    print(f"✅ STGCN output shape: {stgcn_output.shape}")
    
    # Test DCRNN
    dcrnn = DCRNN(num_nodes, features, features, hidden_dim=32, num_layers=2)
    dcrnn_output = dcrnn(sample_input, graph_data.edge_index)
    print(f"✅ DCRNN output shape: {dcrnn_output.shape}")
    
    # Test GMAN
    gman = GMAN(num_nodes, features, features, hidden_dim=32, num_layers=2)
    gman_output = gman(sample_input, graph_data.edge_index)
    print(f"✅ GMAN output shape: {gman_output.shape}")
    
    # Test metrics
    print("\n📈 Testing metrics...")
    metrics = TrafficMetrics()
    
    # Create dummy predictions
    predictions = torch.randn_like(sample_target[:, -1, :, :])
    
    mae = metrics.mae(predictions, sample_target[:, -1, :, :])
    rmse = metrics.rmse(predictions, sample_target[:, -1, :, :])
    mape = metrics.mape(predictions, sample_target[:, -1, :, :])
    
    print(f"✅ MAE: {mae:.4f}")
    print(f"✅ RMSE: {rmse:.4f}")
    print(f"✅ MAPE: {mape:.2f}%")
    
    # Test parameter counts
    print("\n🔢 Model parameters:")
    print(f"✅ STGCN: {sum(p.numel() for p in stgcn.parameters()):,} parameters")
    print(f"✅ DCRNN: {sum(p.numel() for p in dcrnn.parameters()):,} parameters")
    print(f"✅ GMAN: {sum(p.numel() for p in gman.parameters()):,} parameters")
    
    print("\n🎉 All tests passed! The project is working correctly.")
    print("\nNext steps:")
    print("1. Run 'python train.py' to train a model")
    print("2. Run 'streamlit run demo/app.py' to launch the interactive demo")
    print("3. Check the README.md for more detailed usage instructions")


if __name__ == "__main__":
    test_basic_functionality()
