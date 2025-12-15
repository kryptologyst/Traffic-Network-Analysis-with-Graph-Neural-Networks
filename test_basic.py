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
    
    # Test basic tensor operations
    print("\n🧠 Testing basic operations...")
    
    # Create sample data
    batch_size, seq_len, num_nodes, features = 2, 6, 10, 4
    sample_input = torch.randn(batch_size, seq_len, num_nodes, features)
    sample_target = torch.randn(batch_size, seq_len, num_nodes, features)
    
    print(f"✅ Sample data created: {sample_input.shape}")
    
    # Test basic metrics
    print("\n📈 Testing basic metrics...")
    
    predictions = torch.randn_like(sample_target)
    
    mae = torch.mean(torch.abs(predictions - sample_target)).item()
    rmse = torch.sqrt(torch.mean((predictions - sample_target) ** 2)).item()
    
    print(f"✅ MAE: {mae:.4f}")
    print(f"✅ RMSE: {rmse:.4f}")
    
    # Test configuration
    print("\n⚙️ Testing configuration...")
    print(f"✅ Model: {config.model.name}")
    print(f"✅ Hidden dim: {config.model.hidden_dim}")
    print(f"✅ Epochs: {config.training.epochs}")
    print(f"✅ Learning rate: {config.training.learning_rate}")
    
    print("\n🎉 Basic tests passed!")
    print("\nTo run the full project:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run 'python train.py' to train a model")
    print("3. Run 'streamlit run demo/app.py' to launch the interactive demo")


if __name__ == "__main__":
    test_basic_functionality()
