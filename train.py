#!/usr/bin/env python3
"""Main training script for traffic network analysis."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
from torch.utils.data import DataLoader

from utils import set_seed, get_device, create_config_file
from utils.config import Config, get_default_config
from data import TrafficDataModule
from models import STGCN, DCRNN, GMAN
from train import train_model
from eval import create_metrics_table, plot_horizon_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_model(config: Config, num_nodes: int, in_channels: int, out_channels: int):
    """Create model based on configuration.
    
    Args:
        config: Configuration object
        num_nodes: Number of nodes in the graph
        in_channels: Number of input features
        out_channels: Number of output features
        
    Returns:
        Model instance
    """
    model_name = config.model.name.lower()
    
    if model_name == "stgcn":
        return STGCN(
            num_nodes=num_nodes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            temporal_kernel_size=config.model.temporal_kernel_size,
            dropout=config.model.dropout
        )
    elif model_name == "dcrnn":
        return DCRNN(
            num_nodes=num_nodes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        )
    elif model_name == "gman":
        return GMAN(
            num_nodes=num_nodes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train traffic forecasting models")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model", type=str, choices=["stgcn", "dcrnn", "gman"],
                       help="Model to train")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"],
                       help="Device to use")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        logger.info(f"Loading configuration from {args.config}")
        config = Config.from_yaml(args.config)
    else:
        logger.info("Creating default configuration")
        config = get_default_config()
        # Create config file
        Path(args.config).parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(args.config)
    
    # Override config with command line arguments
    if args.model:
        config.model.name = args.model.upper()
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.device:
        config.device = args.device
    if args.seed:
        config.seed = args.seed
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    # Set device
    if config.device == "auto":
        device = get_device()
    else:
        device = torch.device(config.device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration: {config}")
    
    # Set random seed
    set_seed(config.seed)
    
    # Create data module
    logger.info("Setting up data module...")
    data_module = TrafficDataModule(
        data_dir=config.data.data_dir,
        sequence_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio
    )
    
    # Setup data
    data_module.setup()
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    graph_data = data_module.get_graph_data()
    
    logger.info(f"Data setup complete:")
    logger.info(f"  Train samples: {len(train_loader.dataset)}")
    logger.info(f"  Val samples: {len(val_loader.dataset)}")
    logger.info(f"  Test samples: {len(test_loader.dataset)}")
    logger.info(f"  Graph nodes: {graph_data.num_nodes}")
    logger.info(f"  Graph edges: {graph_data.edge_index.size(1)}")
    
    # Get feature dimensions
    sample_input, sample_target = next(iter(train_loader))
    in_channels = sample_input.size(-1)  # Number of features
    out_channels = sample_target.size(-1)  # Number of features to predict
    
    logger.info(f"Input features: {in_channels}")
    logger.info(f"Output features: {out_channels}")
    
    # Create model
    logger.info(f"Creating {config.model.name} model...")
    model = create_model(config, graph_data.num_nodes, in_channels, out_channels)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Train model
    logger.info("Starting training...")
    results = train_model(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        graph_data=graph_data
    )
    
    # Print results
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {results['training']['best_val_loss']:.4f}")
    logger.info(f"Training time: {results['training']['training_time']:.2f}s")
    
    # Print evaluation metrics
    eval_results = results['evaluation']
    metrics_table = create_metrics_table(eval_results, config.evaluation.horizons)
    logger.info(f"\n{metrics_table}")
    
    # Save results
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.to_yaml(output_dir / "config.yaml")
    
    # Save results
    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            'training': results['training'],
            'evaluation': eval_results
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    
    # Plot horizon-wise metrics
    try:
        from eval import TrafficEvaluator, TrafficMetrics
        evaluator = TrafficEvaluator(TrafficMetrics(config.evaluation.horizons))
        horizon_results = evaluator.evaluate_horizon_wise(model, test_loader, device, graph_data)
        plot_horizon_metrics(horizon_results, save_path=output_dir / "horizon_metrics.png")
    except Exception as e:
        logger.warning(f"Could not plot horizon metrics: {e}")


if __name__ == "__main__":
    main()
