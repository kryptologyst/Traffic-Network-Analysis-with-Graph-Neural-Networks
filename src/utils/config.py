"""Configuration management for traffic network analysis."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for GNN models."""
    name: str = "STGCN"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "relu"
    use_batch_norm: bool = True
    use_residual: bool = True
    
    # Traffic-specific parameters
    temporal_kernel_size: int = 3
    spatial_kernel_size: int = 3
    num_temporal_filters: int = 64
    num_spatial_filters: int = 64


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    dataset_name: str = "synthetic_traffic"
    data_dir: str = "data"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Traffic-specific parameters
    sequence_length: int = 12  # Historical time steps
    prediction_horizon: int = 3  # Future time steps to predict
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Data augmentation
    noise_std: float = 0.01
    dropout_rate: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 10
    gradient_clip: float = 1.0
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging
    log_interval: int = 10
    save_interval: int = 10


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    metrics: List[str] = None
    horizons: List[int] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["mae", "rmse", "mape"]
        if self.horizons is None:
            self.horizons = [1, 3, 6, 12]


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    evaluation: EvaluationConfig = None
    
    # General settings
    seed: int = 42
    device: str = "auto"  # auto, cuda, mps, cpu
    output_dir: str = "outputs"
    experiment_name: str = "traffic_forecasting"
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Initialize default configs if None
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create experiment directory
        exp_dir = Path(self.output_dir) / self.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config: Loaded configuration
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries to config objects
        model_config = ModelConfig(**config_dict.get('model', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        # Extract general settings
        general_settings = {k: v for k, v in config_dict.items() 
                          if k not in ['model', 'data', 'training', 'evaluation']}
        
        return cls(
            model=model_config,
            data=data_config,
            training=training_config,
            evaluation=evaluation_config,
            **general_settings
        )
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config_path: Path to save YAML configuration file
        """
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'seed': self.seed,
            'device': self.device,
            'output_dir': self.output_dir,
            'experiment_name': self.experiment_name
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def get_default_config() -> Config:
    """Get default configuration.
    
    Returns:
        Config: Default configuration
    """
    return Config()


def create_config_file(config_path: str = "configs/default.yaml") -> None:
    """Create a default configuration file.
    
    Args:
        config_path: Path to save the configuration file
    """
    config = get_default_config()
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(config_path)
    print(f"Default configuration saved to {config_path}")
