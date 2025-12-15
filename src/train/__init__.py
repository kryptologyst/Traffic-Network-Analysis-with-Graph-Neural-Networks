"""Training module for traffic forecasting models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, Optional, Callable, Any
import logging
from pathlib import Path
import time
from tqdm import tqdm

from .utils import EarlyStopping, get_device, set_seed, format_time
from .eval import TrafficEvaluator, TrafficMetrics

logger = logging.getLogger(__name__)


class TrafficTrainer:
    """Trainer for traffic forecasting models."""
    
    def __init__(self, model: nn.Module, config: Any, device: Optional[torch.device] = None):
        """Initialize trainer.
        
        Args:
            model: Model to train
            config: Configuration object
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device or get_device()
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = nn.MSELoss()
        
        # Initialize metrics
        self.metrics = TrafficMetrics(horizons=config.evaluation.horizons)
        self.evaluator = TrafficEvaluator(self.metrics)
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta
        )
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer.
        
        Returns:
            Optimizer instance
        """
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler.
        
        Returns:
            Scheduler instance or None
        """
        if self.config.training.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs
            )
        elif self.config.training.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.epochs // 3,
                gamma=0.1
            )
        elif self.config.training.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader, graph_data: Any) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            graph_data: Graph structure data
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (input_seq, target_seq) in enumerate(pbar):
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.scaler is not None:
                with autocast():
                    predictions = self.model(input_seq, graph_data.edge_index.to(self.device))
                    loss = self.criterion(predictions, target_seq[:, -1, :, :])
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(input_seq, graph_data.edge_index.to(self.device))
                loss = self.criterion(predictions, target_seq[:, -1, :, :])
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}"
            })
            
            # Log training progress
            if batch_idx % self.config.training.log_interval == 0:
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader, graph_data: Any) -> float:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            graph_data: Graph structure data
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_seq, target_seq in val_loader:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        predictions = self.model(input_seq, graph_data.edge_index.to(self.device))
                        loss = self.criterion(predictions, target_seq[:, -1, :, :])
                else:
                    predictions = self.model(input_seq, graph_data.edge_index.to(self.device))
                    loss = self.criterion(predictions, target_seq[:, -1, :, :])
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              graph_data: Any) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            graph_data: Graph structure data
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader, graph_data)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch(val_loader, graph_data)
            self.val_losses.append(val_loss)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
            
            # Save regular checkpoint
            if epoch % self.config.training.save_interval == 0:
                self.save_checkpoint(is_best=False)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {format_time(training_time)}")
        
        # Load best model
        self.load_checkpoint(is_best=True)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time,
            'epochs_trained': self.current_epoch + 1
        }
    
    def evaluate(self, test_loader: DataLoader, graph_data: Any) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            test_loader: Test data loader
            graph_data: Graph structure data
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model...")
        
        # Load best model
        self.load_checkpoint(is_best=True)
        
        # Evaluate
        results = self.evaluator.evaluate(self.model, test_loader, self.device, graph_data)
        
        # Log results
        logger.info("Evaluation Results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        if is_best:
            checkpoint_path = self.output_dir / 'best_model.pth'
        else:
            checkpoint_path = self.output_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, is_best: bool = False):
        """Load model checkpoint.
        
        Args:
            is_best: Whether to load the best model
        """
        if is_best:
            checkpoint_path = self.output_dir / 'best_model.pth'
        else:
            checkpoint_path = self.output_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves.
        
        Args:
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate
        plt.subplot(1, 2, 2)
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        plt.plot(lrs, label='Learning Rate', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def train_model(model: nn.Module, config: Any, train_loader: DataLoader,
                val_loader: DataLoader, test_loader: DataLoader, 
                graph_data: Any) -> Dict[str, Any]:
    """Train a traffic forecasting model.
    
    Args:
        model: Model to train
        config: Configuration object
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        graph_data: Graph structure data
        
    Returns:
        Training and evaluation results
    """
    # Set random seed
    set_seed(config.seed)
    
    # Initialize trainer
    trainer = TrafficTrainer(model, config)
    
    # Train model
    training_results = trainer.train(train_loader, val_loader, graph_data)
    
    # Evaluate model
    evaluation_results = trainer.evaluate(test_loader, graph_data)
    
    # Plot training curves
    trainer.plot_training_curves(save_path=trainer.output_dir / 'training_curves.png')
    
    return {
        'training': training_results,
        'evaluation': evaluation_results
    }
