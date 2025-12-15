"""Evaluation metrics for traffic forecasting."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TrafficMetrics:
    """Traffic forecasting evaluation metrics."""
    
    def __init__(self, horizons: List[int] = None):
        """Initialize metrics.
        
        Args:
            horizons: List of prediction horizons to evaluate
        """
        if horizons is None:
            horizons = [1, 3, 6, 12]
        self.horizons = horizons
        
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Compute all metrics.
        
        Args:
            predictions: Predicted values [batch, horizon, num_nodes, features]
            targets: Ground truth values [batch, horizon, num_nodes, features]
            mask: Optional mask for valid predictions
            
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        # Compute metrics for each horizon
        for h in self.horizons:
            if h <= predictions.size(1):
                pred_h = predictions[:, h-1, :, :]  # [batch, num_nodes, features]
                target_h = targets[:, h-1, :, :]
                
                if mask is not None:
                    mask_h = mask[:, h-1, :, :]
                    pred_h = pred_h * mask_h
                    target_h = target_h * mask_h
                
                # Compute metrics for this horizon
                mae = self.mae(pred_h, target_h, mask_h if mask is not None else None)
                rmse = self.rmse(pred_h, target_h, mask_h if mask is not None else None)
                mape = self.mape(pred_h, target_h, mask_h if mask is not None else None)
                
                metrics[f'mae_h{h}'] = mae
                metrics[f'rmse_h{h}'] = rmse
                metrics[f'mape_h{h}'] = mape
        
        # Compute overall metrics
        metrics['mae'] = self.mae(predictions, targets, mask)
        metrics['rmse'] = self.rmse(predictions, targets, mask)
        metrics['mape'] = self.mape(predictions, targets, mask)
        
        return metrics
    
    def mae(self, predictions: torch.Tensor, targets: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> float:
        """Compute Mean Absolute Error.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            mask: Optional mask for valid predictions
            
        Returns:
            MAE value
        """
        if mask is not None:
            error = torch.abs(predictions - targets) * mask
            return error.sum() / mask.sum().clamp(min=1e-8)
        else:
            return torch.mean(torch.abs(predictions - targets)).item()
    
    def rmse(self, predictions: torch.Tensor, targets: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> float:
        """Compute Root Mean Square Error.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            mask: Optional mask for valid predictions
            
        Returns:
            RMSE value
        """
        if mask is not None:
            error = (predictions - targets) ** 2 * mask
            return torch.sqrt(error.sum() / mask.sum().clamp(min=1e-8)).item()
        else:
            return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    
    def mape(self, predictions: torch.Tensor, targets: torch.Tensor,
             mask: Optional[torch.Tensor] = None, epsilon: float = 1e-8) -> float:
        """Compute Mean Absolute Percentage Error.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            mask: Optional mask for valid predictions
            epsilon: Small value to avoid division by zero
            
        Returns:
            MAPE value
        """
        if mask is not None:
            # Avoid division by zero
            targets_safe = torch.where(torch.abs(targets) < epsilon, 
                                     torch.ones_like(targets) * epsilon, targets)
            error = torch.abs((predictions - targets) / targets_safe) * mask
            return (error.sum() / mask.sum().clamp(min=1e-8) * 100).item()
        else:
            targets_safe = torch.where(torch.abs(targets) < epsilon, 
                                     torch.ones_like(targets) * epsilon, targets)
            return (torch.mean(torch.abs((predictions - targets) / targets_safe)) * 100).item()
    
    def smape(self, predictions: torch.Tensor, targets: torch.Tensor,
              mask: Optional[torch.Tensor] = None) -> float:
        """Compute Symmetric Mean Absolute Percentage Error.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            mask: Optional mask for valid predictions
            
        Returns:
            SMAPE value
        """
        if mask is not None:
            numerator = torch.abs(predictions - targets) * mask
            denominator = (torch.abs(predictions) + torch.abs(targets)) / 2 * mask
            error = numerator / denominator.clamp(min=1e-8)
            return (error.sum() / mask.sum().clamp(min=1e-8) * 100).item()
        else:
            numerator = torch.abs(predictions - targets)
            denominator = (torch.abs(predictions) + torch.abs(targets)) / 2
            return (torch.mean(numerator / denominator.clamp(min=1e-8)) * 100).item()
    
    def r2_score(self, predictions: torch.Tensor, targets: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> float:
        """Compute R-squared score.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            mask: Optional mask for valid predictions
            
        Returns:
            R-squared value
        """
        if mask is not None:
            pred_masked = predictions * mask
            target_masked = targets * mask
            
            ss_res = torch.sum((target_masked - pred_masked) ** 2)
            ss_tot = torch.sum((target_masked - torch.mean(target_masked)) ** 2)
            
            return (1 - ss_res / ss_tot.clamp(min=1e-8)).item()
        else:
            ss_res = torch.sum((targets - predictions) ** 2)
            ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
            
            return (1 - ss_res / ss_tot.clamp(min=1e-8)).item()


class TrafficEvaluator:
    """Traffic forecasting evaluator."""
    
    def __init__(self, metrics: TrafficMetrics = None):
        """Initialize evaluator.
        
        Args:
            metrics: Metrics to compute
        """
        if metrics is None:
            metrics = TrafficMetrics()
        self.metrics = metrics
        
    def evaluate(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                device: torch.device, graph_data: torch_geometric.data.Data) -> Dict[str, float]:
        """Evaluate model on dataset.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader
            device: Device to run evaluation on
            graph_data: Graph structure data
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                
                # Forward pass
                predictions = model(input_seq, graph_data.edge_index.to(device))
                
                # Reshape for metrics computation
                batch_size, seq_len, num_nodes, features = input_seq.shape
                pred_horizon = target_seq.size(1)
                
                # Expand predictions to match target horizon
                if predictions.dim() == 3:  # [batch, num_nodes, features]
                    predictions = predictions.unsqueeze(1).expand(-1, pred_horizon, -1, -1)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(target_seq.cpu())
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        results = self.metrics.compute_metrics(all_predictions, all_targets)
        
        return results
    
    def evaluate_horizon_wise(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                            device: torch.device, graph_data: torch_geometric.data.Data) -> Dict[str, List[float]]:
        """Evaluate model horizon-wise.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader
            device: Device to run evaluation on
            graph_data: Graph structure data
            
        Returns:
            Dictionary of horizon-wise metrics
        """
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                
                # Forward pass
                predictions = model(input_seq, graph_data.edge_index.to(device))
                
                # Reshape for metrics computation
                batch_size, seq_len, num_nodes, features = input_seq.shape
                pred_horizon = target_seq.size(1)
                
                # Expand predictions to match target horizon
                if predictions.dim() == 3:  # [batch, num_nodes, features]
                    predictions = predictions.unsqueeze(1).expand(-1, pred_horizon, -1, -1)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(target_seq.cpu())
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute horizon-wise metrics
        results = {
            'mae': [],
            'rmse': [],
            'mape': []
        }
        
        for h in range(all_predictions.size(1)):
            pred_h = all_predictions[:, h, :, :]
            target_h = all_targets[:, h, :, :]
            
            mae = self.metrics.mae(pred_h, target_h)
            rmse = self.metrics.rmse(pred_h, target_h)
            mape = self.metrics.mape(pred_h, target_h)
            
            results['mae'].append(mae)
            results['rmse'].append(rmse)
            results['mape'].append(mape)
        
        return results


def create_metrics_table(results: Dict[str, float], horizons: List[int] = None) -> str:
    """Create a formatted metrics table.
    
    Args:
        results: Dictionary of metric results
        horizons: List of horizons to include
        
    Returns:
        Formatted table string
    """
    if horizons is None:
        horizons = [1, 3, 6, 12]
    
    table = "Traffic Forecasting Results\n"
    table += "=" * 50 + "\n"
    table += f"{'Metric':<10} {'Overall':<10}"
    
    for h in horizons:
        table += f" {'H' + str(h):<10}"
    table += "\n"
    table += "-" * (10 + 10 + len(horizons) * 10) + "\n"
    
    # MAE
    table += f"{'MAE':<10} {results.get('mae', 0):<10.4f}"
    for h in horizons:
        table += f" {results.get(f'mae_h{h}', 0):<10.4f}"
    table += "\n"
    
    # RMSE
    table += f"{'RMSE':<10} {results.get('rmse', 0):<10.4f}"
    for h in horizons:
        table += f" {results.get(f'rmse_h{h}', 0):<10.4f}"
    table += "\n"
    
    # MAPE
    table += f"{'MAPE':<10} {results.get('mape', 0):<10.4f}"
    for h in horizons:
        table += f" {results.get(f'mape_h{h}', 0):<10.4f}"
    table += "\n"
    
    return table


def plot_horizon_metrics(results: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot horizon-wise metrics.
    
    Args:
        results: Dictionary of horizon-wise results
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    horizons = list(range(1, len(results['mae']) + 1))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MAE
    axes[0].plot(horizons, results['mae'], 'b-o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Prediction Horizon')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('Mean Absolute Error')
    axes[0].grid(True, alpha=0.3)
    
    # RMSE
    axes[1].plot(horizons, results['rmse'], 'r-o', linewidth=2, markersize=6)
    axes[1].set_xlabel('Prediction Horizon')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Root Mean Square Error')
    axes[1].grid(True, alpha=0.3)
    
    # MAPE
    axes[2].plot(horizons, results['mape'], 'g-o', linewidth=2, markersize=6)
    axes[2].set_xlabel('Prediction Horizon')
    axes[2].set_ylabel('MAPE (%)')
    axes[2].set_title('Mean Absolute Percentage Error')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
