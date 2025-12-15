"""Traffic-specific Graph Neural Network models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple, List
import math


class TemporalConv(nn.Module):
    """Temporal convolution layer for traffic forecasting."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 dilation: int = 1, dropout: float = 0.1):
        """Initialize temporal convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of temporal kernel
            dilation: Dilation rate
            dropout: Dropout rate
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Padding to maintain sequence length
        padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, num_nodes, features]
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Reshape for 1D convolution: [batch * num_nodes, features, seq_len]
        batch_size, seq_len, num_nodes, features = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [batch, num_nodes, features, seq_len]
        x = x.view(-1, features, seq_len)  # [batch * num_nodes, features, seq_len]
        
        # Apply temporal convolution
        x = self.conv(x)  # [batch * num_nodes, out_channels, seq_len]
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Reshape back
        x = x.view(batch_size, num_nodes, -1, seq_len)
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch, seq_len, num_nodes, features]
        
        return x


class SpatialConv(nn.Module):
    """Spatial convolution layer using Graph Convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        """Initialize spatial convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dropout: Dropout rate
        """
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features of shape [num_nodes, features]
            edge_index: Edge connectivity
            
        Returns:
            torch.Tensor: Output features
        """
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class STGCNBlock(nn.Module):
    """Spatio-Temporal Graph Convolutional Network block."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 temporal_kernel_size: int = 3, spatial_kernel_size: int = 3,
                 dropout: float = 0.1):
        """Initialize STGCN block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            temporal_kernel_size: Size of temporal kernel
            spatial_kernel_size: Size of spatial kernel
            dropout: Dropout rate
        """
        super().__init__()
        
        # Temporal convolution
        self.temporal_conv = TemporalConv(
            in_channels, out_channels, temporal_kernel_size, dropout=dropout
        )
        
        # Spatial convolution
        self.spatial_conv = SpatialConv(out_channels, out_channels, dropout=dropout)
        
        # Residual connection
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, num_nodes, features]
            edge_index: Edge connectivity
            
        Returns:
            torch.Tensor: Output tensor
        """
        batch_size, seq_len, num_nodes, features = x.shape
        
        # Apply temporal convolution
        x_temporal = self.temporal_conv(x)
        
        # Apply spatial convolution for each time step
        x_spatial = []
        for t in range(seq_len):
            x_t = x_temporal[:, t, :, :]  # [batch, num_nodes, features]
            x_t = x_t.view(-1, x_t.size(-1))  # [batch * num_nodes, features]
            x_t = self.spatial_conv(x_t, edge_index)
            x_t = x_t.view(batch_size, num_nodes, -1)
            x_spatial.append(x_t)
        
        x_spatial = torch.stack(x_spatial, dim=1)  # [batch, seq_len, num_nodes, features]
        
        # Residual connection
        if self.residual is not None:
            residual = self.residual(x.view(-1, features)).view(batch_size, seq_len, num_nodes, -1)
            x_spatial = x_spatial + residual
        
        return x_spatial


class STGCN(nn.Module):
    """Spatio-Temporal Graph Convolutional Network for traffic forecasting."""
    
    def __init__(self, num_nodes: int, in_channels: int, out_channels: int,
                 hidden_dim: int = 64, num_layers: int = 2, 
                 temporal_kernel_size: int = 3, dropout: float = 0.1):
        """Initialize STGCN model.
        
        Args:
            num_nodes: Number of nodes in the graph
            in_channels: Number of input features
            out_channels: Number of output features
            hidden_dim: Hidden dimension size
            num_layers: Number of STGCN blocks
            temporal_kernel_size: Size of temporal kernel
            dropout: Dropout rate
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.out_channels = out_channels
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        
        # STGCN blocks
        self.stgcn_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.stgcn_blocks.append(
                STGCNBlock(in_dim, hidden_dim, temporal_kernel_size, dropout=dropout)
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, num_nodes, features]
            edge_index: Edge connectivity
            
        Returns:
            torch.Tensor: Output tensor of shape [batch, num_nodes, out_channels]
        """
        # Input projection
        x = self.input_proj(x)
        
        # Apply STGCN blocks
        for block in self.stgcn_blocks:
            x = block(x, edge_index)
        
        # Take the last time step for prediction
        x = x[:, -1, :, :]  # [batch, num_nodes, hidden_dim]
        
        # Output projection
        x = self.output_proj(x)  # [batch, num_nodes, out_channels]
        
        return x


class DCRNN(nn.Module):
    """Diffusion Convolutional Recurrent Neural Network for traffic forecasting."""
    
    def __init__(self, num_nodes: int, in_channels: int, out_channels: int,
                 hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        """Initialize DCRNN model.
        
        Args:
            num_nodes: Number of nodes in the graph
            in_channels: Number of input features
            out_channels: Number of output features
            hidden_dim: Hidden dimension size
            num_layers: Number of RNN layers
            dropout: Dropout rate
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        
        # Diffusion convolution layers
        self.diffusion_convs = nn.ModuleList()
        for _ in range(num_layers):
            self.diffusion_convs.append(
                DiffusionConv(hidden_dim, hidden_dim, dropout=dropout)
            )
        
        # GRU cells
        self.gru_cells = nn.ModuleList()
        for _ in range(num_layers):
            self.gru_cells.append(nn.GRUCell(hidden_dim, hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, num_nodes, features]
            edge_index: Edge connectivity
            edge_weight: Optional edge weights
            
        Returns:
            torch.Tensor: Output tensor of shape [batch, num_nodes, out_channels]
        """
        batch_size, seq_len, num_nodes, features = x.shape
        
        # Input projection
        x = self.input_proj(x)
        
        # Initialize hidden states
        h = [torch.zeros(batch_size * num_nodes, self.hidden_dim, device=x.device) 
             for _ in range(self.num_layers)]
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :, :].view(-1, self.hidden_dim)  # [batch * num_nodes, hidden_dim]
            
            for layer in range(self.num_layers):
                # Diffusion convolution
                x_t = self.diffusion_convs[layer](x_t, edge_index, edge_weight)
                
                # GRU update
                h[layer] = self.gru_cells[layer](x_t, h[layer])
                x_t = h[layer]
        
        # Output projection
        x = self.output_proj(x_t.view(batch_size, num_nodes, -1))
        
        return x


class DiffusionConv(nn.Module):
    """Diffusion convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        """Initialize diffusion convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dropout: Dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Forward and backward diffusion
        self.conv_forward = GCNConv(in_channels, out_channels)
        self.conv_backward = GCNConv(in_channels, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_weight: Optional edge weights
            
        Returns:
            torch.Tensor: Output features
        """
        # Forward diffusion
        x_forward = self.conv_forward(x, edge_index)
        
        # Backward diffusion (reverse edges)
        edge_index_reverse = torch.stack([edge_index[1], edge_index[0]], dim=0)
        x_backward = self.conv_backward(x, edge_index_reverse)
        
        # Combine forward and backward
        x = x_forward + x_backward
        x = F.relu(x)
        x = self.dropout(x)
        
        return x


class GMAN(nn.Module):
    """Graph Multi-Attention Network for traffic forecasting."""
    
    def __init__(self, num_nodes: int, in_channels: int, out_channels: int,
                 hidden_dim: int = 64, num_heads: int = 8, num_layers: int = 2,
                 dropout: float = 0.1):
        """Initialize GMAN model.
        
        Args:
            num_nodes: Number of nodes in the graph
            in_channels: Number of input features
            out_channels: Number of output features
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            dropout: Dropout rate
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        
        # Spatial attention layers
        self.spatial_attentions = nn.ModuleList()
        for _ in range(num_layers):
            self.spatial_attentions.append(
                SpatialAttention(hidden_dim, num_heads, dropout=dropout)
            )
        
        # Temporal attention layers
        self.temporal_attentions = nn.ModuleList()
        for _ in range(num_layers):
            self.temporal_attentions.append(
                TemporalAttention(hidden_dim, num_heads, dropout=dropout)
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, num_nodes, features]
            edge_index: Edge connectivity
            
        Returns:
            torch.Tensor: Output tensor
        """
        batch_size, seq_len, num_nodes, features = x.shape
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply attention layers
        for spatial_attn, temporal_attn in zip(self.spatial_attentions, self.temporal_attentions):
            # Spatial attention
            x = spatial_attn(x, edge_index)
            
            # Temporal attention
            x = temporal_attn(x)
        
        # Take the last time step for prediction
        x = x[:, -1, :, :]  # [batch, num_nodes, hidden_dim]
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class SpatialAttention(nn.Module):
    """Spatial attention mechanism."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        """Initialize spatial attention.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, num_nodes, features]
            edge_index: Edge connectivity
            
        Returns:
            torch.Tensor: Output tensor
        """
        batch_size, seq_len, num_nodes, features = x.shape
        
        # Reshape for attention computation
        x = x.view(-1, num_nodes, features)  # [batch * seq_len, num_nodes, features]
        
        # Compute attention
        Q = self.query(x)  # [batch * seq_len, num_nodes, hidden_dim]
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(-1, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(-1, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(-1, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply edge mask (only attend to connected nodes)
        mask = torch.zeros(num_nodes, num_nodes, device=x.device)
        mask[edge_index[0], edge_index[1]] = 1
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, num_nodes, num_nodes]
        scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # [batch * seq_len, num_heads, num_nodes, head_dim]
        out = out.transpose(1, 2).contiguous().view(-1, num_nodes, features)
        
        # Reshape back
        out = out.view(batch_size, seq_len, num_nodes, features)
        
        return out


class TemporalAttention(nn.Module):
    """Temporal attention mechanism."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        """Initialize temporal attention.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, num_nodes, features]
            
        Returns:
            torch.Tensor: Output tensor
        """
        batch_size, seq_len, num_nodes, features = x.shape
        
        # Reshape for attention computation
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch, num_nodes, seq_len, features]
        x = x.view(-1, seq_len, features)  # [batch * num_nodes, seq_len, features]
        
        # Compute attention
        Q = self.query(x)  # [batch * num_nodes, seq_len, hidden_dim]
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(-1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(-1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(-1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # [batch * num_nodes, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2).contiguous().view(-1, seq_len, features)
        
        # Reshape back
        out = out.view(batch_size, num_nodes, seq_len, features)
        out = out.permute(0, 2, 1, 3).contiguous()  # [batch, seq_len, num_nodes, features]
        
        return out
