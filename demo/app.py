"""Streamlit demo for traffic network analysis."""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils import get_device, set_seed
from utils.config import Config, get_default_config
from data import TrafficDataModule, create_traffic_network, visualize_traffic_network
from models import STGCN, DCRNN, GMAN
from eval import TrafficMetrics, TrafficEvaluator


# Page configuration
st.set_page_config(
    page_title="Traffic Network Analysis",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache traffic data."""
    set_seed(42)
    
    # Create data module
    data_module = TrafficDataModule(
        data_dir="data",
        sequence_length=12,
        prediction_horizon=3,
        batch_size=32
    )
    
    data_module.setup()
    
    return data_module


@st.cache_data
def create_sample_network():
    """Create a sample traffic network for visualization."""
    return create_traffic_network(num_nodes=15, seed=42)


def create_model(model_name: str, num_nodes: int, in_channels: int, out_channels: int):
    """Create a model instance."""
    if model_name == "STGCN":
        return STGCN(num_nodes, in_channels, out_channels, hidden_dim=32, num_layers=2)
    elif model_name == "DCRNN":
        return DCRNN(num_nodes, in_channels, out_channels, hidden_dim=32, num_layers=2)
    elif model_name == "GMAN":
        return GMAN(num_nodes, in_channels, out_channels, hidden_dim=32, num_layers=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def plot_traffic_network(G: nx.DiGraph):
    """Create an interactive plot of the traffic network."""
    pos = nx.spring_layout(G, seed=42)
    
    # Extract node and edge information
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge information
        edge_info.append({
            'from': edge[0],
            'to': edge[1],
            'length': G[edge[0]][edge[1]]['length'],
            'capacity': G[edge[0]][edge[1]]['capacity'],
            'speed_limit': G[edge[0]][edge[1]]['speed_limit']
        })
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[f"Node {i}" for i in G.nodes()],
        textposition="middle center",
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Traffic Network Graph',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Interactive traffic network visualization",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color='gray', size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig


def plot_traffic_forecast(predictions: torch.Tensor, targets: torch.Tensor, 
                         node_idx: int = 0, feature_idx: int = 0):
    """Plot traffic forecasting results."""
    pred_np = predictions[:, node_idx, feature_idx].detach().cpu().numpy()
    target_np = targets[:, node_idx, feature_idx].detach().cpu().numpy()
    
    fig = go.Figure()
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        y=pred_np,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red', width=2),
        marker=dict(size=6)
    ))
    
    # Add target line
    fig.add_trace(go.Scatter(
        y=target_np,
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f'Traffic Forecast - Node {node_idx}, Feature {feature_idx}',
        xaxis_title='Time Steps',
        yaxis_title='Value',
        hovermode='x unified',
        legend=dict(x=0, y=1)
    )
    
    return fig


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🚦 Traffic Network Analysis</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["STGCN", "DCRNN", "GMAN"],
        help="Choose the Graph Neural Network model for traffic forecasting"
    )
    
    # Data parameters
    st.sidebar.subheader("Data Parameters")
    sequence_length = st.sidebar.slider("Sequence Length", 6, 24, 12, 
                                      help="Number of historical time steps")
    prediction_horizon = st.sidebar.slider("Prediction Horizon", 1, 12, 3,
                                         help="Number of future time steps to predict")
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    hidden_dim = st.sidebar.slider("Hidden Dimension", 16, 128, 64)
    num_layers = st.sidebar.slider("Number of Layers", 1, 4, 2)
    dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.1)
    
    # Load data
    with st.spinner("Loading traffic data..."):
        data_module = load_data()
        graph_data = data_module.get_graph_data()
        sample_network = create_sample_network()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Network Overview", "Model Training", "Forecasting", "Analysis"])
    
    with tab1:
        st.header("Traffic Network Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Network visualization
            fig = plot_traffic_network(sample_network)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Network statistics
            st.subheader("Network Statistics")
            
            metrics = {
                "Number of Nodes": len(sample_network.nodes()),
                "Number of Edges": len(sample_network.edges()),
                "Average Degree": np.mean([d for n, d in sample_network.degree()]),
                "Density": nx.density(sample_network),
                "Average Path Length": nx.average_shortest_path_length(sample_network.to_undirected())
            }
            
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.3f}")
            
            # Edge information
            st.subheader("Road Information")
            edges_df = pd.DataFrame([
                {
                    "From": edge[0],
                    "To": edge[1],
                    "Length": f"{sample_network[edge[0]][edge[1]]['length']:.2f}",
                    "Capacity": f"{sample_network[edge[0]][edge[1]]['capacity']:.0f}",
                    "Speed Limit": f"{sample_network[edge[0]][edge[1]]['speed_limit']:.0f}"
                }
                for edge in sample_network.edges()
            ])
            st.dataframe(edges_df, use_container_width=True)
    
    with tab2:
        st.header("Model Training")
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                # Create model
                in_channels = 4  # traffic_volume, speed, occupancy, incidents
                out_channels = 4
                
                model = create_model(model_name, graph_data.num_nodes, in_channels, out_channels)
                
                # Get data loaders
                train_loader = data_module.train_dataloader()
                val_loader = data_module.val_dataloader()
                test_loader = data_module.test_dataloader()
                
                # Simple training loop (for demo purposes)
                device = get_device()
                model.to(device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = torch.nn.MSELoss()
                
                train_losses = []
                val_losses = []
                
                for epoch in range(10):  # Reduced epochs for demo
                    # Training
                    model.train()
                    train_loss = 0.0
                    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
                        if batch_idx >= 5:  # Limit batches for demo
                            break
                        
                        input_seq = input_seq.to(device)
                        target_seq = target_seq.to(device)
                        
                        optimizer.zero_grad()
                        predictions = model(input_seq, graph_data.edge_index.to(device))
                        loss = criterion(predictions, target_seq[:, -1, :, :])
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                    
                    train_losses.append(train_loss / 5)
                    
                    # Validation
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch_idx, (input_seq, target_seq) in enumerate(val_loader):
                            if batch_idx >= 3:  # Limit batches for demo
                                break
                            
                            input_seq = input_seq.to(device)
                            target_seq = target_seq.to(device)
                            
                            predictions = model(input_seq, graph_data.edge_index.to(device))
                            loss = criterion(predictions, target_seq[:, -1, :, :])
                            val_loss += loss.item()
                    
                    val_losses.append(val_loss / 3)
                
                # Plot training curves
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=train_losses,
                    mode='lines+markers',
                    name='Training Loss',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    y=val_losses,
                    mode='lines+markers',
                    name='Validation Loss',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title='Training Progress',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Model info
                st.success(f"Model trained successfully!")
                st.info(f"Model: {model_name}")
                st.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
                st.info(f"Final Training Loss: {train_losses[-1]:.4f}")
                st.info(f"Final Validation Loss: {val_losses[-1]:.4f}")
    
    with tab3:
        st.header("Traffic Forecasting")
        
        # Generate sample predictions
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating traffic forecast..."):
                # Create a simple model for demonstration
                in_channels = 4
                out_channels = 4
                
                model = create_model(model_name, graph_data.num_nodes, in_channels, out_channels)
                device = get_device()
                model.to(device)
                
                # Get a sample from test data
                test_loader = data_module.test_dataloader()
                sample_input, sample_target = next(iter(test_loader))
                
                # Generate predictions
                model.eval()
                with torch.no_grad():
                    sample_input = sample_input.to(device)
                    predictions = model(sample_input, graph_data.edge_index.to(device))
                
                # Plot results
                col1, col2 = st.columns(2)
                
                with col1:
                    # Select node and feature
                    node_idx = st.selectbox("Select Node", range(graph_data.num_nodes))
                    feature_names = ["Traffic Volume", "Speed", "Occupancy", "Incidents"]
                    feature_idx = st.selectbox("Select Feature", range(4), 
                                             format_func=lambda x: feature_names[x])
                    
                    # Plot forecast
                    fig = plot_traffic_forecast(predictions, sample_target, node_idx, feature_idx)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Metrics
                    st.subheader("Forecasting Metrics")
                    
                    metrics = TrafficMetrics()
                    mae = metrics.mae(predictions, sample_target[:, -1, :, :])
                    rmse = metrics.rmse(predictions, sample_target[:, -1, :, :])
                    mape = metrics.mape(predictions, sample_target[:, -1, :, :])
                    
                    st.metric("MAE", f"{mae:.4f}")
                    st.metric("RMSE", f"{rmse:.4f}")
                    st.metric("MAPE", f"{mape:.2f}%")
                    
                    # Feature importance (simplified)
                    st.subheader("Feature Importance")
                    feature_importance = torch.abs(predictions - sample_target[:, -1, :, :]).mean(dim=(0, 1))
                    
                    fig = px.bar(
                        x=feature_names,
                        y=feature_importance.detach().cpu().numpy(),
                        title="Average Prediction Error by Feature"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Traffic Analysis")
        
        # Network analysis
        st.subheader("Network Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Centrality measures
            st.write("**Node Centrality Measures**")
            
            # Betweenness centrality
            betweenness = nx.betweenness_centrality(sample_network)
            betweenness_df = pd.DataFrame([
                {"Node": node, "Betweenness": value}
                for node, value in betweenness.items()
            ]).sort_values("Betweenness", ascending=False)
            
            st.dataframe(betweenness_df.head(10), use_container_width=True)
        
        with col2:
            # Edge centrality
            st.write("**Edge Betweenness Centrality**")
            
            edge_betweenness = nx.edge_betweenness_centrality(sample_network)
            edge_betweenness_df = pd.DataFrame([
                {"From": edge[0], "To": edge[1], "Centrality": value}
                for edge, value in edge_betweenness.items()
            ]).sort_values("Centrality", ascending=False)
            
            st.dataframe(edge_betweenness_df.head(10), use_container_width=True)
        
        # Traffic flow analysis
        st.subheader("Traffic Flow Analysis")
        
        # Generate synthetic traffic data
        np.random.seed(42)
        time_steps = 24
        traffic_data = np.random.uniform(10, 100, (time_steps, graph_data.num_nodes))
        
        # Add temporal patterns
        rush_hours = [7, 8, 9, 17, 18, 19]
        for hour in rush_hours:
            traffic_data[hour] *= 1.5
        
        # Create heatmap
        fig = px.imshow(
            traffic_data,
            labels=dict(x="Node", y="Hour", color="Traffic Volume"),
            title="Traffic Volume Heatmap (24 Hours)",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Traffic statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Peak Hour", "8:00 AM", "7:00 AM")
        
        with col2:
            st.metric("Average Volume", f"{traffic_data.mean():.1f}", "vehicles/hour")
        
        with col3:
            st.metric("Peak Volume", f"{traffic_data.max():.1f}", "vehicles/hour")


if __name__ == "__main__":
    main()
