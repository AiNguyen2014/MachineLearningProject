"""
Visualization and evaluation utilities
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix
)
import plotly.graph_objects as go
import plotly.express as px


def purity_score(y_true, y_pred):
    """
    Calculate purity score
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted cluster labels
        
    Returns:
    --------
    purity : float
        Purity score (0-1)
    """
    contingency = pd.crosstab(y_true, y_pred)
    return np.sum(np.max(contingency.values, axis=0)) / np.sum(contingency.values)


def calculate_all_metrics(X, labels, y_true=None):
    """
    Calculate all clustering metrics
    
    Parameters:
    -----------
    X : numpy array
        Feature matrix
    labels : numpy array
        Cluster labels
    y_true : numpy array, optional
        True labels for external validation
        
    Returns:
    --------
    metrics : dict
        Dictionary of all metrics
    """
    metrics = {}
    
    # Internal validation
    metrics['silhouette'] = silhouette_score(X, labels)
    
    # External validation (if true labels provided)
    if y_true is not None:
        metrics['ari'] = adjusted_rand_score(y_true, labels)
        metrics['nmi'] = normalized_mutual_info_score(y_true, labels)
        metrics['purity'] = purity_score(y_true, labels)
    
    return metrics


def plot_clusters_2d(X_2d, labels, title="Clustering Results", colors=None, ax=None):
    """
    Plot 2D scatter plot of clusters
    
    Parameters:
    -----------
    X_2d : numpy array
        2D projection of data
    labels : numpy array
        Cluster labels
    title : str
        Plot title
    colors : list, optional
        Custom colors for clusters
    ax : matplotlib axis, optional
        Axis to plot on
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    if colors is None:
        colors = ['#2ecc71', '#9b59b6', '#e74c3c', '#3498db', '#f39c12']
    
    for i in np.unique(labels):
        mask = labels == i
        ax.scatter(
            X_2d[mask, 0], 
            X_2d[mask, 1],
            c=colors[i % len(colors)],
            s=80,
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5,
            label=f'Cluster {i}'
        )
    
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_clusters_interactive(X_2d, labels, title="Clustering Results"):
    """
    Create interactive Plotly scatter plot
    
    Parameters:
    -----------
    X_2d : numpy array
        2D projection of data
    labels : numpy array
        Cluster labels
    title : str
        Plot title
        
    Returns:
    --------
    fig : plotly figure
    """
    df_plot = pd.DataFrame({
        'PC1': X_2d[:, 0],
        'PC2': X_2d[:, 1],
        'Cluster': labels.astype(str),
        'Sample': range(len(labels))
    })
    
    fig = px.scatter(
        df_plot,
        x='PC1',
        y='PC2',
        color='Cluster',
        hover_data=['Sample'],
        title=title,
        color_discrete_sequence=['#2ecc71', '#9b59b6', '#e74c3c', '#3498db']
    )
    
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
    fig.update_layout(
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        font=dict(size=12),
        plot_bgcolor='white',
        showlegend=True
    )
    
    return fig


def plot_comparison_grid(X_2d, results_dict, metrics_dict, figsize=(16, 12)):
    """
    Plot comparison grid of all clustering methods
    
    Parameters:
    -----------
    X_2d : numpy array
        2D projection of data
    results_dict : dict
        Dictionary with algorithm names and labels
    metrics_dict : dict
        Dictionary with algorithm names and metrics
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib figure
    """
    n_methods = len(results_dict)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    
    colors = ['#2ecc71', '#9b59b6']
    
    for idx, (name, labels) in enumerate(results_dict.items()):
        metrics = metrics_dict[name]
        
        # Create title with metrics
        title = f"{name}\n"
        title += f"Sil={metrics['silhouette']:.3f}"
        if 'ari' in metrics:
            title += f" | ARI={metrics['ari']:.3f}"
        if 'purity' in metrics:
            title += f" | Purity={metrics['purity']:.3f}"
        
        # Highlight ensemble
        if 'ensemble' in name.lower():
            for spine in axes[idx].spines.values():
                spine.set_edgecolor('gold')
                spine.set_linewidth(3)
        
        plot_clusters_2d(X_2d, labels, title=title, colors=colors, ax=axes[idx])
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics_dict):
    """
    Plot bar chart comparing metrics across methods
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with algorithm names and metrics
        
    Returns:
    --------
    fig : matplotlib figure
    """
    # Prepare data
    methods = list(metrics_dict.keys())
    metric_names = list(metrics_dict[methods[0]].keys())
    
    data = {metric: [metrics_dict[m].get(metric, 0) for m in methods] 
            for metric in metric_names}
    
    # Create subplots
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    
    for idx, metric in enumerate(metric_names):
        axes[idx].bar(methods, data[metric], color=colors)
        axes[idx].set_title(metric.upper(), fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Score', fontsize=12)
        axes[idx].set_ylim(0, 1)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    title : str
        Plot title
        
    Returns:
    --------
    fig : matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        square=True,
        ax=ax
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    return fig


def create_metrics_table(metrics_dict):
    """
    Create a formatted DataFrame of metrics
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with algorithm names and metrics
        
    Returns:
    --------
    df : pandas DataFrame
        Formatted metrics table
    """
    df = pd.DataFrame(metrics_dict).T
    df = df.round(4)
    return df


def plot_radar_chart(metrics_dict):
    """
    Create radar chart comparing methods across metrics
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with algorithm names and metrics
        
    Returns:
    --------
    fig : plotly figure
    """
    methods = list(metrics_dict.keys())
    
    # Get metric names (excluding silhouette as it can be negative)
    metric_names = [k for k in metrics_dict[methods[0]].keys() 
                   if k in ['ari', 'nmi', 'purity']]
    
    fig = go.Figure()
    
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    
    for idx, method in enumerate(methods):
        values = [metrics_dict[method].get(m, 0) for m in metric_names]
        values.append(values[0])  # Close the radar chart
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_names + [metric_names[0]],
            fill='toself',
            name=method,
            line_color=colors[idx % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="Metrics Comparison Across Methods",
        font=dict(size=12)
    )
    
    return fig
