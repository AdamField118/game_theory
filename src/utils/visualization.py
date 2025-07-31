"""
Visualization utilities for evolutionary game theory simulations.
Creates plots matching Figure 1 from Olson et al. (2022).
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Union, Optional


def create_figure_1_plot(
    strategy_history: Union[np.ndarray, jnp.ndarray],
    delta: float,
    epsilon: float,
    output_dir: str = "plots",
    figsize: tuple = (10, 8),
    dpi: int = 150
) -> str:
    """
    Create a plot matching Figure 1 from the paper.
    
    Args:
        strategy_history: Array of shape (time_steps, n_vertices) with strategy evolution
        delta: Tie bonus parameter (δ)
        epsilon: Winning bonus parameter (ε)
        output_dir: Directory to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        
    Returns:
        Path to saved figure
    """
    # Convert to numpy if needed
    if isinstance(strategy_history, jnp.ndarray):
        strategy_history = np.array(strategy_history)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Define colors for the three strategies (Rock, Paper, Scissors)
    # Using colors that match the paper's aesthetic
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    cmap_colors = [(0, colors[0]), (0.5, colors[1]), (1, colors[2])]
    n_bins = 3
    cmap = mcolors.LinearSegmentedColormap.from_list('strategies', cmap_colors, N=n_bins)
    
    # Create the strategy evolution plot
    im = ax.imshow(
        strategy_history,
        aspect='auto',
        cmap=cmap,
        vmin=0,
        vmax=2,
        interpolation='nearest',
        origin='upper'
    )
    
    # Set labels and title
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Time', fontsize=12)
    
    # Determine community type for title
    if epsilon < 2 * delta:
        community_type = "Stationary Communities"
    elif epsilon > 2 * delta:
        community_type = "Drifting Communities"
    else:
        community_type = "Critical Communities"
    
    title = f'{community_type}\n(δ = {delta}, ε = {epsilon})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar with strategy labels
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_ticklabels(['Rock', 'Paper', 'Scissors'])
    cbar.set_label('Strategy', fontsize=12)
    
    # Set axis ticks to match paper style
    n_time, n_vertices = strategy_history.shape
    
    # X-axis (position)
    x_ticks = np.linspace(0, n_vertices-1, min(6, n_vertices//50 + 1))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{int(x)}' for x in x_ticks])
    
    # Y-axis (time)
    y_ticks = np.linspace(0, n_time-1, min(6, n_time//50 + 1))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{int(y)}' for y in y_ticks])
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Save figure
    output_path = Path(output_dir)
    filename = f"evolution_d{delta}_e{epsilon}.png"
    filepath = output_path / filename
    
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(filepath)


def create_community_analysis_plot(
    strategy_history: Union[np.ndarray, jnp.ndarray],
    delta: float,
    epsilon: float,
    output_dir: str = "plots"
) -> str:
    """
    Create additional analysis plots showing community statistics.
    
    Args:
        strategy_history: Array of shape (time_steps, n_vertices)
        delta: Tie bonus parameter
        epsilon: Winning bonus parameter
        output_dir: Output directory
        
    Returns:
        Path to saved analysis figure
    """
    if isinstance(strategy_history, jnp.ndarray):
        strategy_history = np.array(strategy_history)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Strategy frequencies over time
    ax1 = axes[0, 0]
    strategy_counts = np.array([
        np.sum(strategy_history == s, axis=1) for s in range(3)
    ]).T
    
    for s, (label, color) in enumerate(zip(['Rock', 'Paper', 'Scissors'], 
                                         ['blue', 'orange', 'green'])):
        ax1.plot(strategy_counts[:, s], label=label, color=color, alpha=0.7)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Strategy Count')
    ax1.set_title('Strategy Frequencies Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final state
    ax2 = axes[0, 1]
    final_state = strategy_history[-1]
    ax2.plot(final_state, 'o-', markersize=3, linewidth=1)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Strategy')
    ax2.set_title('Final Strategy Distribution')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Rock', 'Paper', 'Scissors'])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Community size distribution at final time
    ax3 = axes[1, 0]
    community_sizes = _calculate_community_sizes(final_state)
    if len(community_sizes) > 0:
        ax3.hist(community_sizes, bins=min(20, len(community_sizes)), 
                alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Community Size')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Community Size Distribution')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Diversity over time (Shannon entropy)
    ax4 = axes[1, 1]
    diversities = []
    for t in range(len(strategy_history)):
        counts = np.bincount(strategy_history[t], minlength=3)
        probs = counts / np.sum(counts)
        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        diversities.append(entropy)
    
    ax4.plot(diversities, color='purple', alpha=0.7)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Shannon Diversity')
    ax4.set_title('Strategy Diversity Over Time')
    ax4.grid(True, alpha=0.3)
    
    # Save figure to analysis subdirectory
    analysis_dir = Path(output_dir) / "analysis"
    filename = f"analysis_d{delta}_e{epsilon}.png"
    filepath = analysis_dir / filename
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(filepath)


def _calculate_community_sizes(strategies: np.ndarray) -> list:
    """
    Calculate sizes of communities (contiguous regions of same strategy).
    
    Args:
        strategies: 1D array of strategies
        
    Returns:
        List of community sizes
    """
    if len(strategies) == 0:
        return []
    
    # Handle periodic boundary conditions by extending array
    extended = np.concatenate([strategies, strategies, strategies])
    n = len(strategies)
    
    communities = []
    current_strategy = extended[n]
    current_size = 1
    
    for i in range(n + 1, 2 * n):
        if extended[i] == current_strategy:
            current_size += 1
        else:
            if current_size >= 2:  # Only count communities of size >= 2
                communities.append(current_size)
            current_strategy = extended[i]
            current_size = 1
    
    # Don't double count due to periodic boundaries
    unique_communities = []
    total_size = sum(communities)
    if total_size <= n:  # Avoid double counting
        unique_communities = communities
    else:
        # More sophisticated deduplication needed for complex cases
        unique_communities = communities[:len(communities)//2]
    
    return unique_communities


def create_comparison_plot(
    strategy_histories: list,
    parameters: list,
    output_dir: str = "plots"
) -> str:
    """
    Create a comparison plot of multiple parameter sets.
    
    Args:
        strategy_histories: List of strategy history arrays
        parameters: List of (delta, epsilon) parameter tuples
        output_dir: Output directory
        
    Returns:
        Path to saved comparison figure
    """
    n_plots = len(strategy_histories)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    # Fix axes handling for different subplot configurations
    if n_plots == 1:
        axes = [axes]
    elif rows == 1 and cols > 1:
        # axes is already a 1D array, don't wrap in list
        pass  
    elif rows > 1 and cols == 1:
        # axes is already a 1D array, don't wrap in list
        pass
    else:
        # For 2D array of subplots, flatten it
        axes = axes.flatten()
    
    # Colors for strategies
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    cmap = mcolors.ListedColormap(colors)
    
    for i, (history, (delta, epsilon)) in enumerate(zip(strategy_histories, parameters)):
        if isinstance(history, jnp.ndarray):
            history = np.array(history)
        
        ax = axes[i] if n_plots > 1 else axes[0]
        im = ax.imshow(history, aspect='auto', cmap=cmap, vmin=0, vmax=2,
                      interpolation='nearest', origin='upper')
        
        # Determine community type
        if epsilon < 2 * delta:
            community_type = "Stationary"
        elif epsilon > 2 * delta:
            community_type = "Drifting"
        else:
            community_type = "Critical"
        
        ax.set_title(f'{community_type}\n(δ={delta}, ε={epsilon})')
        ax.set_xlabel('Position')
        ax.set_ylabel('Time')
    
    # Hide unused subplots
    if n_plots > 1:
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
    
    # Save figure to analysis subdirectory
    analysis_dir = Path(output_dir) / "analysis"
    filename = "comparison_plot.png"
    filepath = analysis_dir / filename
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(filepath)
