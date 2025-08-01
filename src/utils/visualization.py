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

def create_zero_sum_analysis_plot(
    strategy_history: Union[np.ndarray, jnp.ndarray],
    bank_history: Union[np.ndarray, jnp.ndarray],
    output_dir: str = "plots",
    figsize: tuple = (20, 12)
) -> str:
    """
    Create comprehensive analysis plot specifically for zero-sum evolutionary dynamics.
    
    Args:
        strategy_history: Array of shape (time_steps, n_vertices) with strategy evolution
        bank_history: Array of shape (time_steps, n_vertices) with bank evolution
        output_dir: Output directory for plots
        figsize: Figure size
        
    Returns:
        Path to saved figure
    """
    if isinstance(strategy_history, jnp.ndarray):
        strategy_history = np.array(strategy_history)
    if isinstance(bank_history, jnp.ndarray):
        bank_history = np.array(bank_history)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[3, 2, 2, 2])
    
    # Colors for strategies
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    cmap = mcolors.ListedColormap(colors)
    
    # **MAIN STRATEGY EVOLUTION PLOT** (large, left side)
    ax_main = fig.add_subplot(gs[:2, 0])
    im = ax_main.imshow(strategy_history, aspect='auto', cmap=cmap, vmin=0, vmax=2,
                       interpolation='nearest', origin='upper')
    ax_main.set_title('Zero-Sum Deterministic Evolution', fontsize=16, fontweight='bold')
    ax_main.set_xlabel('Position', fontsize=12)
    ax_main.set_ylabel('Time', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_main, fraction=0.02, pad=0.02)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['Rock', 'Paper', 'Scissors'])
    cbar.set_label('Strategy', fontsize=12)
    
    # **WEALTH CONSERVATION PLOT** (top-right)
    ax_wealth = fig.add_subplot(gs[0, 1])
    total_wealth = np.sum(bank_history, axis=1)
    ax_wealth.plot(total_wealth, 'r-', linewidth=2, label='Total Wealth')
    ax_wealth.axhline(y=total_wealth[0], color='black', linestyle='--', alpha=0.7, 
                      label=f'Expected: {total_wealth[0]:.0f}')
    ax_wealth.set_xlabel('Time')
    ax_wealth.set_ylabel('Total System Wealth')
    ax_wealth.set_title('Wealth Conservation', fontsize=12, fontweight='bold')
    ax_wealth.legend()
    ax_wealth.grid(True, alpha=0.3)
    
    # **STRATEGY FREQUENCIES** (top-center-right)
    ax_freq = fig.add_subplot(gs[0, 2])
    strategy_counts = np.array([
        np.sum(strategy_history == s, axis=1) for s in range(3)
    ]).T
    
    for s, (label, color) in enumerate(zip(['Rock', 'Paper', 'Scissors'], colors)):
        ax_freq.plot(strategy_counts[:, s], label=label, color=color, alpha=0.8, linewidth=2)
    
    ax_freq.set_xlabel('Time')
    ax_freq.set_ylabel('Count')
    ax_freq.set_title('Strategy Frequencies', fontsize=12, fontweight='bold')
    ax_freq.legend()
    ax_freq.grid(True, alpha=0.3)
    
    # **WEALTH DISTRIBUTION** (top-far-right)
    ax_wealth_dist = fig.add_subplot(gs[0, 3])
    wealth_std = np.std(bank_history, axis=1)
    wealth_mean = np.mean(bank_history, axis=1)
    
    ax_wealth_dist.plot(wealth_mean, 'purple', linewidth=2, label='Mean Wealth')
    ax_wealth_dist.fill_between(range(len(wealth_mean)), 
                                wealth_mean - wealth_std, 
                                wealth_mean + wealth_std, 
                                alpha=0.3, color='purple', label='±1 Std Dev')
    ax_wealth_dist.set_xlabel('Time')
    ax_wealth_dist.set_ylabel('Wealth Statistics')
    ax_wealth_dist.set_title('Wealth Distribution', fontsize=12, fontweight='bold')
    ax_wealth_dist.legend()
    ax_wealth_dist.grid(True, alpha=0.3)
    
    # **INITIAL VS FINAL STATES** (middle row)
    
    # Initial state
    ax_initial = fig.add_subplot(gs[1, 1])
    initial_state = strategy_history[0]
    colors_discrete = [colors[i] for i in initial_state]
    ax_initial.scatter(range(len(initial_state)), initial_state, c=colors_discrete, s=2, alpha=0.8)
    ax_initial.set_xlabel('Position')
    ax_initial.set_ylabel('Strategy')
    ax_initial.set_title('Initial State', fontsize=12, fontweight='bold')
    ax_initial.set_yticks([0, 1, 2])
    ax_initial.set_yticklabels(['Rock', 'Paper', 'Scissors'])
    ax_initial.grid(True, alpha=0.3)
    
    # Final state
    ax_final = fig.add_subplot(gs[1, 2])
    final_state = strategy_history[-1]
    colors_discrete = [colors[i] for i in final_state]
    ax_final.scatter(range(len(final_state)), final_state, c=colors_discrete, s=2, alpha=0.8)
    ax_final.set_xlabel('Position')
    ax_final.set_ylabel('Strategy')
    ax_final.set_title('Final State', fontsize=12, fontweight='bold')
    ax_final.set_yticks([0, 1, 2])
    ax_final.set_yticklabels(['Rock', 'Paper', 'Scissors'])
    ax_final.grid(True, alpha=0.3)
    
    # Diversity over time
    ax_diversity = fig.add_subplot(gs[1, 3])
    diversities = []
    for t in range(len(strategy_history)):
        counts = np.bincount(strategy_history[t], minlength=3)
        probs = counts / np.sum(counts)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        diversities.append(entropy)
    
    ax_diversity.plot(diversities, color='darkgreen', alpha=0.8, linewidth=2)
    ax_diversity.set_xlabel('Time')
    ax_diversity.set_ylabel('Shannon Entropy')
    ax_diversity.set_title('Diversity Over Time', fontsize=12, fontweight='bold')
    ax_diversity.grid(True, alpha=0.3)
    
    # **COMMUNITY ANALYSIS** (bottom row, spanning most columns)
    ax_communities = fig.add_subplot(gs[2, :3])
    
    # Calculate community sizes over time
    community_counts = []
    avg_community_sizes = []
    
    sample_points = min(50, len(strategy_history))  # Sample up to 50 time points
    time_indices = np.linspace(0, len(strategy_history)-1, sample_points, dtype=int)
    
    for t in time_indices:
        communities = _calculate_community_sizes_zero_sum(strategy_history[t])
        community_counts.append(len(communities))
        avg_community_sizes.append(np.mean(communities) if communities else 0)
    
    ax_communities.plot(time_indices, community_counts, 'o-', label='Number of Communities', 
                       alpha=0.8, markersize=4)
    ax_communities_twin = ax_communities.twinx()
    ax_communities_twin.plot(time_indices, avg_community_sizes, 's-', color='red', 
                           label='Avg Community Size', alpha=0.8, markersize=4)
    
    ax_communities.set_xlabel('Time')
    ax_communities.set_ylabel('Number of Communities', color='blue')
    ax_communities_twin.set_ylabel('Average Community Size', color='red')
    ax_communities.set_title('Community Structure Evolution', fontsize=12, fontweight='bold')
    ax_communities.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax_communities.get_legend_handles_labels()
    lines2, labels2 = ax_communities_twin.get_legend_handles_labels()
    ax_communities.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # **FINAL WEALTH HISTOGRAM** (bottom-right)
    ax_wealth_hist = fig.add_subplot(gs[2, 3])
    final_wealth = bank_history[-1]
    ax_wealth_hist.hist(final_wealth, bins=30, alpha=0.7, color='gold', edgecolor='black')
    ax_wealth_hist.set_xlabel('Individual Wealth')
    ax_wealth_hist.set_ylabel('Frequency')
    ax_wealth_hist.set_title('Final Wealth Distribution', fontsize=12, fontweight='bold')
    ax_wealth_hist.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"zero_sum_analysis_{len(strategy_history[0])}agents_{len(strategy_history)}steps.png"
    filepath = Path(output_dir) / filename
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(filepath)

def create_zero_sum_evolution_plot(
    strategy_history: Union[np.ndarray, jnp.ndarray],
    output_dir: str = "plots",
    figsize: tuple = (12, 8)
) -> str:
    """
    Create a clean evolution plot showing just time vs position for zero-sum dynamics.
    This is the core visualization without all the extra analysis panels.
    """
    if isinstance(strategy_history, jnp.ndarray):
        strategy_history = np.array(strategy_history)
    
    # Create simple figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors for strategies - same as other plots for consistency
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue=Rock, Orange=Paper, Green=Scissors
    cmap = mcolors.ListedColormap(colors)
    
    # Main evolution plot
    im = ax.imshow(strategy_history, aspect='auto', cmap=cmap, vmin=0, vmax=2,
                   interpolation='nearest', origin='upper')
    
    ax.set_title('Zero-Sum Deterministic Evolution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Position', fontsize=14)
    ax.set_ylabel('Time', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.02, pad=0.04)
    cbar.set_ticklabels(['Rock', 'Paper', 'Scissors'])
    cbar.set_label('Strategy', fontsize=12)
    
    # Clean up the plot
    ax.grid(False)  # No grid for clean look
    
    plt.tight_layout()
    
    # Save with descriptive filename
    n_agents = strategy_history.shape[1]
    n_steps = strategy_history.shape[0]
    filename = f"zero_sum_evolution_{n_agents}agents_{n_steps}steps.png"
    filepath = Path(output_dir) / filename
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(filepath)


def _calculate_community_sizes_zero_sum(strategies: np.ndarray) -> list:
    """
    Calculate community sizes specifically optimized for zero-sum analysis.
    Handles periodic boundary conditions properly.
    """
    if len(strategies) == 0:
        return []
    
    communities = []
    n = len(strategies)
    
    # Handle periodic boundary by extending array
    extended_strategies = np.concatenate([strategies, strategies])
    
    i = 0
    while i < n:
        current_strategy = strategies[i]
        size = 1
        
        # Count forward
        j = i + 1
        while j < n and strategies[j % n] == current_strategy:
            size += 1
            j += 1
        
        # Check wraparound for the first community
        if i == 0:
            # Check backward from end
            k = n - 1
            while k > j - 1 and strategies[k] == current_strategy:
                size += 1
                k -= 1
            
            # If we wrapped around completely, it's one big community
            if size >= n:
                return [n]
        
        if size >= 2:  # Only count communities of size >= 2
            communities.append(min(size, n))  # Cap at total size
        
        i = j
    
    return communities


def create_wealth_conservation_diagnostic(
    bank_history: Union[np.ndarray, jnp.ndarray],
    output_dir: str = "plots"
) -> str:
    """
    Create detailed diagnostic plot for wealth conservation in zero-sum games.
    """
    if isinstance(bank_history, jnp.ndarray):
        bank_history = np.array(bank_history)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total wealth over time
    ax1 = axes[0, 0]
    total_wealth = np.sum(bank_history, axis=1)
    expected_total = total_wealth[0]
    deviation = total_wealth - expected_total
    
    ax1.plot(total_wealth, 'b-', linewidth=2, label='Actual Total')
    ax1.axhline(y=expected_total, color='red', linestyle='--', linewidth=2, 
                label=f'Expected: {expected_total:.0f}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Total Wealth')
    ax1.set_title('Wealth Conservation Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Deviation from expected
    ax2 = axes[0, 1]
    ax2.plot(deviation, 'r-', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Deviation from Expected')
    ax2.set_title(f'Max Deviation: {np.max(np.abs(deviation)):.2e}')
    ax2.grid(True, alpha=0.3)
    
    # Wealth distribution evolution
    ax3 = axes[1, 0]
    mean_wealth = np.mean(bank_history, axis=1)
    std_wealth = np.std(bank_history, axis=1)
    min_wealth = np.min(bank_history, axis=1)
    max_wealth = np.max(bank_history, axis=1)
    
    ax3.plot(mean_wealth, 'g-', linewidth=2, label='Mean')
    ax3.fill_between(range(len(mean_wealth)), 
                     mean_wealth - std_wealth, 
                     mean_wealth + std_wealth, 
                     alpha=0.3, color='green', label='±1 Std')
    ax3.plot(min_wealth, 'b--', alpha=0.7, label='Min')
    ax3.plot(max_wealth, 'r--', alpha=0.7, label='Max')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Wealth Statistics')
    ax3.set_title('Wealth Distribution Statistics')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Final wealth histogram
    ax4 = axes[1, 1]
    final_wealth = bank_history[-1]
    ax4.hist(final_wealth, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(x=np.mean(final_wealth), color='red', linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(final_wealth):.1f}')
    ax4.axvline(x=np.median(final_wealth), color='orange', linestyle='-', linewidth=2, 
                label=f'Median: {np.median(final_wealth):.1f}')
    ax4.set_xlabel('Individual Wealth')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Final Wealth Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filepath = Path(output_dir) / "wealth_conservation_diagnostic.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(filepath)