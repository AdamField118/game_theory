#!/usr/bin/env python3
"""
Create comparison plots from multiple simulation results.
This script is designed to run after batch simulations complete.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.visualization import create_comparison_plot
from utils.io import load_results


def find_latest_results(plots_dir="plots"):
    """Find the most recent simulation results."""
    data_dir = Path(plots_dir) / "data"
    
    # Find all .npz files
    npz_files = list(data_dir.glob("simulation_*.npz"))
    
    if not npz_files:
        print("No simulation results found!")
        return []
    
    # Sort by modification time (most recent first)
    npz_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"Found {len(npz_files)} simulation files:")
    for f in npz_files[:5]:  # Show first 5
        print(f"  {f.name}")
    
    return npz_files


def extract_parameters_from_filename(filename):
    """Extract delta and epsilon from filename."""
    # Expected format: simulation_d{delta}_e{epsilon}_{timestamp}.npz
    parts = filename.stem.split('_')
    try:
        delta_part = [p for p in parts if p.startswith('d')][0]
        epsilon_part = [p for p in parts if p.startswith('e')][0]
        
        delta = float(delta_part[1:])  # Remove 'd' prefix
        epsilon = float(epsilon_part[1:])  # Remove 'e' prefix
        
        return delta, epsilon
    except (IndexError, ValueError) as e:
        print(f"Could not parse parameters from {filename}: {e}")
        return None, None


def main():
    print("="*60)
    print("CREATING COMPARISON PLOTS")
    print("="*60)
    
    # Find simulation results
    results_files = find_latest_results()
    
    if len(results_files) < 2:
        print("Need at least 2 simulation results for comparison!")
        return
    
    # Load results and extract parameters
    histories = []
    parameters = []
    
    for file_path in results_files[:4]:  # Limit to 4 for visualization
        try:
            data = load_results(str(file_path))
            delta, epsilon = extract_parameters_from_filename(file_path)
            
            if delta is not None and epsilon is not None:
                histories.append(data["strategy_history"])
                parameters.append((delta, epsilon))
                print(f"Loaded: δ={delta}, ε={epsilon} from {file_path.name}")
            else:
                print(f"Skipping {file_path.name} - could not parse parameters")
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if len(histories) < 2:
        print("Could not load enough valid simulation results!")
        return
    
    # Create comparison plot
    print(f"Creating comparison plot with {len(histories)} simulations...")
    comparison_path = create_comparison_plot(histories, parameters, "plots")
    print(f"Comparison plot saved: {comparison_path}")
    
    # Create individual analysis plots for each simulation
    for i, (history, (delta, epsilon)) in enumerate(zip(histories, parameters)):
        from utils.visualization import create_community_analysis_plot
        analysis_path = create_community_analysis_plot(history, delta, epsilon, "plots")
        print(f"Analysis plot {i+1} saved: {analysis_path}")
    
    # Create summary figure combining everything
    create_figure_1_reproduction(histories, parameters)
    
    print("="*60)
    print("COMPARISON PLOTS COMPLETE")
    print("="*60)


def create_figure_1_reproduction(histories, parameters):
    """Create a reproduction of Figure 1 with side-by-side plots."""
    
    # Find stationary and drifting examples
    stationary_idx = None
    drifting_idx = None
    
    for i, (delta, epsilon) in enumerate(parameters):
        if epsilon < 2 * delta and stationary_idx is None:
            stationary_idx = i
        elif epsilon > 2 * delta and drifting_idx is None:
            drifting_idx = i
    
    if stationary_idx is None or drifting_idx is None:
        print("Could not find both stationary and drifting examples")
        return
    
    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors for strategies
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    
    # Plot stationary communities
    stat_history = histories[stationary_idx]
    stat_params = parameters[stationary_idx]
    im1 = ax1.imshow(stat_history, aspect='auto', cmap=cmap, vmin=0, vmax=2,
                     interpolation='nearest', origin='upper')
    ax1.set_title(f'Stationary Communities\n(δ = {stat_params[0]}, ε = {stat_params[1]})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('Time', fontsize=12)
    
    # Plot drifting communities  
    drift_history = histories[drifting_idx]
    drift_params = parameters[drifting_idx]
    im2 = ax2.imshow(drift_history, aspect='auto', cmap=cmap, vmin=0, vmax=2,
                     interpolation='nearest', origin='upper')
    ax2.set_title(f'Drifting Communities\n(δ = {drift_params[0]}, ε = {drift_params[1]})', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('Time', fontsize=12)
    
    # Add colorbar
    cbar = fig.colorbar(im1, ax=[ax1, ax2], ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cbar.set_ticklabels(['Rock', 'Paper', 'Scissors'])
    cbar.set_label('Strategy', fontsize=12)
    
    # Set main title
    fig.suptitle('Community Formation in Wealth-Mediated Thermodynamic Strategy Evolution', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("plots") / "figure_1_reproduction.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Figure 1 reproduction saved: {output_path}")


if __name__ == "__main__":
    main()
