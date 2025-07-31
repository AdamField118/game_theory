#!/usr/bin/env python3
"""
Generate all analysis plots and summaries from existing simulation data.
Run this after completing simulations to get the missing analysis files.
"""

import sys
from pathlib import Path
import glob
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.io import load_results, create_summary_report
from utils.visualization import create_community_analysis_plot, create_comparison_plot


def main():
    """Generate all missing analysis files."""
    print("="*60)
    print("GENERATING ANALYSIS PLOTS AND SUMMARIES")
    print("="*60)
    
    # Find all simulation data files
    data_dir = Path("plots/data")
    npz_files = list(data_dir.glob("simulation_*.npz"))
    
    if not npz_files:
        print("  No simulation data files found!")
        print(f"   Expected files in: {data_dir}")
        return
    
    print(f"Found {len(npz_files)} simulation files:")
    for f in npz_files:
        print(f"  - {f.name}")
    
    histories = []
    parameters_list = []
    
    # Process each simulation file
    for npz_file in npz_files:
        try:
            print(f"\nProcessing: {npz_file.name}")
            data = load_results(str(npz_file))
            
            strategy_history = data["strategy_history"]
            parameters = data["parameters"]
            
            delta = parameters["delta"]
            epsilon = parameters["epsilon"]
            
            # Generate community analysis plot
            analysis_path = create_community_analysis_plot(
                strategy_history, delta, epsilon, "plots"
            )
            print(f"    Analysis plot: plots/analysis/{Path(analysis_path).name}")
            
            # Generate summary report (if it doesn't exist)
            summary_path = Path("plots/analysis") / f"summary_d{delta}_e{epsilon}.txt"
            if not summary_path.exists():
                summary_file = create_summary_report(
                    strategy_history, parameters, "plots"
                )
                print(f"    Summary report: plots/analysis/{Path(summary_file).name}")
            else:
                print(f"  - Summary already exists: analysis/{summary_path.name}")
            
            # Store for comparison plots
            histories.append(strategy_history)
            parameters_list.append((delta, epsilon))
            
        except Exception as e:
            print(f"    Error processing {npz_file.name}: {e}")
    
    # Generate comparison plots if we have multiple simulations
    if len(histories) >= 2:
        print(f"\nGenerating comparison plots...")
        
        try:
            comparison_path = create_comparison_plot(histories, parameters_list, "plots")
            print(f"  Comparison plot: plots/analysis/{Path(comparison_path).name}")
        except Exception as e:
            print(f"  Error creating comparison plot: {e}")
        
        # Try to create Figure 1 reproduction
        try:
            fig1_path = create_figure_1_reproduction(histories, parameters_list, "plots")
            if fig1_path:
                print(f" Figure 1 reproduction: plots/analysis/{Path(fig1_path).name}")
        except Exception as e:
            print(f" Error creating Figure 1 reproduction: {e}")
    
    print("\n" + "="*60)
    print("ANALYSIS GENERATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    
    # List all output files in organized structure
    main_plots = list(Path("plots").glob("evolution_*.png"))
    analysis_plots = list(Path("plots/analysis").glob("*.png"))
    summary_files = list(Path("plots/analysis").glob("*.txt"))
    data_files = list(Path("plots/data").glob("*.npz"))
    
    if main_plots:
        print("\n Main Evolution Plots (plots/):")
        for f in sorted(main_plots):
            print(f"  {f.name}")
    
    if analysis_plots:
        print("\n Analysis Plots (plots/analysis/):")
        for f in sorted(analysis_plots):
            print(f"  {f.name}")
    
    if summary_files:
        print("\n Summary Reports (plots/analysis/):")
        for f in sorted(summary_files):
            print(f"  {f.name}")
    
    if data_files:
        print(f"\n Data Files (plots/data/): {len(data_files)} files")


def create_figure_1_reproduction(histories, parameters_list, output_dir="plots"):
    """
    Create a reproduction of Figure 1 with side-by-side plots.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    # Find stationary and drifting examples
    stationary_idx = None
    drifting_idx = None
    
    for i, (delta, epsilon) in enumerate(parameters_list):
        if epsilon < 2 * delta and stationary_idx is None:
            stationary_idx = i
        elif epsilon > 2 * delta and drifting_idx is None:
            drifting_idx = i
    
    if stationary_idx is None or drifting_idx is None:
        print(" Could not find both stationary and drifting examples for Figure 1 reproduction")
        return None
    
    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors for strategies
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    cmap = ListedColormap(colors)
    
    # Plot stationary communities
    stat_history = histories[stationary_idx]
    stat_params = parameters_list[stationary_idx]
    im1 = ax1.imshow(stat_history, aspect='auto', cmap=cmap, vmin=0, vmax=2,
                     interpolation='nearest', origin='upper')
    ax1.set_title(f'Stationary Communities\n(  = {stat_params[0]},   = {stat_params[1]})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('Time', fontsize=12)
    
    # Plot drifting communities  
    drift_history = histories[drifting_idx]
    drift_params = parameters_list[drifting_idx]
    im2 = ax2.imshow(drift_history, aspect='auto', cmap=cmap, vmin=0, vmax=2,
                     interpolation='nearest', origin='upper')
    ax2.set_title(f'Drifting Communities\n(  = {drift_params[0]},   = {drift_params[1]})', 
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
    
    # Save figure to analysis subdirectory
    analysis_dir = Path(output_dir) / "analysis"
    output_path = analysis_dir / "figure_1_reproduction.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(output_path)


if __name__ == "__main__":
    main()