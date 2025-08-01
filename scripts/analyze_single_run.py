#!/usr/bin/env python3
"""
Simple analysis script for a single simulation run.
Analyzes the most recent simulation output or a specified file.
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.io import (
    load_results, 
    load_zero_sum_results, 
    _calculate_shannon_diversity, 
    _calculate_community_sizes_proper
)
from utils.visualization import create_zero_sum_analysis_plot, create_community_analysis_plot


def find_latest_simulation():
    """Find the most recent simulation file."""
    data_dir = Path("plots/data")
    
    if not data_dir.exists():
        print("No plots/data directory found. Run a simulation first.")
        return None
    
    # Find all simulation files
    wealth_files = list(data_dir.glob("simulation_d*.npz"))
    zero_sum_files = list(data_dir.glob("zero_sum_simulation_*.npz"))
    
    all_files = wealth_files + zero_sum_files
    
    if not all_files:
        print("No simulation files found in plots/data/")
        return None
    
    # Get the most recent file
    latest_file = max(all_files, key=lambda x: x.stat().st_mtime)
    
    return latest_file


def analyze_wealth_mediated(data, filepath):
    """Analyze wealth-mediated simulation results."""
    strategy_history = data["strategy_history"]
    parameters = data["parameters"]
    
    print("="*60)
    print("WEALTH-MEDIATED SIMULATION ANALYSIS")
    print("="*60)
    print(f"File: {filepath.name}")
    print(f"δ (tie bonus): {parameters['delta']}")
    print(f"ε (winning bonus): {parameters['epsilon']}")
    print(f"Vertices: {parameters['vertices']}")
    print(f"Time steps: {parameters['time_steps']}")
    
    # Determine community type
    delta = parameters['delta']
    epsilon = parameters['epsilon']
    if epsilon < 2 * delta:
        community_type = "Stationary (ε < 2δ)"
    elif epsilon > 2 * delta:
        community_type = "Drifting (ε > 2δ)"
    else:
        community_type = "Critical (ε = 2δ)"
    
    print(f"Community type: {community_type}")
    
    # Final state analysis
    final_state = strategy_history[-1]
    strategy_counts = np.bincount(final_state, minlength=3)
    strategy_names = ['Rock', 'Paper', 'Scissors']
    
    print("\nFinal Strategy Distribution:")
    for i, (name, count) in enumerate(zip(strategy_names, strategy_counts)):
        percentage = count / len(final_state) * 100
        print(f"  {name}: {count} agents ({percentage:.1f}%)")
    
    # Diversity analysis using imported function
    initial_diversity = _calculate_shannon_diversity(strategy_history[0])
    final_diversity = _calculate_shannon_diversity(final_state)
    
    print(f"\nDiversity Evolution:")
    print(f"  Initial Shannon diversity: {initial_diversity:.3f}")
    print(f"  Final Shannon diversity: {final_diversity:.3f}")
    print(f"  Change: {final_diversity - initial_diversity:+.3f}")
    
    # Community analysis using imported function
    final_communities = _calculate_community_sizes_proper(final_state)
    
    print(f"\nCommunity Structure:")
    print(f"  Number of communities: {len(final_communities)}")
    if final_communities:
        print(f"  Average size: {np.mean(final_communities):.1f} ± {np.std(final_communities):.1f}")
        print(f"  Size range: [{min(final_communities)}, {max(final_communities)}]")
    
    # Create analysis plot
    analysis_path = create_community_analysis_plot(
        strategy_history, delta, epsilon, "plots"
    )
    print(f"\nAnalysis plot created: {Path(analysis_path).name}")
    
    return analysis_path


def analyze_zero_sum(data, filepath):
    """Analyze zero-sum simulation results."""
    strategy_history = data["strategy_history"]
    bank_history = data.get("bank_history")
    parameters = data["parameters"]
    
    print("="*60)
    print("ZERO-SUM SIMULATION ANALYSIS")
    print("="*60)
    print(f"File: {filepath.name}")
    print(f"Vertices: {parameters['vertices']}")
    print(f"Time steps: {parameters['time_steps']}")
    print(f"Initial pattern: {parameters['initial_pattern']}")
    print(f"Initial bank value: {parameters['initial_bank_value']}")
    
    # Final state analysis
    final_state = strategy_history[-1]
    strategy_counts = np.bincount(final_state, minlength=3)
    strategy_names = ['Rock', 'Paper', 'Scissors']
    
    print("\nFinal Strategy Distribution:")
    for i, (name, count) in enumerate(zip(strategy_names, strategy_counts)):
        percentage = count / len(final_state) * 100
        print(f"  {name}: {count} agents ({percentage:.1f}%)")
    
    # Wealth analysis
    if bank_history is not None:
        initial_total_wealth = np.sum(bank_history[0])
        final_total_wealth = np.sum(bank_history[-1])
        max_deviation = np.max(np.abs(np.sum(bank_history, axis=1) - initial_total_wealth))
        
        print(f"\nWealth Conservation Analysis:")
        print(f"  Initial total wealth: {initial_total_wealth:.2f}")
        print(f"  Final total wealth: {final_total_wealth:.2f}")
        print(f"  Max deviation: {max_deviation:.2e}")
        print(f"  Wealth conserved: {max_deviation < 1e-10}")
        
        # Wealth distribution
        final_wealth = bank_history[-1]
        print(f"  Final wealth mean: {np.mean(final_wealth):.2f} ± {np.std(final_wealth):.2f}")
        print(f"  Final wealth range: [{np.min(final_wealth):.1f}, {np.max(final_wealth):.1f}]")
    
    # Diversity analysis using imported function
    initial_diversity = _calculate_shannon_diversity(strategy_history[0])
    final_diversity = _calculate_shannon_diversity(final_state)
    
    print(f"\nDiversity Evolution:")
    print(f"  Initial Shannon diversity: {initial_diversity:.3f}")
    print(f"  Final Shannon diversity: {final_diversity:.3f}")
    print(f"  Change: {final_diversity - initial_diversity:+.3f}")
    
    # Community analysis using imported function
    initial_communities = _calculate_community_sizes_proper(strategy_history[0])
    final_communities = _calculate_community_sizes_proper(final_state)
    
    print(f"\nCommunity Structure Evolution:")
    print(f"  Initial communities: {len(initial_communities)}")
    if initial_communities:
        print(f"    Average size: {np.mean(initial_communities):.1f} ± {np.std(initial_communities):.1f}")
    
    print(f"  Final communities: {len(final_communities)}")
    if final_communities:
        print(f"    Average size: {np.mean(final_communities):.1f} ± {np.std(final_communities):.1f}")
        print(f"    Size range: [{min(final_communities)}, {max(final_communities)}]")
    
    # Create analysis plot
    if bank_history is not None:
        analysis_path = create_zero_sum_analysis_plot(
            strategy_history, bank_history, "plots"
        )
        print(f"\nAnalysis plot created: {Path(analysis_path).name}")
        return analysis_path
    else:
        print("\nWarning: No bank history available for detailed zero-sum analysis")
        return None


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Analyze evolutionary game simulation results"
    )
    parser.add_argument("--file", "-f", type=str, default=None,
                       help="Specific file to analyze (default: most recent)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")
    
    args = parser.parse_args()
    
    # Find file to analyze
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return
    else:
        filepath = find_latest_simulation()
        if filepath is None:
            return
    
    print(f"Analyzing: {filepath}")
    
    # Load and analyze based on file type
    try:
        if "zero_sum" in filepath.name:
            data = load_zero_sum_results(str(filepath))
            analysis_path = analyze_zero_sum(data, filepath)
        else:
            data = load_results(str(filepath))
            analysis_path = analyze_wealth_mediated(data, filepath)
        
        print("="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()