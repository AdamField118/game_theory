"""
Input/Output utilities for evolutionary game simulations.
Handles file saving, directory setup, and data management.
"""

import json
import pickle
from pathlib import Path
from typing import Union, Any, Dict, Optional
import numpy as np
import jax.numpy as jnp
from datetime import datetime


def setup_output_dirs(base_dir: str = "plots") -> None:
    """
    Create necessary output directories.
    
    Args:
        base_dir: Base directory for outputs
    """
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    subdirs = ["data", "analysis", "logs"]
    for subdir in subdirs:
        (base_path / subdir).mkdir(exist_ok=True)
    
    print(f"Output directories created in: {base_path.absolute()}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load simulation results from file.
    
    Args:
        filepath: Path to saved results file
        
    Returns:
        Dictionary containing loaded data
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.npz':
        data = np.load(filepath, allow_pickle=True)
        return {
            "strategy_history": data["strategy_history"],
            "parameters": data["parameters"].item(),
            "metadata": data["metadata"].item()
        }
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def load_zero_sum_results(filepath: str) -> Dict[str, Any]:
    """
    Load zero-sum simulation results with proper handling of bank history.
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.npz':
        data = np.load(filepath, allow_pickle=True)
        result = {
            "strategy_history": data["strategy_history"],
            "parameters": data["parameters"].item(),
            "metadata": data["metadata"].item()
        }
        
        # Add bank history if present
        if "bank_history" in data:
            result["bank_history"] = data["bank_history"]
        
        # Add additional data if present
        if "additional_data" in data:
            result["additional_data"] = data["additional_data"].item()
            
        return result
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def save_log(
    message: str,
    log_type: str = "info",
    output_dir: str = "plots"
) -> None:
    """
    Save log message to file.
    
    Args:
        message: Log message
        log_type: Type of log (info, error, warning)
        output_dir: Output directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{log_type.upper()}] {message}\n"
    
    log_dir = Path(output_dir) / "logs"
    log_file = log_dir / "simulation.log"
    
    with open(log_file, 'a') as f:
        f.write(log_entry)


def save_results(
    strategy_history: Union[np.ndarray, jnp.ndarray],
    args: Any,
    output_dir: str = "plots",
    bank_history: Optional[Union[np.ndarray, jnp.ndarray]] = None,
    additional_data: Optional[Dict] = None
) -> str:
    """
    save_results function that handles both wealth-mediated and zero-sum data.
    """
    # Convert JAX arrays to numpy
    if isinstance(strategy_history, jnp.ndarray):
        strategy_history = np.array(strategy_history)
    if bank_history is not None and isinstance(bank_history, jnp.ndarray):
        bank_history = np.array(bank_history)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data dictionary
    data = {
        "strategy_history": strategy_history,
        "parameters": {
            "payoff_scheme": getattr(args, 'payoff_scheme', 'wealth_mediated'),
            "vertices": args.vertices,
            "time_steps": args.time_steps,
            "seed": args.seed
        },
        "metadata": {
            "timestamp": timestamp,
            "shape": strategy_history.shape,
            "n_strategies": len(np.unique(strategy_history)),
            "final_diversity": _calculate_shannon_diversity(strategy_history[-1])
        }
    }
    
    # Add game-specific parameters
    if hasattr(args, 'payoff_scheme') and args.payoff_scheme == 'zero_sum':
        data["parameters"].update({
            "initial_pattern": args.initial_pattern,
            "initial_bank_value": args.initial_bank_value,
            "stripe_size": getattr(args, 'stripe_size', None)
        })
        if bank_history is not None:
            data["bank_history"] = bank_history
        if additional_data:
            data["wealth_conservation"] = additional_data
    else:
        # Wealth-mediated parameters
        data["parameters"].update({
            "delta": args.delta,
            "epsilon": args.epsilon,
            "temperature": args.temperature,
            "beta": args.beta
        })
    
    # Save data
    output_path = Path(output_dir) / "data"
    
    # Create filename based on game type
    if hasattr(args, 'payoff_scheme') and args.payoff_scheme == 'zero_sum':
        filename = f"zero_sum_simulation_{args.vertices}agents_{timestamp}.npz"
    else:
        filename = f"simulation_d{args.delta}_e{args.epsilon}_{timestamp}.npz"
    
    filepath = output_path / filename
    
    # Save numpy format
    save_dict = {
        "strategy_history": strategy_history,
        "parameters": data["parameters"],
        "metadata": data["metadata"]
    }
    if bank_history is not None:
        save_dict["bank_history"] = bank_history
    if additional_data:
        save_dict["additional_data"] = additional_data
        
    np.savez_compressed(filepath, **save_dict)
    
    # Save parameters as JSON
    json_filename = filename.replace('.npz', '_params.json')
    json_filepath = output_path / json_filename
    with open(json_filepath, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_data = {
            "parameters": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v 
                         for k, v in data["parameters"].items()},
            "metadata": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else str(v) if not isinstance(v, (list, tuple)) else v
                        for k, v in data["metadata"].items()}
        }
        json.dump(json_data, f, indent=2)
    
    print(f"Data saved to: {filepath}")
    print(f"Parameters saved to: {json_filepath}")
    
    return str(filepath)


def create_summary_report(
    strategy_history: Union[np.ndarray, jnp.ndarray],
    parameters: Dict[str, Any],
    output_dir: str = "plots"
) -> str:
    """
    Create a summary report of the simulation results.
    
    Args:
        strategy_history: Strategy evolution data
        parameters: Simulation parameters
        output_dir: Output directory
        
    Returns:
        Path to summary report file
    """
    if isinstance(strategy_history, jnp.ndarray):
        strategy_history = np.array(strategy_history)
    
    # Calculate summary statistics
    final_state = strategy_history[-1]
    strategy_counts = np.bincount(final_state, minlength=3)
    strategy_fractions = strategy_counts / len(final_state)
    
    # Community analysis using CORRECTED function
    community_sizes = _calculate_community_sizes_proper(final_state)
    
    # Diversity metrics
    initial_diversity = _calculate_shannon_diversity(strategy_history[0])
    final_diversity = _calculate_shannon_diversity(final_state)
    
    # Create report
    report = f"""
GAME SIMULATION SUMMARY
=====================================

Simulation Parameters:
- δ (tie bonus): {parameters['delta']}
- ε (winning bonus): {parameters['epsilon']}
- Vertices: {parameters['vertices']}
- Time steps: {parameters['time_steps']}
- Temperature: {parameters['temperature']}
- β: {parameters['beta']}
- Random seed: {parameters['seed']}

Community Classification:
- Type: {'Stationary' if parameters['epsilon'] < 2 * parameters['delta'] else 'Drifting' if parameters['epsilon'] > 2 * parameters['delta'] else 'Critical'}
- Condition: ε {'<' if parameters['epsilon'] < 2 * parameters['delta'] else '>' if parameters['epsilon'] > 2 * parameters['delta'] else '='} 2δ

Final State Analysis:
- Rock (0): {strategy_counts[0]} players ({strategy_fractions[0]:.1%})
- Paper (1): {strategy_counts[1]} players ({strategy_fractions[1]:.1%})
- Scissors (2): {strategy_counts[2]} players ({strategy_fractions[2]:.1%})

Diversity Metrics:
- Initial Shannon diversity: {initial_diversity:.3f}
- Final Shannon diversity: {final_diversity:.3f}
- Diversity change: {final_diversity - initial_diversity:+.3f}

Community Structure:
- Number of communities: {len(community_sizes)}
- Average community size: {np.mean(community_sizes):.1f} (std: {np.std(community_sizes):.1f}) if community_sizes else (0, 0)
- Largest community: {max(community_sizes) if community_sizes else 0}
- Smallest community: {min(community_sizes) if community_sizes else 0}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    # Save report to analysis subdirectory
    analysis_dir = Path(output_dir) / "analysis"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"summary_d{parameters['delta']}_e{parameters['epsilon']}_{timestamp}.txt"
    filepath = analysis_dir / filename
    
    with open(filepath, 'w') as f:
        f.write(report)
    
    return str(filepath)


def create_zero_sum_summary_report(
    strategy_history: Union[np.ndarray, jnp.ndarray],
    bank_history: Union[np.ndarray, jnp.ndarray],
    parameters: Dict[str, Any],
    system_info: Dict[str, Any],
    wealth_conservation: Dict[str, Any],
    output_dir: str = "plots"
) -> str:
    """
    Create summary report specifically for zero-sum simulations.
    """
    if isinstance(strategy_history, jnp.ndarray):
        strategy_history = np.array(strategy_history)
    if isinstance(bank_history, jnp.ndarray):
        bank_history = np.array(bank_history)
    
    # Basic statistics
    final_state = strategy_history[-1]
    strategy_counts = np.bincount(final_state, minlength=3)
    strategy_fractions = strategy_counts / len(final_state)
    
    # Community analysis using CORRECTED function
    initial_communities = _calculate_community_sizes_proper(strategy_history[0])
    final_communities = _calculate_community_sizes_proper(final_state)
    
    # Diversity metrics
    initial_diversity = _calculate_shannon_diversity(strategy_history[0])
    final_diversity = _calculate_shannon_diversity(final_state)
    
    # Zero-sum specific wealth analysis
    initial_wealth = bank_history[0]
    final_wealth = bank_history[-1]
    wealth_redistribution = np.std(final_wealth) - np.std(initial_wealth)
    
    # Create comprehensive report
    report = f"""
ZERO-SUM EVOLUTIONARY GAME SIMULATION SUMMARY
============================================

System Configuration:
- Game type: Zero-Sum Rock-Paper-Scissors
- Selection method: Deterministic (argmax)
- Vertices: {parameters['vertices']}
- Time steps: {parameters['time_steps']}
- Initial pattern: {parameters['initial_pattern']}
- Initial bank value: {parameters['initial_bank_value']}
- Random seed: {parameters['seed']}

Zero-Sum Properties:
- Payoff matrix: Winner +1, Loser -1, Tie 0
- Total system wealth: {system_info['total_initial_wealth']} (constant)
- Wealth conservation: {wealth_conservation['wealth_conserved']}
- Max wealth deviation: {wealth_conservation['max_deviation']:.2e}
- Expected dynamics: Deterministic spatial patterns

Initial Pattern Analysis:
- Pattern type: {parameters['initial_pattern']}
- Initial wealth std: {np.std(initial_wealth):.2f}
- Initial communities: {len(initial_communities)}
- Largest initial community: {max(initial_communities) if initial_communities else 0}

Final State Analysis:
- Rock (0): {strategy_counts[0]} agents ({strategy_fractions[0]:.1%})
- Paper (1): {strategy_counts[1]} agents ({strategy_fractions[1]:.1%})
- Scissors (2): {strategy_counts[2]} agents ({strategy_fractions[2]:.1%})

Diversity Evolution:
- Initial Shannon diversity: {initial_diversity:.3f}
- Final Shannon diversity: {final_diversity:.3f}
- Diversity change: {final_diversity - initial_diversity:+.3f}

Community Structure Evolution:
Initial State:
- Number of communities: {len(initial_communities)}
- Average community size: {np.mean(initial_communities):.1f} (std: {np.std(initial_communities):.1f}) if initial_communities else (0, 0)
- Largest community: {max(initial_communities) if initial_communities else 0}

Final State:
- Number of communities: {len(final_communities)}
- Average community size: {np.mean(final_communities):.1f} (std: {np.std(final_communities):.1f}) if final_communities else (0, 0)
- Largest community: {max(final_communities) if final_communities else 0}

Wealth Redistribution:
- Initial wealth mean: {np.mean(initial_wealth):.2f} ± {np.std(initial_wealth):.2f}
- Final wealth mean: {np.mean(final_wealth):.2f} ± {np.std(final_wealth):.2f}
- Wealth redistribution: {wealth_redistribution:+.2f} (change in std)
- Wealth range: [{np.min(final_wealth):.1f}, {np.max(final_wealth):.1f}]

Zero-Sum Validation:
- Total wealth conserved: {wealth_conservation['wealth_conserved']}
- Expected total: {wealth_conservation['expected_total_wealth']:.0f}
- Actual final total: {wealth_conservation['actual_total_wealth']:.0f}
- Conservation error: {abs(wealth_conservation['actual_total_wealth'] - wealth_conservation['expected_total_wealth']):.2e}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    # Save report
    analysis_dir = Path(output_dir) / "analysis"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"zero_sum_summary_{parameters['vertices']}agents_{timestamp}.txt"
    filepath = analysis_dir / filename
    
    with open(filepath, 'w') as f:
        f.write(report)
    
    return str(filepath)


def _calculate_shannon_diversity(strategies: np.ndarray) -> float:
    """Calculate Shannon diversity index for strategy distribution."""
    counts = np.bincount(strategies, minlength=3)
    probabilities = counts / np.sum(counts)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    return entropy


def _calculate_community_sizes_proper(strategies: np.ndarray) -> list:
    """
    Calculate community sizes with PROPER periodic boundary handling.
    This handles the ring topology of the 1D lattice.
    
    In a ring, a community can wrap around from the end to the beginning, so use a visited array to avoid double-counting.
    """
    if len(strategies) == 0:
        return []
    
    n = len(strategies)
    communities = []
    visited = np.zeros(n, dtype=bool)
    
    for start in range(n):
        if visited[start]:
            continue
            
        # Find community starting from this position
        current_strategy = strategies[start]
        size = 0
        
        # Go forward around the ring
        pos = start
        while not visited[pos] and strategies[pos] == current_strategy:
            visited[pos] = True
            size += 1
            pos = (pos + 1) % n
        
        # Go backward around the ring from start
        pos = (start - 1) % n
        while not visited[pos] and strategies[pos] == current_strategy:
            visited[pos] = True
            size += 1
            pos = (pos - 1) % n
        
        if size >= 2:  # Only count communities of size >= 2
            communities.append(size)
    
    return communities


# Keep the old function for backward compatibility but mark as deprecated
def _calculate_community_sizes(strategies: np.ndarray) -> list:
    """
    DEPRECATED: Use _calculate_community_sizes_proper() instead.
    This version has bugs with periodic boundary handling.
    """
    return _calculate_community_sizes_proper(strategies)


def export_for_analysis(
    strategy_history: Union[np.ndarray, jnp.ndarray],
    output_dir: str = "plots",
    format: str = "csv"
) -> str:
    """
    Export strategy history in format suitable for external analysis.
    
    Args:
        strategy_history: Strategy evolution data
        output_dir: Output directory
        format: Export format ('csv', 'txt', 'mat')
        
    Returns:
        Path to exported file
    """
    if isinstance(strategy_history, jnp.ndarray):
        strategy_history = np.array(strategy_history)
    
    output_path = Path(output_dir) / "data"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "csv":
        filename = f"strategy_history_{timestamp}.csv"
        filepath = output_path / filename
        np.savetxt(filepath, strategy_history, delimiter=',', fmt='%d')
    elif format == "txt":
        filename = f"strategy_history_{timestamp}.txt"
        filepath = output_path / filename
        np.savetxt(filepath, strategy_history, fmt='%d')
    elif format == "mat":
        try:
            from scipy.io import savemat
            filename = f"strategy_history_{timestamp}.mat"
            filepath = output_path / filename
            savemat(filepath, {"strategy_history": strategy_history})
        except ImportError:
            raise ImportError("scipy required for .mat export")
    else:
        raise ValueError(f"Unsupported export format: {format}")
    
    return str(filepath)


def export_zero_sum_data(
    strategy_history: Union[np.ndarray, jnp.ndarray],
    bank_history: Union[np.ndarray, jnp.ndarray],
    output_dir: str = "plots",
    format: str = "csv"
) -> tuple:
    """
    Export zero-sum simulation data in format suitable for external analysis.
    
    Returns:
        tuple: (strategy_export_path, bank_export_path)
    """
    if isinstance(strategy_history, jnp.ndarray):
        strategy_history = np.array(strategy_history)
    if isinstance(bank_history, jnp.ndarray):
        bank_history = np.array(bank_history)
    
    output_path = Path(output_dir) / "data"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "csv":
        strategy_file = f"zero_sum_strategies_{timestamp}.csv"
        bank_file = f"zero_sum_banks_{timestamp}.csv"
        
        strategy_path = output_path / strategy_file
        bank_path = output_path / bank_file
        
        np.savetxt(strategy_path, strategy_history, delimiter=',', fmt='%d')
        np.savetxt(bank_path, bank_history, delimiter=',', fmt='%.6f')
        
    elif format == "mat":
        try:
            from scipy.io import savemat
            
            mat_file = f"zero_sum_data_{timestamp}.mat"
            mat_path = output_path / mat_file
            
            savemat(mat_path, {
                "strategy_history": strategy_history,
                "bank_history": bank_history
            })
            
            return str(mat_path), str(mat_path)
            
        except ImportError:
            raise ImportError("scipy required for .mat export")
    else:
        raise ValueError(f"Unsupported export format: {format}")
    
    return str(strategy_path), str(bank_path)