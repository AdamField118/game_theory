"""
Input/Output utilities for evolutionary game simulations.
Handles file saving, directory setup, and data management.
"""

import json
import pickle
from pathlib import Path
from typing import Union, Any, Dict
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


def save_results(
    strategy_history: Union[np.ndarray, jnp.ndarray],
    args: Any,
    output_dir: str = "plots"
) -> str:
    """
    Save simulation results and parameters.
    
    Args:
        strategy_history: Strategy evolution data
        args: Command line arguments object
        output_dir: Output directory
        
    Returns:
        Path to saved data file
    """
    # Convert JAX arrays to numpy
    if isinstance(strategy_history, jnp.ndarray):
        strategy_history = np.array(strategy_history)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data dictionary
    data = {
        "strategy_history": strategy_history,
        "parameters": {
            "delta": args.delta,
            "epsilon": args.epsilon,
            "vertices": args.vertices,
            "time_steps": args.time_steps,
            "temperature": args.temperature,
            "beta": args.beta,
            "seed": args.seed
        },
        "metadata": {
            "timestamp": timestamp,
            "shape": strategy_history.shape,
            "n_strategies": len(np.unique(strategy_history)),
            "final_diversity": _calculate_shannon_diversity(strategy_history[-1])
        }
    }
    
    # Save as both numpy and pickle formats
    output_path = Path(output_dir) / "data"
    
    # Save numpy format (for easy loading)
    np_filename = f"simulation_d{args.delta}_e{args.epsilon}_{timestamp}.npz"
    np_filepath = output_path / np_filename
    np.savez_compressed(
        np_filepath,
        strategy_history=strategy_history,
        parameters=data["parameters"],
        metadata=data["metadata"]
    )
    
    # Save parameters as JSON for easy reading
    json_filename = f"parameters_d{args.delta}_e{args.epsilon}_{timestamp}.json"
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
    
    print(f"Data saved to: {np_filepath}")
    print(f"Parameters saved to: {json_filepath}")
    
    return str(np_filepath)


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
    
    # Community analysis
    community_sizes = _calculate_community_sizes(final_state)
    
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
- Average community size: {np.mean(community_sizes):.1f} (std: {np.std(community_sizes):.1f})
- Largest community: {max(community_sizes) if community_sizes else 0}
- Smallest community: {min(community_sizes) if community_sizes else 0}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    # Save report to analysis subdirectory
    analysis_dir = Path(output_dir) / "analysis"
    filename = f"summary_d{parameters['delta']}_e{parameters['epsilon']}.txt"
    filepath = analysis_dir / filename
    
    with open(filepath, 'w') as f:
        f.write(report)
    
    return str(filepath)


def _calculate_shannon_diversity(strategies: np.ndarray) -> float:
    """Calculate Shannon diversity index for strategy distribution."""
    counts = np.bincount(strategies, minlength=3)
    probabilities = counts / np.sum(counts)
    # Avoid log(0) by adding small epsilon
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    return entropy


def _calculate_community_sizes(strategies: np.ndarray) -> list:
    """Calculate sizes of communities (contiguous regions of same strategy)."""
    if len(strategies) == 0:
        return []
    
    # Handle periodic boundary conditions
    extended = np.concatenate([strategies, strategies])
    n = len(strategies)
    
    communities = []
    i = 0
    while i < n:
        current_strategy = extended[i]
        size = 1
        j = i + 1
        
        # Count contiguous same strategies (with wraparound)
        while j < i + n and extended[j % (2*n)] == current_strategy:
            size += 1
            j += 1
        
        if size >= 2:  # Only count communities of size >= 2
            communities.append(size)
        
        i = j if j < i + n else i + 1
    
    return communities


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