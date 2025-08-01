#!/usr/bin/env python3
"""
Main script supporting both wealth-mediated and zero-sum
"""

import argparse
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.game import EvolutionaryGame  # Wealth-mediated
from core.zero_sum_game import ZeroSumEvolutionaryGame  # Zero-sum
from utils.visualization import create_figure_1_plot, create_zero_sum_analysis_plot
from utils.io import save_results, setup_output_dirs


def parse_arguments():
    """Parse command line arguments with support for both game types."""
    parser = argparse.ArgumentParser(
        description="Evolutionary Game Theory Simulation - Wealth-Mediated and Zero-Sum"
    )
    
    # Game type selection
    parser.add_argument("--payoff_scheme", type=str, default="wealth_mediated",
                       choices=["wealth_mediated", "zero_sum"],
                       help="Choose game type: wealth_mediated (original) or zero_sum")
    
    # Wealth-mediated parameters
    group_wealth = parser.add_argument_group('Wealth-Mediated Parameters')
    group_wealth.add_argument("--delta", "--d", type=float, default=0.5,
                       help="Tie bonus parameter (δ) - wealth-mediated only")
    group_wealth.add_argument("--epsilon", "--e", type=float, default=0.0,
                       help="Winning bonus parameter (ε) - wealth-mediated only")
    group_wealth.add_argument("--temperature", "--T", type=float, default=100.0,
                       help="Temperature parameter - wealth-mediated only")
    group_wealth.add_argument("--beta", type=float, default=0.01,
                       help="Inverse temperature (1/kT) - wealth-mediated only")
    
    # Zero-sum Parameters
    group_zero = parser.add_argument_group('Zero-Sum Parameters')
    group_zero.add_argument("--initial_pattern", type=str, default="random",
                       choices=["random", "striped"],
                       help="Initial strategy pattern - affects both game types")
    group_zero.add_argument("--initial_bank_value", type=float, default=0.0,
                       help="Starting bank value for all agents")
    group_zero.add_argument("--stripe_size", type=int, default=None,
                       help="Size of each stripe (default: vertices//3)")
    
    # Shared parameters
    group_sim = parser.add_argument_group('Simulation Parameters')
    group_sim.add_argument("--vertices", type=int, default=300,
                       help="Number of vertices in the lattice")
    group_sim.add_argument("--time_steps", type=int, default=400,
                       help="Number of time steps to simulate")
    group_sim.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Preset configs
    parser.add_argument("--preset", type=str, default=None,
                       choices=["original_stationary", "original_drifting", "zero_sum_512"],
                       help="Use preset configuration")
    
    # Output parameters
    group_output = parser.add_argument_group('Output Parameters')
    group_output.add_argument("--output_dir", type=str, default="plots",
                       help="Output directory for plots")
    group_output.add_argument("--save_data", action="store_true",
                       help="Save simulation data as numpy arrays")
    
    # Performance parameters
    group_perf = parser.add_argument_group('Performance Parameters')
    group_perf.add_argument("--use_gpu", action="store_true", default=True,
                       help="Use GPU acceleration if available")
    
    return parser.parse_args()


def apply_preset(args):
    """Apply preset configurations for easy result reproduction."""
    if args.preset == "original_stationary":
        # Reproduce original paper Figure 1 left (stationary communities)
        args.payoff_scheme = "wealth_mediated"
        args.delta = 0.5
        args.epsilon = 0.0
        args.initial_bank_value = 0.0
        args.initial_pattern = "random"
        
    elif args.preset == "original_drifting":
        # Reproduce original paper Figure 1 right (drifting communities)
        args.payoff_scheme = "wealth_mediated"
        args.delta = 0.0
        args.epsilon = 16.0
        args.initial_bank_value = 0.0
        args.initial_pattern = "random"
        
    elif args.preset == "zero_sum_512":
        # Zero-sum configuration
        args.payoff_scheme = "zero_sum"
        args.vertices = 512
        args.initial_bank_value = -1000.0
        args.initial_pattern = "striped"
        args.stripe_size = args.vertices // 3
        
    return args


def create_wealth_mediated_game(args):
    """Factory function to create wealth-mediated game."""
    return EvolutionaryGame(
        n_vertices=args.vertices,
        delta=args.delta,
        epsilon=args.epsilon,
        temperature=args.temperature,
        beta=args.beta,
        seed=args.seed
    )


def create_zero_sum_game(args):
    """Factory function to create zero-sum game."""
    return ZeroSumEvolutionaryGame(
        n_vertices=args.vertices,
        initial_pattern=args.initial_pattern,
        initial_bank_value=args.initial_bank_value,
        stripe_size=args.stripe_size,
        seed=args.seed
    )


def run_wealth_mediated_simulation(game, args):
    """Run wealth-mediated simulation and return results."""
    print("Running wealth-mediated simulation...")
    start_time = time.time()
    
    strategy_history = game.run_simulation(args.time_steps)
    
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Create visualization
    fig_path = create_figure_1_plot(
        strategy_history, 
        args.delta, 
        args.epsilon,
        args.output_dir
    )
    
    return {
        "strategy_history": strategy_history,
        "bank_history": None,
        "figure_path": fig_path,
        "simulation_time": end_time - start_time
    }


def run_zero_sum_simulation(game, args):
    """Run zero-sum simulation and return results."""
    print("Running zero-sum simulation...")
    start_time = time.time()
    
    strategy_history, bank_history = game.run_simulation(args.time_steps)
    
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Verify wealth conservation
    wealth_check = game.verify_wealth_conservation(bank_history)
    print(f"Wealth conservation check:")
    print(f"  Expected total: {wealth_check['expected_total_wealth']}")
    print(f"  Actual total: {wealth_check['actual_total_wealth']}")
    print(f"  Max deviation: {wealth_check['max_deviation']:.2e}")
    print(f"  Wealth conserved: {wealth_check['wealth_conserved']}")
    
    # Create both visualizations
    from utils.visualization import create_zero_sum_evolution_plot
    
    # Simple evolution plot
    evolution_path = create_zero_sum_evolution_plot(strategy_history, args.output_dir)
    
    # Detailed analysis plot
    analysis_path = create_zero_sum_analysis_plot(
        strategy_history,
        bank_history,
        args.output_dir
    )
    
    return {
        "strategy_history": strategy_history,
        "bank_history": bank_history,
        "figure_path": evolution_path,  # Main plot path
        "analysis_path": analysis_path,  # Detailed analysis path
        "simulation_time": end_time - start_time,
        "wealth_conservation": wealth_check
    }


def main():
    """Main simulation function with separated implementations."""
    args = parse_arguments()
    
    # Apply preset if specified
    if args.preset:
        args = apply_preset(args)
        print(f"Applied preset: {args.preset}")
    
    # Setup output directories
    setup_output_dirs(args.output_dir)
    
    # Print simulation parameters
    print("="*70)
    print("EVOLUTIONARY GAME SIMULATION")
    print("="*70)
    print(f"Game type: {args.payoff_scheme}")
    print(f"Vertices: {args.vertices}")
    print(f"Time steps: {args.time_steps}")
    print(f"Random seed: {args.seed}")
    
    if args.payoff_scheme == "wealth_mediated":
        print(f"δ (tie bonus): {args.delta}")
        print(f"ε (winning bonus): {args.epsilon}")
        print(f"Temperature: {args.temperature}")
        print(f"β: {args.beta}")
    else:  # zero_sum
        print(f"Initial pattern: {args.initial_pattern}")
        print(f"Initial bank value: {args.initial_bank_value}")
        if args.initial_pattern == "striped":
            stripe_size = args.stripe_size or args.vertices // 3
            print(f"Stripe size: {stripe_size}")
    
    print(f"Output dir: {args.output_dir}")
    
    # Check GPU availability
    if args.use_gpu:
        devices = jax.devices()
        print(f"JAX devices: {[str(d) for d in devices]}")
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        if gpu_devices:
            print("GPU acceleration available")
        else:
            print("No GPU detected, using CPU")
    else:
        print("Using CPU only")
    
    print("="*70)
    
    # Create appropriate game instances
    if args.payoff_scheme == "wealth_mediated":
        game = create_wealth_mediated_game(args)
        results = run_wealth_mediated_simulation(game, args)
    else:  # zero_sum
        game = create_zero_sum_game(args)
        results = run_zero_sum_simulation(game, args)
    
    print(f"Plot saved to: {results['figure_path']}")

    # Only print analysis path if it exists (zero-sum only)
    if 'analysis_path' in results:
        print(f"Analysis plot saved to: {results['analysis_path']}")
    
    # Save data if requested
    if args.save_data:
        data_path = save_results(
            results["strategy_history"],
            args,
            args.output_dir,
            bank_history=results.get("bank_history"),
            additional_data=results.get("wealth_conservation")
        )
        
        # Create appropriate summary report
        if args.payoff_scheme == "zero_sum":
            from utils.io import create_zero_sum_summary_report
            summary_path = create_zero_sum_summary_report(
                results["strategy_history"],
                results["bank_history"],
                vars(args),
                game.get_system_info(),
                results["wealth_conservation"],
                args.output_dir
            )
        else:
            from utils.io import create_summary_report
            summary_path = create_summary_report(
                results["strategy_history"],
                vars(args),
                args.output_dir
            )
        
        print(f"Data saved to: {data_path}")
        print(f"Summary saved to: {summary_path}")
    
    # Print final analysis
    print("\nFinal state analysis:")
    final_state = np.array(results["strategy_history"][-1])
    strategy_counts = np.bincount(final_state, minlength=3)
    strategy_names = ['Rock', 'Paper', 'Scissors']
    
    for i, (name, count) in enumerate(zip(strategy_names, strategy_counts)):
        percentage = count / len(final_state) * 100
        print(f"  {name}: {count} agents ({percentage:.1f}%)")
    
    print("="*70)
    print("SIMULATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()