#!/usr/bin/env python3
"""
Main entry point for the wealth-mediated thermodynamic strategy evolution simulation.
Reproduces Figure 1 from Olson et al. (2022) paper.
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

from core.game import EvolutionaryGame
from utils.visualization import create_figure_1_plot
from utils.io import save_results, setup_output_dirs


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evolutionary Game Theory Simulation - Figure 1 Reproduction"
    )
    
    # Game parameters
    parser.add_argument("--delta", "--d", type=float, default=0.5,
                       help="Tie bonus parameter (δ)")
    parser.add_argument("--epsilon", "--e", type=float, default=0.0,
                       help="Winning bonus parameter (ε)")
    
    # Simulation parameters
    parser.add_argument("--vertices", type=int, default=300,
                       help="Number of vertices in the lattice")
    parser.add_argument("--time_steps", type=int, default=400,
                       help="Number of time steps to simulate")
    parser.add_argument("--temperature", "--T", type=float, default=100.0,
                       help="Temperature parameter")
    parser.add_argument("--beta", type=float, default=0.01,
                       help="Inverse temperature parameter (1/kT)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="plots",
                       help="Output directory for plots")
    parser.add_argument("--save_data", action="store_true",
                       help="Save simulation data as numpy arrays")
    
    # Performance parameters
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="Use GPU acceleration if available")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    return parser.parse_args()


def main():
    """Main simulation function."""
    args = parse_arguments()
    
    # Setup output directories
    setup_output_dirs(args.output_dir)
    
    # Print simulation parameters
    print("="*60)
    print("GAME THEORY SIMULATION")
    print("="*60)
    print(f"Parameters:")
    print(f"  δ (tie bonus): {args.delta}")
    print(f"  ε (winning bonus): {args.epsilon}")
    print(f"  Vertices: {args.vertices}")
    print(f"  Time steps: {args.time_steps}")
    print(f"  Temperature: {args.temperature}")
    print(f"  β: {args.beta}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output dir: {args.output_dir}")
    
    # Check GPU availability
    if args.use_gpu:
        devices = jax.devices()
        print(f"  JAX devices: {[str(d) for d in devices]}")
        # Check for CUDA/GPU devices (more robust detection)
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        if gpu_devices:
            print("GPU acceleration available")
        else:
            print("No GPU detected, using CPU")
    else:
        print("Using CPU only")
    
    print("="*60)
    
    # Initialize game
    print("Initializing simulation...")
    game = EvolutionaryGame(
        n_vertices=args.vertices,
        delta=args.delta,
        epsilon=args.epsilon,
        temperature=args.temperature,
        beta=args.beta,
        seed=args.seed
    )
    
    # Run simulation
    print("Running simulation...")
    start_time = time.time()
    
    # Log simulation start
    from utils.io import save_log
    save_log(f"Starting simulation: δ={args.delta}, ε={args.epsilon}, vertices={args.vertices}, steps={args.time_steps}", "info", args.output_dir)
    
    strategy_history = game.run_simulation(args.time_steps)
    
    end_time = time.time()
    simulation_time = end_time - start_time
    print(f"Simulation completed in {simulation_time:.2f} seconds")
    
    # Log simulation completion
    save_log(f"Simulation completed in {simulation_time:.2f} seconds", "info", args.output_dir)
    
    # Create visualization
    print("Creating visualization...")
    fig_path = create_figure_1_plot(
        strategy_history, 
        args.delta, 
        args.epsilon,
        args.output_dir
    )
    print(f"Plot saved to: {fig_path}")
    
    # Save data if requested
    if args.save_data:
        data_path = save_results(
            strategy_history,
            args,
            args.output_dir
        )
        # Create summary report
        from utils.io import create_summary_report
        summary_path = create_summary_report(
            strategy_history,
            vars(args),
            args.output_dir
        )
        print(f"Data saved to: {data_path}")
        print(f"Summary saved to: {summary_path}")
    
    print("="*60)
    print("SIMULATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()