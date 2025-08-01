"""
Zero-sum evolutionary game implementation with deterministic selection.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from typing import Tuple, Optional, Literal
import numpy as np


class ZeroSumEvolutionaryGame:
    """
    Zero-sum evolutionary game
    
    Key features:
    - Zero-sum payoffs: winner gets +1, loser gets -1, tie gets 0
    - Deterministic selection: always choose strategy with highest local bank value
    - Wealth conservation: total system wealth remains constant
    - Striped or random initial conditions
    """
    
    def __init__(
        self,
        n_vertices: int = 512,
        initial_pattern: Literal["random", "striped"] = "striped",
        initial_bank_value: float = -1000.0,
        stripe_size: Optional[int] = None,
        seed: int = 42
    ):
        """
        Initialize the zero-sum evolutionary game.
        
        Args:
            n_vertices: Number of vertices in the 1D lattice
            initial_pattern: "random" or "striped" initial strategy pattern
            initial_bank_value: Starting bank value for all agents
            stripe_size: Size of each stripe (if None, uses n_vertices//3 per strategy)
            seed: Random seed for reproducibility
        """
        self.n_vertices = n_vertices
        self.initial_pattern = initial_pattern
        self.initial_bank_value = initial_bank_value
        self.stripe_size = stripe_size if stripe_size is not None else n_vertices // 3
        self.seed = seed
        
        # Initialize random key
        self.key = jr.PRNGKey(seed)
        
        # Zero-sum RPS payoff matrix: winner gets +1, loser gets -1, tie gets 0
        self.payoff_matrix = jnp.array([
            [ 0, -1, +1],  # Rock vs [Rock, Paper, Scissors]
            [+1,  0, -1],  # Paper vs [Rock, Paper, Scissors]
            [-1, +1,  0]   # Scissors vs [Rock, Paper, Scissors]
        ])
        
        # JIT compile functions for performance
        self._update_banks_jit = jax.jit(self._update_banks)
        self._update_strategies_jit = jax.jit(self._update_strategies_deterministic)
        self._simulation_step_jit = jax.jit(self._simulation_step)
        
        print(f"Initialized Zero-Sum Evolutionary Game:")
        print(f"  Vertices: {n_vertices}")
        print(f"  Initial pattern: {initial_pattern}")
        print(f"  Initial bank value: {initial_bank_value}")
        print(f"  Stripe size: {self.stripe_size}")
        print(f"  Total system wealth: {n_vertices * initial_bank_value}")
        print(f"  Zero-sum payoff matrix:")
        print(np.array(self.payoff_matrix))
    
    def initialize_state(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize strategies and bank values based on specified pattern."""
        if self.initial_pattern == "striped":
            strategies = self._create_striped_pattern()
        else:
            strategies = jr.randint(key, (self.n_vertices,), 0, 3)
        
        banks = jnp.full(self.n_vertices, self.initial_bank_value)
        return strategies, banks
    
    def _create_striped_pattern(self) -> jnp.ndarray:
        """Create equal stripes of rock, paper, scissors."""
        strategies = jnp.zeros(self.n_vertices, dtype=jnp.int32)
        
        # Create pattern: stripe_size of each strategy in sequence
        for i in range(self.n_vertices):
            cycle_position = i % (3 * self.stripe_size)
            if cycle_position < self.stripe_size:
                strategy = 0  # Rock
            elif cycle_position < 2 * self.stripe_size:
                strategy = 1  # Paper
            else:
                strategy = 2  # Scissors
            strategies = strategies.at[i].set(strategy)
        
        return strategies
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_banks(self, strategies: jnp.ndarray, banks: jnp.ndarray) -> jnp.ndarray:
        """
        Update bank values based on zero-sum payoffs from playing neighbors.
        In zero-sum games, total wealth is conserved.
        """
        # Vectorized neighbor interactions
        left_neighbors = jnp.roll(strategies, 1)
        right_neighbors = jnp.roll(strategies, -1)
        
        # Calculate payoffs against both neighbors using zero-sum matrix
        left_payoffs = self.payoff_matrix[strategies, left_neighbors]
        right_payoffs = self.payoff_matrix[strategies, right_neighbors]
        
        # Update banks with total payoffs
        new_banks = banks + left_payoffs + right_payoffs
        
        return new_banks
    
    @partial(jax.jit, static_argnums=(0,))
    def _calculate_neighborhood_banks(
        self, 
        strategies: jnp.ndarray, 
        banks: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate neighborhood bank values for deterministic selection.
        For each vertex, sum bank values of neighbors playing each strategy.
        """
        # Get neighbor banks and strategies
        left_banks = jnp.roll(banks, 1)
        right_banks = jnp.roll(banks, -1)
        left_strategies = jnp.roll(strategies, 1)
        right_strategies = jnp.roll(strategies, -1)
        
        # Initialize neighborhood bank sums for each strategy
        neighborhood_banks = jnp.zeros((self.n_vertices, 3))
        
        # Sum bank values for each strategy in the neighborhood (self + neighbors)
        for s in range(3):
            # Self contribution
            self_contribution = jnp.where(strategies == s, banks, 0.0)
            # Left neighbor contribution
            left_contribution = jnp.where(left_strategies == s, left_banks, 0.0)
            # Right neighbor contribution  
            right_contribution = jnp.where(right_strategies == s, right_banks, 0.0)
            
            neighborhood_banks = neighborhood_banks.at[:, s].set(
                self_contribution + left_contribution + right_contribution
            )
        
        return neighborhood_banks
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_strategies_deterministic(
        self, 
        strategies: jnp.ndarray, 
        banks: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Update strategies deterministically: choose strategy with highest neighborhood bank value.
        This is the key difference from wealth-mediated Boltzmann selection.
        """
        neighborhood_banks = self._calculate_neighborhood_banks(strategies, banks)
        
        # Deterministic selection: argmax of neighborhood bank values
        # Cast to int32 to match input dtype
        new_strategies = jnp.argmax(neighborhood_banks, axis=1).astype(jnp.int32)
        
        return new_strategies
    
    @partial(jax.jit, static_argnums=(0,))
    def _simulation_step(
        self, 
        state: Tuple[jnp.ndarray, jnp.ndarray], 
        key: jax.random.PRNGKey  # Not used in deterministic model but kept for interface compatibility
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """
        Perform one simulation step: update banks, then update strategies deterministically.
        """
        strategies, banks = state
        
        # Update bank values based on zero-sum payoffs
        new_banks = self._update_banks(strategies, banks)
        
        # Update strategies deterministically 
        new_strategies = self._update_strategies_deterministic(strategies, new_banks)
        
        new_state = (new_strategies, new_banks)
        return new_state, strategies  # Return old strategies for history tracking
    
    def run_simulation(self, n_steps: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run the zero-sum simulation and return both strategy and bank histories.
        
        Args:
            n_steps: Number of time steps to simulate
            
        Returns:
            strategy_history: Array of shape (n_steps, n_vertices) with strategy evolution
            bank_history: Array of shape (n_steps, n_vertices) with bank evolution
        """
        # Initialize state
        key, subkey = jr.split(self.key)
        initial_strategies, initial_banks = self.initialize_state(subkey)
        initial_state = (initial_strategies, initial_banks)
        
        # Generate random keys for each step (not used but maintains interface)
        keys = jr.split(key, n_steps)
        
        # Run simulation using scan for efficiency
        final_state, strategy_history = jax.lax.scan(
            self._simulation_step,
            initial_state,
            keys
        )
        
        # Extract bank history by running simulation again (could be optimized)
        # For now, I'll just return strategy history and compute banks when needed
        bank_history = self._compute_bank_history(strategy_history)
        
        return strategy_history, bank_history
    
    def _compute_bank_history(self, strategy_history: jnp.ndarray) -> jnp.ndarray:
        """Compute bank evolution from strategy history."""
        n_steps, n_vertices = strategy_history.shape
        bank_history = jnp.zeros((n_steps, n_vertices))
        
        # Initialize banks
        banks = jnp.full(n_vertices, self.initial_bank_value)
        bank_history = bank_history.at[0].set(banks)
        
        # Compute bank evolution
        for t in range(1, n_steps):
            banks = self._update_banks(strategy_history[t-1], banks)
            bank_history = bank_history.at[t].set(banks)
        
        return bank_history
    
    def get_system_info(self) -> dict:
        """Get comprehensive information about the zero-sum system."""
        total_initial_wealth = self.n_vertices * self.initial_bank_value
        
        return {
            "game_type": "zero_sum",
            "selection_method": "deterministic",
            "initial_pattern": self.initial_pattern,
            "n_vertices": self.n_vertices,
            "initial_bank_value": self.initial_bank_value,
            "stripe_size": self.stripe_size,
            "total_initial_wealth": total_initial_wealth,
            "wealth_conservation": "Strict (zero-sum payoffs)",
            "payoff_matrix": np.array(self.payoff_matrix),
            "expected_dynamics": "Deterministic spatial patterns with wealth conservation"
        }
    
    def verify_wealth_conservation(self, bank_history: jnp.ndarray) -> dict:
        """Verify that wealth is conserved throughout the simulation."""
        total_wealth_over_time = jnp.sum(bank_history, axis=1)
        expected_total = self.n_vertices * self.initial_bank_value
        
        max_deviation = jnp.max(jnp.abs(total_wealth_over_time - expected_total))
        
        return {
            "expected_total_wealth": float(expected_total),
            "actual_total_wealth": float(total_wealth_over_time[-1]),
            "max_deviation": float(max_deviation),
            "wealth_conserved": float(max_deviation) < 1e-10,
            "total_wealth_over_time": np.array(total_wealth_over_time)
        }