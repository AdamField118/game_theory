"""
Core evolutionary game implementation with JAX acceleration.
Implements the wealth-mediated thermodynamic strategy evolution model.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from typing import Tuple, Optional
import numpy as np


class EvolutionaryGame:
    """
    Evolutionary game on a 1D lattice with wealth-mediated strategy updates.
    
    Implements the model from Olson et al. (2022):
    - Rock-paper-scissors game with parameterized payoff matrix
    - Boltzmann distribution for strategy updates based on neighborhood wealth
    - Periodic boundary conditions (cycle graph)
    """
    
    def __init__(
        self,
        n_vertices: int = 300,
        delta: float = 0.5,
        epsilon: float = 0.0,
        temperature: float = 100.0,
        beta: Optional[float] = None,
        seed: int = 42
    ):
        """
        Initialize the evolutionary game.
        
        Args:
            n_vertices: Number of vertices in the 1D lattice
            delta: Tie bonus parameter (δ)
            epsilon: Winning bonus parameter (ε)
            temperature: Temperature parameter (T)
            beta: Inverse temperature (1/kT), computed from T if None
            seed: Random seed
        """
        self.n_vertices = n_vertices
        self.delta = delta
        self.epsilon = epsilon
        self.temperature = temperature
        self.beta = beta if beta is not None else 1.0 / temperature
        self.seed = seed
        
        # Initialize random key
        self.key = jr.PRNGKey(seed)
        
        # Create payoff matrix A(δ, ε) from equation (3)
        self.payoff_matrix = self._create_payoff_matrix()
        
        # JIT compile functions for speed
        self._update_banks_jit = jax.jit(self._update_banks)
        self._update_strategies_jit = jax.jit(self._update_strategies)
        self._simulation_step_jit = jax.jit(self._simulation_step)
        
        print(f"Initialized game with payoff matrix:")
        print(f"A(δ={delta}, ε={epsilon}) =")
        print(np.array(self.payoff_matrix))
    
    def _create_payoff_matrix(self) -> jnp.ndarray:
        """
        Create the generalized RPS payoff matrix from equation (3).
        
        A(δ, ε) = [[1+δ,   0,   2+ε],
                   [2+ε, 1+δ,     0],
                   [  0, 2+ε,   1+δ]]
        """
        return jnp.array([
            [1.0 + self.delta, 0.0, 2.0 + self.epsilon],
            [2.0 + self.epsilon, 1.0 + self.delta, 0.0],
            [0.0, 2.0 + self.epsilon, 1.0 + self.delta]
        ])
    
    def initialize_state(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialize random strategies and zero bank values.
        
        Returns:
            strategies: Array of shape (n_vertices,) with values in {0, 1, 2}
            banks: Array of shape (n_vertices,) initialized to zero
        """
        strategies = jr.randint(key, (self.n_vertices,), 0, 3)
        banks = jnp.zeros(self.n_vertices)
        return strategies, banks
    
    def _get_neighbors(self, i: int) -> Tuple[int, int]:
        """Get left and right neighbors for vertex i in a cycle."""
        left = (i - 1) % self.n_vertices
        right = (i + 1) % self.n_vertices
        return left, right
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_banks(self, strategies: jnp.ndarray, banks: jnp.ndarray) -> jnp.ndarray:
        """
        Update bank values based on payoffs from playing neighbors.
        Each player plays against their two neighbors in the cycle.
        """
        new_banks = banks.copy()
        
        # Vectorized neighbor interactions
        left_neighbors = jnp.roll(strategies, 1)
        right_neighbors = jnp.roll(strategies, -1)
        
        # Calculate payoffs against both neighbors
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
        Calculate neighborhood bank values Ns(x,t) for each strategy s.
        
        Returns:
            neighborhood_banks: Array of shape (n_vertices, 3) where
                                neighborhood_banks[i, s] = Ns(i, t)
        """
        # For each vertex, consider itself and its two neighbors
        left_banks = jnp.roll(banks, 1)
        right_banks = jnp.roll(banks, -1)
        left_strategies = jnp.roll(strategies, 1)
        right_strategies = jnp.roll(strategies, -1)
        
        # Initialize neighborhood bank sums
        neighborhood_banks = jnp.zeros((self.n_vertices, 3))
        
        # Sum bank values for each strategy in the neighborhood
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
    def _update_strategies(
        self, 
        strategies: jnp.ndarray, 
        banks: jnp.ndarray, 
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Update strategies using Boltzmann distribution based on neighborhood banks.
        
        Pr[σ(x,t) = s] = exp(β * Ns(x,t)) / Σ_s exp(β * Ns(x,t))
        """
        # Calculate neighborhood bank values
        neighborhood_banks = self._calculate_neighborhood_banks(strategies, banks)
        
        # Apply Boltzmann distribution
        logits = self.beta * neighborhood_banks
        probabilities = jax.nn.softmax(logits, axis=1)
        
        # Sample new strategies
        keys = jr.split(key, self.n_vertices)
        new_strategies = jax.vmap(
            lambda key, probs: jr.categorical(key, jnp.log(probs))
        )(keys, probabilities)
        
        return new_strategies
    
    @partial(jax.jit, static_argnums=(0,))
    def _simulation_step(
        self, 
        state: Tuple[jnp.ndarray, jnp.ndarray], 
        key: jax.random.PRNGKey
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """
        Perform one simulation step: update banks, then update strategies.
        
        Returns:
            New state and current strategies for history tracking
        """
        strategies, banks = state
        
        # Step 1: Update bank values based on current strategies
        new_banks = self._update_banks(strategies, banks)
        
        # Step 2: Update strategies based on neighborhood banks
        new_strategies = self._update_strategies(strategies, new_banks, key)
        
        new_state = (new_strategies, new_banks)
        return new_state, strategies  # Return old strategies for history
    
    def run_simulation(self, n_steps: int) -> jnp.ndarray:
        """
        Run the full simulation and return strategy history.
        
        Args:
            n_steps: Number of time steps to simulate
            
        Returns:
            strategy_history: Array of shape (n_steps, n_vertices) with strategy evolution
        """
        # Initialize state
        key, subkey = jr.split(self.key)
        initial_strategies, initial_banks = self.initialize_state(subkey)
        initial_state = (initial_strategies, initial_banks)
        
        # Generate random keys for each step
        keys = jr.split(key, n_steps)
        
        # Run simulation using scan for efficiency
        final_state, strategy_history = jax.lax.scan(
            self._simulation_step,
            initial_state,
            keys
        )
        
        return strategy_history
    
    def get_payoff_info(self) -> dict:
        """Get information about the current payoff matrix configuration."""
        condition = "Unknown"
        if self.epsilon < 2 * self.delta:
            condition = "Stationary (ε < 2δ)"
        elif self.epsilon > 2 * self.delta:
            condition = "Transient (ε > 2δ)"
        else:
            condition = "Critical (ε = 2δ)"
        
        return {
            "delta": self.delta,
            "epsilon": self.epsilon,
            "condition": condition,
            "payoff_matrix": np.array(self.payoff_matrix)
        }
