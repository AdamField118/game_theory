# src/core/__init__.py
"""
Core evolutionary game theory components.
"""

from .game import EvolutionaryGame  # Wealth-mediated
from .zero_sum_game import ZeroSumEvolutionaryGame  # Zero-sum

__all__ = ["EvolutionaryGame", "ZeroSumEvolutionaryGame"]