"""
This module contains the solvers of the flixOpt framework, making them available to the end user in a compact way.
"""

from .math_modeling import (
    GurobiSolver,
    HighsSolver,
    _Solver,
)

__all__ = [
    '_Solver',
    'HighsSolver',
    'GurobiSolver',
]
