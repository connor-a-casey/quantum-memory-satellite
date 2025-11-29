# Code from arXiv:2402.17752
"""
Quantum Memory Protocol Simulation

This module contains the core components for simulating the quantum memory protocol
described in arXiv:2402.17752, including PDE dynamics, geometry definitions,
plotting utilities, and simulation runners.
"""

from .dynamics import CollectiveSpinsPDE
from .geometry import grid, ngrid
from .simulation import run_simulation
from .plotting import plot_kymograph, plot_magnitude
from .utils import write_magnitudes, write_parameters, efficiency

__all__ = [
    'CollectiveSpinsPDE',
    'grid', 
    'ngrid',
    'run_simulation',
    'plot_kymograph',
    'plot_magnitude', 
    'write_magnitudes',
    'write_parameters',
    'efficiency'
]
