# -*- coding: utf-8 -*-
# Code from arXiv:2402.17752

"""
Simulates the dynamics of an optical quantum memory protocol proposed in [1].

We model the collective spin-exchange dynamics between the alkali and noble-gas
spin ensembles in an optical quantum memory protocol proposed for this mixture
using the Bloch-Langevin formalism. Using the method of lines for discretizing
a symmetric spatial grid (defined in geometry.py), the resulting system of PDE
(defined in dynamics.py) is numerically solved. A 4th-order Runge Kutta method
with adaptive time stepping is used for numerical integration, implemented in
the open-source package py-pde [2]. This numerical solution accounts for the
interaction of the spins with the boundaries of the cell, inducing a mismatch
between the spatial modes of the collective spin operators of each species.

A simulation of the exchange dynamics can be performed by calling the function
run_simulation with the desired physical parameters; the evolution of the spin
operators can then be visualized (using the methods defined in plotting.py).


[1] A. Barbosa, H. Terças, E. Z. Cruzeiro (2024). arXiv:2402.17752
[2] D. Zwicker (2020). Journal of Open Source Software, 5(48), 2158.

"""

import pde
import warnings
from pde import config, FileStorage
from pde.trackers import ProgressTracker
from .geometry import grid
from .utils import write_magnitudes, write_parameters, efficiency
from .plotting import plot_magnitude, plot_kymograph
from .dynamics import CollectiveSpinsPDE
from datetime import datetime
from numba.core.errors import NumbaDeprecationWarning

config["numba.fastmath"] = True
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
startTime = datetime.now()


def run_simulation(dt=0.05, filename='fields.hdf5', **args):
    """
    Evolve the expected values of the spin operators throughout the protocol.

    :dt: initial time interval for integration (with adaptive time stepping)
    :filename: expected values of fields in each point and time are stored here
    :args: parameters forwarded to instance of CollectiveSpinsPDE class
    :return: scalar fields representing the spin operators when simulation ends
    """
    import os
    
    eq = CollectiveSpinsPDE(**args)

    # set up initial state of the system
    state = eq.get_initial_state(grid)

    # Create output directories if they don't exist
    plots_dir = 'plots'
    data_processed_dir = os.path.join('data', 'processed')
    
    for directory in [plots_dir, data_processed_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Define file paths with proper directories
    params_file = os.path.join(data_processed_dir, "{filename}_parameters.txt".format(filename=filename))
    hdf5_file = os.path.join(data_processed_dir, '{filename}.hdf5'.format(filename=filename))
    csv_file = os.path.join(data_processed_dir, '{filename}.csv'.format(filename=filename))
    eta_file = os.path.join(data_processed_dir, '{filename}_eta.csv'.format(filename=filename))

    # write simulation parameters
    write_parameters(eq, params_file)

    # write values of fields over time to file
    storage = FileStorage(hdf5_file, write_mode="truncate")
    writer = FileStorage(hdf5_file, write_mode="append")
    # track simulation progress in real time
    progress = ProgressTracker(1000)

    # solve the system of pde using RK4 with adaptive time stepping
    result = eq.solve(state, t_range=(0, eq.tr), dt=dt,
                      tracker=[writer.tracker(1), progress], adaptive=True,
                      scheme='rk', backend='numba', tolerance=1e-8)

    # read the field values over time
    reader = FileStorage(hdf5_file, write_mode="read_only")

    # result.plot(kind="image") # plot the spatial profile of fields at t=tr

    # write spatial-averaged magnitudes to csv file
    write_magnitudes(hdf5_file, csv_file, eq)

    # Create memory_efficiency subdirectory if it doesn't exist
    memory_efficiency_dir = os.path.join(plots_dir, 'memory_efficiency')
    os.makedirs(memory_efficiency_dir, exist_ok=True)
    
    # plot spatial-averaged magnitudes over time (save to memory_efficiency directory)
    plot_magnitude(csv_file, eq, 'S',
                   os.path.join(memory_efficiency_dir, '{filename}_P.pdf'.format(filename=filename)))
    plot_magnitude(csv_file, eq, 'K',
                   os.path.join(memory_efficiency_dir, '{filename}_S.pdf'.format(filename=filename)))

    plot_kymograph(hdf5_file,
                   eq, operator='S', save_filename=os.path.join(memory_efficiency_dir, '{filename}_kymo_S.png'.format(filename=filename)))
    plot_kymograph(hdf5_file,
                   eq, operator='K', save_filename=os.path.join(memory_efficiency_dir, '{filename}_kymo_K.png'.format(filename=filename)))

    se_efficiency = efficiency(csv_file, eq, eta_file)

    runtime = datetime.now() - startTime  # print total runtime
    print("Total Runtime: {runtime}".format(runtime=runtime))

    writer.close()
    reader.clear()
    reader.close()

    return result


""" Run Simulation """

# Simulation should be run via main.py, not directly from this module
# Use: python src/main.py quantum_memory
