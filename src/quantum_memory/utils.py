# -*- coding: utf-8 -*-
# Code from arXiv:2402.17752

""" Defines functions for manipulating and analysing simulation results. """

import csv
import ast
from .dynamics import CollectiveSpinsPDE
from pde import FileStorage, ScalarField


def write_magnitudes(read_filename, write_filename="magnitudes.csv",
                     params=CollectiveSpinsPDE()):
    """
    Write the spatial average of the spin operator magnitudes to a .csv file.

    :read_filename: file containing operator magnitudes in each point and time
    :write_filename: magnitudes are saved with this parameter as a filename
    :params: instance of CollectiveSpinsPDE class with simulation parameters
    :return: none
    """
    reader = FileStorage(read_filename, write_mode="read_only")

    f = open(write_filename, "w")  # Use "w" mode to overwrite, not append
    f.write("t, S, K \n")

    for time, collection in reader.items():
        S, K = collection.fields
        f.write(f"{time}, {S.magnitude*S.magnitude}, \
                {K.magnitude*K.magnitude} \n")

    f.close()


def write_parameters(params, write_filename="parameters.txt"):
    """
    Write the simulation parameters to a .txt file.

    :params: instance of CollectiveSpinsPDE class with simulation parameters
    :write_filename: simulation parameters are saved with this as a filename
    :return: none
    """
    attrs = vars(params)

    f = open(write_filename, "a")
    f.write(', '.join("%s: %s" % item for item in attrs.items()))
    f.close()


def efficiency(filename, params, write_filename='efficiency.csv'):
    """
    Calculate the efficiency of the spin exchange and write it to a file.

    :params: instance of CollectiveSpinsPDE class with simulation parameters
    :write_filename: simulation parameters are saved with this as a filename
    :return: calculated efficiency of the spin-exchange mechanism
    """
    efficiency = 0
    with open(filename, 'r') as csvfile:
        magnitudes = csv.reader(csvfile, delimiter=',')
        headers = next(magnitudes)  # ignore file header
        for row in magnitudes:
            try:
                # Skip rows that don't have exactly 3 columns or contain headers
                if len(row) != 3 or row[0].strip() == 't':
                    continue
                    
                time_val = float(row[0].strip())
                s_val = float(row[1].strip())
                
                if time_val >= params.tpulse+params.tdark+params.t0:
                    if s_val > efficiency:
                        efficiency = s_val
            except (ValueError, IndexError):
                # Skip malformed rows silently
                continue

    print(f"Exchange Efficiency: {efficiency}")

    f = open(write_filename, "a")
    f.write(f"{params.J/params.gamma_s}, {efficiency} \n")
    f.close()

    return efficiency


def parse_parameters_from_file(filename):
    """
    Parse simulation parameters from a file.
    
    :filename: path to file containing parameters
    :return: dictionary of parsed parameters suitable for CollectiveSpinsPDE
    """
    with open(filename, 'r') as f:
        # Read the first non-empty line
        for line in f:
            line = line.strip()
            if line:
                # Parse the string as key-value pairs
                # Remove the leading/trailing parts and split by comma
                params_str = line
                
                # Split by comma and parse each key-value pair
                params = {}
                pairs = params_str.split(', ')
                
                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Skip internal parameters and computed parameters
                        if key in ['_cache', 'diagnostics', 'noise', 'rng', 'complex_valued', 'tpulse', 'tr']:
                            continue
                            
                        # Convert string values to appropriate types
                        try:
                            # Try to evaluate as a Python literal
                            if value == 'True':
                                params[key] = True
                            elif value == 'False':
                                params[key] = False
                            elif value.startswith('Generator('):
                                continue  # Skip RNG
                            else:
                                params[key] = float(value)
                        except (ValueError, SyntaxError):
                            # Keep as string if conversion fails
                            params[key] = value
                
                return params
    
    return {}
