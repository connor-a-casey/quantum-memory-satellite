#!/usr/bin/env python3
"""
Quantum Memory Satellite Project - Main Entry Point

This script provides a single interface to run different components of the project:
1. Quantum memory protocol simulations (from arXiv:2402.17752)
2. Satellite link analysis and probability calculations
3. SKR  analysis (100-mode quantum memory @ 90 MHz)

Usage:
    python src/main.py quantum_memory  # Run quantum memory simulation
    python src/main.py link_analysis   # Run satellite link analysis
    python src/main.py skr_analysis    # Run SKR dual vs buffered analysis
    python src/main.py --help          # Show help
"""

import argparse
import sys
import os

def run_quantum_memory():
    """Run the quantum memory protocol simulation."""
    print("Running quantum memory protocol simulation...")
    from quantum_memory.simulation import run_simulation
    from quantum_memory.utils import parse_parameters_from_file
    import os
    
    input_params_file = os.path.join('data', 'input', 'satellite_parameters.txt')
    
    if os.path.exists(input_params_file):
        print(f"Loading parameters from {input_params_file}")
        params = parse_parameters_from_file(input_params_file)
        print(f"Loaded parameters: {params}")
        # run from file in input 
        result = run_simulation(dt=5e-4, filename='satellite_simulation', **params)
    else:
        print("No input parameters found, using default parameters")
        # run default
        result = run_simulation(dt=5e-4, filename='quantum_memory_example')
    
    print("Quantum memory simulation completed!")
    return result

def run_link_analysis():
    """Run the satellite link analysis."""
    print("Running satellite link analysis...")
    from link_analysis.downlink_probability import main as downlink_main
    
    downlink_main()
    print("Link analysis completed!")

def run_skr_analysis():
    """Run the Secret Key Rate analysis."""
    print("Running SKR analysis (100-mode quantum memory @ 90 MHz)...")
    from skr_calculation.skr_dual_vs_buffered import main as skr_main
    
    skr_main()
    print("SKR analysis completed!")

def main():
    parser = argparse.ArgumentParser(
        description="Quantum Memory Satellite Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'component',
        choices=['quantum_memory', 'link_analysis', 'skr_analysis'],
        help='Component to run'
    )
    
    args = parser.parse_args()
    
    if args.component == 'quantum_memory':
        run_quantum_memory()
    elif args.component == 'link_analysis':
        run_link_analysis()
    elif args.component == 'skr_analysis':
        run_skr_analysis()

if __name__ == "__main__":
    main()
