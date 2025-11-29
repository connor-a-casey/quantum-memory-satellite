"""
SKR Calculation Module
=====================

This module provides Secret Key Rate (SKR) analysis for satellite-based quantum key distribution,
comparing dual downlink and buffered downlink architectures with quantum memory.

Key Features:
- 100-mode quantum memory analysis
- 90 MHz repetition rate calculations  
- Dual vs buffered downlink comparison
- Publication-quality visualizations
- Parameter space heatmaps

Main Functions:
- calculate_skr_dual_downlink(): Dual downlink SKR calculation
- calculate_skr_buffered_downlink_both_overhead(): Buffered downlink SKR calculation
- create_individual_plots(): Generate publication-quality plots
- main(): Complete analysis workflow
"""

from .skr_dual_vs_buffered import (
    calculate_skr_dual_downlink,
    calculate_skr_buffered_downlink_both_overhead,
    calculate_buffer_time_analysis,
    create_individual_plots,
    main
)

__all__ = [
    'calculate_skr_dual_downlink',
    'calculate_skr_buffered_downlink_both_overhead', 
    'calculate_buffer_time_analysis',
    'create_individual_plots',
    'main'
]



