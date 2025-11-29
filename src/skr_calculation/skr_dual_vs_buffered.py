#!/usr/bin/env python3
"""
Secret Key Rate Analysis: Dual vs Buffered Downlink Architectures

This script analyzes the Secret Key Rate (SKR) for satellite-based QKD
using two architectures:
1. Dual downlink: Both photons sent to different ground stations simultaneously
2. Buffered downlink: One photon stored in quantum memory, enabling non-simultaneous access

Based on the BB84 protocol with one-way post-processing and includes:
- Satellite-to-satellite horizon angle calculation
- Maximum distance calculation for dual downlink at 20° elevation
- Instantaneous SKR calculation
- Buffer time analysis for different orbital geometries
- Comparison plots

Parameters:
- Orbital height: 500 km
- Elevation angle: 20°
- Earth radius: 6,371 km
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from link_analysis.downlink_probability import (
    downlink_prob, slant_range_from_elevation, 
    H_SAT_M, EARTH_RADIUS_M, ETA_MEM_SAT, ETA_DET_GROUND
)

R_EARTH = EARTH_RADIUS_M  # Earth radius [m] - from link_analysis
H_SAT = H_SAT_M          # Satellite altitude [m] - from link_analysis
THETA_ELEV = 20.0        # Elevation angle [degrees]
C_LIGHT = 3e8            # Speed of light [m/s]

# BB84 Protocol parameters
ATTEMPT_RATE = 90e6    # Attempt rate [Hz] - 90 MHz
QBER_X = 0.05         # Quantum bit error rate in X basis
QBER_Z = 0.05         # Quantum bit error rate in Z basis
ERROR_CORRECTION_INEFFICIENCY = 1.16  # Error correction inefficiency factor f
P_HERALD_BSM = 0.50   # Bell State Measurement success probability (heralding probability)
ETA_MEMORY = 0.74     # Memory efficiency for buffered downlink (74%)
ETA_ENTANGLED_SRC = 0.2  # Entangled source efficiency
N_MEMORY_MODES = 112  # Number of parallel quantum memory modes

print(f"Using updated system parameters:")
print(f"  H_SAT_M = {H_SAT_M/1000:.0f} km")
print(f"  EARTH_RADIUS_M = {EARTH_RADIUS_M/1000:.0f} km")
print(f"  ETA_MEM_SAT = {ETA_MEM_SAT}")
print(f"  Updated ETA_MEMORY = {ETA_MEMORY}")
print(f"  ATTEMPT_RATE = {ATTEMPT_RATE/1e6:.0f} MHz")
print(f"  N_MEMORY_MODES = {N_MEMORY_MODES}")
print(f"  ERROR_CORRECTION_INEFFICIENCY = {ERROR_CORRECTION_INEFFICIENCY}")
print(f"  P_HERALD_BSM = {P_HERALD_BSM}")
print()

# Formauls (See Paper for Sources)

def binary_entropy(x: float) -> float:
    """Calculate binary entropy function h(x) = -x*log2(x) - (1-x)*log2(1-x)"""
    if x <= 0 or x >= 1:
        return 0.0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def secure_fraction(qber_x: float, qber_z: float, f: float = ERROR_CORRECTION_INEFFICIENCY) -> float:
    """
    Calculate secure fraction using the updated formula:
    r_secure = (1/2) * max[1 - h(e_X) - f*h(e_Z), 0]
    
    Args:
        qber_x: QBER in X basis
        qber_z: QBER in Z basis  
        f: Error correction inefficiency factor
    """
    return 0.5 * max(1 - binary_entropy(qber_x) - f * binary_entropy(qber_z), 0.0)

def satellite_horizon_angle(h: float) -> float:
    """
    Calculate satellite-to-satellite horizon angle.
    θ_ss = 2 * arccos(R_E / (R_E + h))
    
    Args:
        h: Satellite altitude [m]
    
    Returns:
        Horizon angle [radians]
    """
    return 2 * np.arccos(R_EARTH / (R_EARTH + h))

def max_ground_distance_at_elevation(h: float, theta_elev_deg: float) -> float:
    """
    Calculate maximum ground distance for given elevation angle.
    Uses spherical Earth geometry.
    
    Args:
        h: Satellite altitude [m]
        theta_elev_deg: Elevation angle [degrees]
    
    Returns:
        Maximum ground distance [m]
    """
    theta_elev_rad = np.deg2rad(theta_elev_deg)
    
    # Central angle from satellite to horizon at given elevation
    alpha = np.arccos(R_EARTH * np.cos(theta_elev_rad) / (R_EARTH + h))
    
    # Ground distance
    ground_distance = R_EARTH * alpha
    
    return ground_distance

def orbital_period(h: float) -> float:
    """
    Calculate orbital period using Kepler's third law.
    
    Args:
        h: Satellite altitude [m]
    
    Returns:
        Orbital period [seconds]
    """
    GM = 3.986e14  # Earth's gravitational parameter [m³/s²]
    r = R_EARTH + h  # Orbital radius
    return 2 * np.pi * np.sqrt(r**3 / GM)

def time_between_passes(h: float, ground_distance: float) -> float:
    """
    Calculate time between successive passes over two ground stations.
    
    Args:
        h: Satellite altitude [m]
        ground_distance: Distance between ground stations [m]
    
    Returns:
        Time between passes [seconds]
    """
    period = orbital_period(h)
    angular_separation = ground_distance / R_EARTH  # Central angle [radians]
    
    # Fraction of orbit between the two stations
    orbit_fraction = angular_separation / (2 * np.pi)
    
    return period * orbit_fraction

def calculate_skr_dual_downlink(h: float, theta_elev_deg: float, 
                              sigma_p_rad: float = 1e-6) -> Dict:
    """
    Calculate instantaneous SKR for dual downlink architecture with parallel modes.
    
    Args:
        h: Satellite altitude [m]
        theta_elev_deg: Elevation angle [degrees]
        sigma_p_rad: Pointing jitter [radians]
    
    Returns:
        Dictionary with SKR analysis results
    """
    # Calculate link success probability
    elev_array = np.array([theta_elev_deg])
    p_success_single = downlink_prob(elev_array, sigma_p_rad)[0]
    
    # For dual downlink, both links must succeed
    p_success_dual = p_success_single**2
    
    # Calculate secure fraction
    r_secure = secure_fraction(QBER_X, QBER_Z)
    
    # Calculate instantaneous SKR with parallel modes
    # Each mode can attempt at ATTEMPT_RATE, but we're limited by simultaneity constraint
    # SKR = Y * R where Y = P_herald * η_A * η_B and R is secure fraction
    # Y(t) = P_herald * η_A(t) * η_B(t) - yield decomposition for S-G geometry
    Y_dual = P_HERALD_BSM * p_success_dual  # P_herald * η_A * η_B
    skr_per_mode = Y_dual * r_secure * ETA_ENTANGLED_SRC * ATTEMPT_RATE
    
    # For dual downlink, all modes are constrained by the same simultaneity requirement
    # So we get N_MEMORY_MODES × single mode performance
    skr_instantaneous = N_MEMORY_MODES * skr_per_mode
    
    # Calculate maximum ground distance for this elevation
    max_distance = max_ground_distance_at_elevation(h, theta_elev_deg)
    
    return {
        'elevation_deg': theta_elev_deg,
        'max_ground_distance_km': max_distance / 1000,
        'single_link_success_prob': p_success_single,
        'dual_link_success_prob': p_success_dual,
        'secure_fraction': r_secure,
        'instantaneous_skr_bps': skr_instantaneous,
        'skr_per_mode_bps': skr_per_mode,
        'n_modes': N_MEMORY_MODES,
        'slant_range_km': slant_range_from_elevation(elev_array)[0] / 1000
    }

def calculate_buffer_time_analysis(h: float, max_distance: float) -> Dict:
    """
    Calculate buffer time requirements for different scenarios.
    
    Args:
        h: Satellite altitude [m]
        max_distance: Maximum ground distance between stations [m]
    
    Returns:
        Dictionary with buffer time analysis
    """
    # Time between passes for maximum distance (20° elevation scenario)
    t_buffer_max = time_between_passes(h, max_distance)
    
    # Time for satellite directly overhead (0° central angle)
    t_buffer_overhead = time_between_passes(h, 0)
    
    # Difference in buffer time
    delta_t_buffer = t_buffer_max - t_buffer_overhead
    
    return {
        't_buffer_max_elevation_s': t_buffer_max,
        't_buffer_overhead_s': t_buffer_overhead,
        'delta_t_buffer_s': delta_t_buffer,
        't_buffer_max_elevation_min': t_buffer_max / 60,
        't_buffer_overhead_min': t_buffer_overhead / 60,
        'delta_t_buffer_min': delta_t_buffer / 60
    }

def calculate_skr_buffered_downlink_both_overhead(h: float, max_distance: float, 
                                                sigma_p_rad: float = 1e-6) -> Dict:
    """
    Calculate SKR for buffered downlink where satellite can be directly overhead 
    for BOTH connections with parallel memory modes.
    
    Args:
        h: Satellite altitude [m]
        max_distance: Maximum ground distance between stations [m]
        sigma_p_rad: Pointing jitter [radians]
    
    Returns:
        Dictionary with buffered SKR analysis for both-overhead scenario
    """
    # Both links: satellite directly overhead (90° elevation) for optimal performance
    elev_90_deg = np.array([90.0])
    p_success_overhead = downlink_prob(elev_90_deg, sigma_p_rad)[0]
    
    # Combined success probability: first link × memory efficiency × second link
    # Both links are overhead, so same success probability for each
    p_success_buffered_both_overhead = (p_success_overhead * ETA_MEMORY * 
                                       p_success_overhead)
    
    # Calculate secure fraction (same as before)
    r_secure = secure_fraction(QBER_X, QBER_Z)
    
    # Calculate instantaneous SKR with parallel memory modes
    # SKR = Y * R where Y = P_herald * η_A * η_B and R is secure fraction
    # Y(t) = P_herald * η_A(t) * η_B(t) - yield decomposition for S-G geometry
    Y_buffered = P_HERALD_BSM * p_success_buffered_both_overhead  # P_herald * η_A * η_B
    skr_per_mode = Y_buffered * r_secure * ETA_ENTANGLED_SRC * ATTEMPT_RATE
    
    # With buffered architecture, we can use all N_MEMORY_MODES in parallel
    # Each mode can store one photon and retrieve it later
    skr_instantaneous_buffered = N_MEMORY_MODES * skr_per_mode
    
    # Calculate slant ranges (both overhead)
    slant_range_overhead = slant_range_from_elevation(elev_90_deg)[0]  # Should be = h
    
    return {
        'first_link_elevation_deg': 90.0,
        'second_link_elevation_deg': 90.0,
        'first_link_success_prob_overhead': p_success_overhead,
        'second_link_success_prob_overhead': p_success_overhead,
        'combined_success_prob_with_memory': p_success_buffered_both_overhead,
        'instantaneous_skr_bps': skr_instantaneous_buffered,
        'skr_per_mode_bps': skr_per_mode,
        'n_modes': N_MEMORY_MODES,
        'memory_efficiency': ETA_MEMORY,
        'slant_range_both_links_km': slant_range_overhead / 1000,
        'secure_fraction': r_secure
    }


def apply_publication_style():
    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "figure.titlesize": 24,
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.grid": False,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
    })

def create_individual_plots(dual_results: Dict, buffered_overhead_results: Dict, 
                          buffer_results: Dict, output_dir: str):
    """Create individual publication-quality plots."""
    
    apply_publication_style()
    
    # Calculate improvement factor
    improvement_factor = buffered_overhead_results['instantaneous_skr_bps'] / dual_results['instantaneous_skr_bps']
    
    # Plot 1: SKR Comparison
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    scenarios = ['Dual Downlink\n(Both at 20°)', 'Buffered Downlink\n(Both Overhead)']
    skr_values = [dual_results['instantaneous_skr_bps'], 
                 buffered_overhead_results['instantaneous_skr_bps']]
    
    bars = ax1.bar(scenarios, skr_values, color=['#d62728', '#2ca02c'], 
                   alpha=0.8, edgecolor='black', linewidth=1.0)
    ax1.set_ylabel('Secret Key Rate (bits/s)')
    ax1.set_title('Instantaneous Secret Key Rate Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add value labels and improvement annotation (positioned to avoid title overlap)
    for bar, skr in zip(bars, skr_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height*1.15,
                f'{skr:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax1.text(0.5, 0.75, f'{improvement_factor:.0f}× Improvement', 
             transform=ax1.transAxes, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
             fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'skr_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"[+] SKR comparison plot saved to: {output_path}")
    plt.close()
    
    # Plot 2: Success Probability Comparison
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    success_probs = [dual_results['dual_link_success_prob'], 
                    buffered_overhead_results['combined_success_prob_with_memory']]
    
    bars = ax2.bar(scenarios, success_probs, color=['#d62728', '#2ca02c'], 
                   alpha=0.8, edgecolor='black', linewidth=1.0)
    ax2.set_ylabel('Success Probability')
    ax2.set_title('Link Success Probability Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    for bar, prob in zip(bars, success_probs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height*1.15,
                f'{prob:.2e}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'success_probability_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"[+] Success probability plot saved to: {output_path}")
    plt.close()
    
    # Plot 3: Link Geometry Comparison
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    link_types = ['Dual Link 1\n(20° elev)', 'Dual Link 2\n(20° elev)', 'Buffered Links\n(Both 90°)']
    elevations = [20, 20, 90]
    slant_ranges = [
        dual_results['slant_range_km'],
        dual_results['slant_range_km'],
        buffered_overhead_results['slant_range_both_links_km']
    ]
    
    colors = ['#ff7f0e', '#d62728', '#2ca02c']
    bars = ax3.bar(link_types, slant_ranges, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.0)
    ax3.set_ylabel('Slant Range (km)')
    ax3.set_title('Link Geometry: Slant Range Comparison')
    ax3.grid(True, alpha=0.3)
    
    for bar, range_km, elev in zip(bars, slant_ranges, elevations):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{range_km:.0f} km\n({elev}°)', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'slant_range_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"[+] Slant range plot saved to: {output_path}")
    plt.close()
    
    
    # Plot 4: Heatmap - SKR Improvement vs Memory Efficiency and Elevation Angle
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    
    # Create parameter grids with higher resolution for smoother gradient
    memory_efficiencies = np.linspace(0.5, 0.9, 100)  # 50% to 90% memory efficiency
    elevation_angles = np.linspace(10, 90, 100)       # 10° to 90° elevation
    
    # Calculate SKR improvement matrix
    improvement_matrix = np.zeros((len(memory_efficiencies), len(elevation_angles)))
    
    # Fixed dual downlink SKR (reference)
    dual_skr = dual_results['instantaneous_skr_bps']
    
    for i, eta_mem in enumerate(memory_efficiencies):
        for j, elev in enumerate(elevation_angles):
            # Calculate buffered SKR for this parameter combination
            elev_array = np.array([elev])
            p_success_overhead = downlink_prob(elev_array, 1e-6)[0]  # 1 µrad pointing jitter
            p_success_buffered = (p_success_overhead * eta_mem * p_success_overhead)
            r_secure = secure_fraction(QBER_X, QBER_Z)
            Y_buffered_heatmap = P_HERALD_BSM * p_success_buffered  # P_herald * η_A * η_B
            skr_per_mode_buffered = Y_buffered_heatmap * r_secure * ETA_ENTANGLED_SRC * ATTEMPT_RATE
            skr_buffered = N_MEMORY_MODES * skr_per_mode_buffered
            
            # Calculate improvement factor
            improvement_matrix[i, j] = skr_buffered / dual_skr
    
    # Create heatmap
    from matplotlib.colors import LogNorm
    im = ax4.imshow(improvement_matrix, aspect='auto', origin='lower', 
                   cmap='viridis', norm=LogNorm(vmin=1, vmax=np.max(improvement_matrix)))
    
    n_ticks = 6
    elev_tick_indices = np.linspace(0, len(elevation_angles)-1, n_ticks, dtype=int)
    mem_tick_indices = np.linspace(0, len(memory_efficiencies)-1, n_ticks, dtype=int)
    
    ax4.set_xticks(elev_tick_indices)
    ax4.set_xticklabels([f'{elevation_angles[i]:.0f}°' for i in elev_tick_indices])
    ax4.set_yticks(mem_tick_indices)
    ax4.set_yticklabels([f'{memory_efficiencies[i]:.1f}' for i in mem_tick_indices])
    
    ax4.set_xlabel('Buffered Link Elevation Angle')
    ax4.set_ylabel('Memory Efficiency')
    ax4.set_title('SKR Improvement Factor: Buffered vs Dual Downlink')
    
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('SKR Improvement Factor')
  
    contour_levels = [1, 10, 50, 100, 200]
    cs = ax4.contour(improvement_matrix, levels=contour_levels, colors='white', 
                    linewidths=2.0)
    ax4.clabel(cs, inline=True, fontsize=26, fmt="%.0f×")
    
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'skr_improvement_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"[+] SKR improvement heatmap saved to: {output_path}")
    plt.close()

def create_master_plot(dual_results: Dict, buffered_overhead_results: Dict, 
                      buffer_results: Dict, output_dir: str):
    """Create a master plot combining all individual analyses."""
    
    apply_publication_style()
    
    # Create figure with subplots (2x2 + bottom layout)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # Calculate improvement factor
    improvement_factor = buffered_overhead_results['instantaneous_skr_bps'] / dual_results['instantaneous_skr_bps']
    
    # Get maximum distance for title
    max_dist_km = dual_results['max_ground_distance_km']
    
    # Plot 1: SKR Comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    scenarios = ['Dual Downlink\n(Both at 20°)', 'Buffered Downlink\n(Both Overhead)']
    skr_values = [dual_results['instantaneous_skr_bps'], 
                 buffered_overhead_results['instantaneous_skr_bps']]
    
    bars1 = ax1.bar(scenarios, skr_values, color=['#d62728', '#2ca02c'], 
                   alpha=0.8, edgecolor='black', linewidth=1.0)
    ax1.set_ylabel('Secret Key Rate (bits/s)')
    ax1.set_title('(a) Instantaneous Secret Key Rate')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add value labels
    for bar, skr in zip(bars1, skr_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height*1.15,
                f'{skr:.1e}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add improvement annotation
    ax1.text(0.5, 0.75, f'{improvement_factor:.0f}× Improvement', 
             transform=ax1.transAxes, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
             fontsize=12, fontweight='bold')
    
    # Plot 2: Success Probability Comparison (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    success_probs = [dual_results['dual_link_success_prob'], 
                    buffered_overhead_results['combined_success_prob_with_memory']]
    
    bars2 = ax2.bar(scenarios, success_probs, color=['#d62728', '#2ca02c'], 
                   alpha=0.8, edgecolor='black', linewidth=1.0)
    ax2.set_ylabel('Success Probability')
    ax2.set_title('(b) Link Success Probability')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    for bar, prob in zip(bars2, success_probs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height*1.15,
                f'{prob:.2e}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 3: Slant Range Comparison (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    link_types = ['Dual Link 1\n(20° elev)', 'Dual Link 2\n(20° elev)', 'Buffered Links\n(Both 90°)']
    elevations = [20, 20, 90]
    slant_ranges = [
        dual_results['slant_range_km'],
        dual_results['slant_range_km'],
        buffered_overhead_results['slant_range_both_links_km']
    ]
    
    colors = ['#ff7f0e', '#d62728', '#2ca02c']
    bars3 = ax3.bar(link_types, slant_ranges, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.0)
    ax3.set_ylabel('Slant Range (km)')
    ax3.set_title('(c) Link Geometry Comparison')
    ax3.grid(True, alpha=0.3)
    
    for bar, range_km, elev in zip(bars3, slant_ranges, elevations):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{range_km:.0f} km\n({elev}°)', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    # Plot 4: SKR Improvement Heatmap (bottom, spanning both columns)
    ax4 = fig.add_subplot(gs[1, :])
    
    # Create parameter grids with higher resolution for smoother gradient
    memory_efficiencies = np.linspace(0.5, 0.9, 100)
    elevation_angles = np.linspace(10, 90, 100)
    
    # Calculate SKR improvement matrix
    improvement_matrix = np.zeros((len(memory_efficiencies), len(elevation_angles)))
    dual_skr = dual_results['instantaneous_skr_bps']
    
    for i, eta_mem in enumerate(memory_efficiencies):
        for j, elev in enumerate(elevation_angles):
            elev_array = np.array([elev])
            p_success_overhead = downlink_prob(elev_array, 1e-6)[0]
            p_success_buffered = (p_success_overhead * eta_mem * p_success_overhead)
            r_secure = secure_fraction(QBER_X, QBER_Z)
            Y_buffered_heatmap = P_HERALD_BSM * p_success_buffered  # P_herald * η_A * η_B
            skr_per_mode_buffered = Y_buffered_heatmap * r_secure * ETA_ENTANGLED_SRC * ATTEMPT_RATE
            skr_buffered = N_MEMORY_MODES * skr_per_mode_buffered
            improvement_matrix[i, j] = skr_buffered / dual_skr
    
    # Create heatmap
    from matplotlib.colors import LogNorm
    im = ax4.imshow(improvement_matrix, aspect='auto', origin='lower', 
                   cmap='viridis', norm=LogNorm(vmin=1, vmax=np.max(improvement_matrix)))
    
    n_ticks = 6
    elev_tick_indices = np.linspace(0, len(elevation_angles)-1, n_ticks, dtype=int)
    mem_tick_indices = np.linspace(0, len(memory_efficiencies)-1, n_ticks, dtype=int)
    
    ax4.set_xticks(elev_tick_indices)
    ax4.set_xticklabels([f'{elevation_angles[i]:.0f}°' for i in elev_tick_indices])
    ax4.set_yticks(mem_tick_indices)
    ax4.set_yticklabels([f'{memory_efficiencies[i]:.1f}' for i in mem_tick_indices])
    
    ax4.set_xlabel('Buffered Link Elevation Angle')
    ax4.set_ylabel('Memory Efficiency')
    ax4.set_title('(d) SKR Improvement Factor: Parameter Space Analysis')
    
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('SKR Improvement Factor')
    
    contour_levels = [1, 10, 50, 100, 200]
    cs = ax4.contour(improvement_matrix, levels=contour_levels, colors='white', 
                    linewidths=2.0)
    ax4.clabel(cs, inline=True, fontsize=26, fmt="%.0f×")
    
    # Overall chaty title
    fig.suptitle('Secret Key Rate Analysis: Dual vs Buffered Downlink\n' +
                f'112-Mode Quantum Memory @ 90 MHz, Ground Station Separation: {max_dist_km:.0f} km',
                fontsize=18, fontweight='bold', y=0.98)
    
    # Add system parameters text box bottom
    params_text = (f'System Parameters:\n'
                  f'• Orbital height: {H_SAT/1000:.0f} km\n'
                  f'• Memory efficiency: {ETA_MEMORY*100:.0f}%\n'
                  f'• Repetition rate: {ATTEMPT_RATE/1e6:.0f} MHz\n'
                  f'• Memory modes: {N_MEMORY_MODES}\n'
                  f'• Buffer time: {buffer_results["t_buffer_max_elevation_min"]:.1f} min')
    
    fig.text(0.02, 0.02, params_text, fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
             verticalalignment='bottom')
    
    # Save plot
    output_path = os.path.join(output_dir, 'skr_master_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"[+] Master analysis plot saved to: {output_path}")
    plt.close()

def create_comparison_plot(dual_results: Dict, buffered_overhead_results: Dict, 
                         buffer_results: Dict, output_dir: str):
    """Create both individual and master publication-quality plots."""
    # Create individual plots
    create_individual_plots(dual_results, buffered_overhead_results, buffer_results, output_dir)
    
    # Create master combined plot
    create_master_plot(dual_results, buffered_overhead_results, buffer_results, output_dir)

def main():
    """Main analysis function."""
    print("=" * 70)
    print("SECRET KEY RATE ANALYSIS: DUAL vs BUFFERED DOWNLINK")
    print("112-MODE QUANTUM MEMORY @ 90 MHz")
    print("=" * 70)
    print(f"Orbital height: {H_SAT/1000:.0f} km")
    print(f"Elevation angle: {THETA_ELEV}°")
    print(f"Earth radius: {R_EARTH/1000:.0f} km")
    print(f"Repetition rate: {ATTEMPT_RATE/1e6:.0f} MHz")
    print(f"Memory modes: {N_MEMORY_MODES}")
    print(f"Memory efficiency: {ETA_MEMORY*100:.0f}%")
    print()
    
    
    # Calculate dual downlink SKR
    print("\n" + "─" * 40)
    print("DUAL DOWNLINK ANALYSIS")
    print("─" * 40)
    
    dual_results = calculate_skr_dual_downlink(H_SAT, THETA_ELEV)
    
    print(f"Maximum ground distance (20° elevation): {dual_results['max_ground_distance_km']:.1f} km")
    print(f"Slant range: {dual_results['slant_range_km']:.1f} km")
    print(f"Single link success probability: {dual_results['single_link_success_prob']:.2e}")
    print(f"Dual link success probability: {dual_results['dual_link_success_prob']:.2e}")
    print(f"Secure fraction: {dual_results['secure_fraction']:.3f}")
    print(f"SKR per mode: {dual_results['skr_per_mode_bps']:.2e} bits/s")
    print(f"Total instantaneous SKR (dual): {dual_results['instantaneous_skr_bps']:.2e} bits/s")
    print(f"Number of parallel modes: {dual_results['n_modes']}")
    
    # Calculate buffer time analysis
    print("\n" + "─" * 40)
    print("BUFFER TIME ANALYSIS")
    print("─" * 40)
    
    max_distance_m = dual_results['max_ground_distance_km'] * 1000
    buffer_results = calculate_buffer_time_analysis(H_SAT, max_distance_m)
    
    print(f"Buffer time (max distance): {buffer_results['t_buffer_max_elevation_min']:.1f} minutes")
    print(f"Buffer time (overhead): {buffer_results['t_buffer_overhead_min']:.1f} minutes")
    print(f"Difference in buffer time: {buffer_results['delta_t_buffer_min']:.1f} minutes")
    
    # Calculate buffered downlink SKR (both overhead scenario)
    print("\n" + "─" * 40)
    print("BUFFERED DOWNLINK ANALYSIS (BOTH OVERHEAD)")
    print("─" * 40)
    
    max_distance_m = dual_results['max_ground_distance_km'] * 1000
    buffered_overhead_results = calculate_skr_buffered_downlink_both_overhead(H_SAT, max_distance_m)
    
    print(f"Both links (overhead, 90° elevation):")
    print(f"  Single link success probability: {buffered_overhead_results['first_link_success_prob_overhead']:.2e}")
    print(f"  Slant range: {buffered_overhead_results['slant_range_both_links_km']:.1f} km")
    print(f"Combined success prob. (with memory): {buffered_overhead_results['combined_success_prob_with_memory']:.2e}")
    print(f"Memory efficiency: {buffered_overhead_results['memory_efficiency']:.1f}")
    print(f"SKR per mode: {buffered_overhead_results['skr_per_mode_bps']:.2e} bits/s")
    print(f"Total instantaneous SKR (buffered): {buffered_overhead_results['instantaneous_skr_bps']:.2e} bits/s")
    print(f"Number of parallel modes: {buffered_overhead_results['n_modes']}")
    
    improvement_factor = buffered_overhead_results['instantaneous_skr_bps'] / dual_results['instantaneous_skr_bps']
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON: DUAL vs BUFFERED (BOTH OVERHEAD)")
    print("=" * 60)
    print(f"Scenario: Ground stations at maximum separation ({dual_results['max_ground_distance_km']:.0f} km)")
    print()
    print(f"DUAL DOWNLINK (both links at 20° elevation):")
    print(f"  Success probability: {dual_results['dual_link_success_prob']:.2e}")
    print(f"  Instantaneous SKR: {dual_results['instantaneous_skr_bps']:.2e} bits/s")
    print(f"  Slant range: {dual_results['slant_range_km']:.1f} km")
    print()
    print(f"BUFFERED DOWNLINK (both links overhead):")
    print(f"  Success probability: {buffered_overhead_results['combined_success_prob_with_memory']:.2e}")
    print(f"  Instantaneous SKR: {buffered_overhead_results['instantaneous_skr_bps']:.2e} bits/s")
    print(f"  Slant range: {buffered_overhead_results['slant_range_both_links_km']:.1f} km")
    print()
    print(f"Buffer time requirement: {buffer_results['t_buffer_max_elevation_min']:.1f} minutes")
    print(f"Memory must satisfy: T_mem ≥ {buffer_results['t_buffer_max_elevation_s']:.0f} seconds")
    
    # Create output directory and generate plots (relative to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_dir = os.path.join(project_root, 'plots', 'skr_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    create_comparison_plot(dual_results, buffered_overhead_results, buffer_results, output_dir)
    
    # Save detailed results to file
    results_file = os.path.join(output_dir, 'skr_both_overhead_analysis_results.txt')
    with open(results_file, 'w') as f:
        f.write("SECRET KEY RATE ANALYSIS: DUAL vs BUFFERED (BOTH OVERHEAD)\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Orbital height: {H_SAT/1000:.0f} km\n")
        f.write(f"Ground station separation: {dual_results['max_ground_distance_km']:.1f} km\n")
        
        f.write("SCENARIO COMPARISON:\n")
        f.write("─" * 35 + "\n")
        f.write("DUAL DOWNLINK (both links at 20° elevation):\n")
        f.write(f"  Success probability: {dual_results['dual_link_success_prob']:.2e}\n")
        f.write(f"  Instantaneous SKR: {dual_results['instantaneous_skr_bps']:.2e} bits/s\n")
        f.write(f"  Slant range: {dual_results['slant_range_km']:.1f} km\n\n")
        
        f.write("BUFFERED DOWNLINK (both links overhead):\n")
        f.write(f"  Single link success prob (90°): {buffered_overhead_results['first_link_success_prob_overhead']:.2e}\n")
        f.write(f"  Combined success prob (with memory): {buffered_overhead_results['combined_success_prob_with_memory']:.2e}\n")
        f.write(f"  Instantaneous SKR: {buffered_overhead_results['instantaneous_skr_bps']:.2e} bits/s\n")
        f.write(f"  Slant range: {buffered_overhead_results['slant_range_both_links_km']:.1f} km\n")
        f.write(f"  Buffer time required: {buffer_results['t_buffer_max_elevation_min']:.1f} minutes\n")
        f.write(f"  Memory efficiency: {buffered_overhead_results['memory_efficiency']:.1f}\n\n")
    
    
    print(f"\n[+] Detailed results saved to: {results_file}")
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()
