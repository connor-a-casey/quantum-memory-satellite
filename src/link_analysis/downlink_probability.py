"""
Link‑Establishment Probability Model
====================================
For every elementary connection — ground–satellite or satellite–satellite — the
single‑shot success probability

    η = η_mem · η_det · η_atm(θ) · η_dif(l),

is obtained from four statistically independent processes (see paper for equations)

(1) Memory efficiency   η_mem   (satellite only, default 0.74)
(2) Detector efficiency η_det   (ground 0.7, satellite memory 1.0)
(3) Atmospheric trans.  η_atm   (Beer–Lambert)
(4) Diffraction & pointing loss η_dif (Gaussian beam with jitter)

Geometric relations
-------------------
The elevation angle θ and slant range l follow directly from the ground‑track
distance L_g and the satellite altitude h :

    l²  = R_E² + (R_E+h)² – 2R_E(R_E+h)·cos(L_g/R_E)
    θ   = π/2 – L_g/R_E – arcsin[ (R_E/l)·sin(L_g/R_E) ],

with R_E = 6 371 km.

Default Parameter Set (hard‑coded below)
---------------------------------------
λ                 780 nm     (signal wavelength)
η_atm^z           0.8        (zenith clear‑sky transmission)
σ_p               0–3 µrad   (pointing jitter sweep)
θ_d               3  µrad    (half‑angle divergence)
R_rec             0.50 m     (ground station telescope radius)
R_rec             0.25 m     (satellite telescope radius)
h                 500 km     (satellite altitude)
ISL distance      0.1–5 Mm   (100 km–5 000 km)

All values can be edited in SECTION ▼ PHYSICAL CONSTANTS below.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


LAMBDA_M           = 795e-9        # wavelength λ [m]
ETA_ATM_Z          = 0.8           # zenith atmospheric transmissivity η_atm^z
ETA_MEM_SAT        = 0.74          # satellite quantum‑memory efficiency η_mem
ETA_MEM_GROUND     = 1.0           # ground station (no memory)
ETA_DET_SAT_MEM    = 1.0           # detector eff. for in‑memory links η_det
ETA_DET_GROUND     = 0.7           # ground SNSPD efficiency η_det
ETA_ENTANGLED_SRC  = 0.2           # entangled source efficiency
H_SAT_M            = 500e3         # satellite altitude h [m]
THETA_D_RAD        = 3e-6          # beam divergence half‑angle θ_d [rad]
R_REC_GROUND_M     = 0.50          # ground station receiver aperture radius [m]
R_REC_SAT_M        = 0.25          # satellite receiver aperture radius [m]
EARTH_RADIUS_M     = 6_371e3       # mean Earth radius R_E [m]


def apply_plot_style() -> None:
    """A lightweight, vector‑friendly plot style."""
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "figure.titlesize": 20,
        "figure.figsize": (18, 6),
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


# Loss Models (See Appendix Section of Paper for Formulas)
def L_atm_db(elev_deg: np.ndarray) -> np.ndarray:
    """Atmospheric loss in dB: L_atm = -10*log10(η_atm(θ))."""
    eta_atm_vals = eta_atm(elev_deg)
    # Avoid log(0) by setting minimum value
    eta_atm_vals = np.maximum(eta_atm_vals, 1e-10)
    return -10 * np.log10(eta_atm_vals)

def L_geo_db(l_m: np.ndarray) -> np.ndarray:
    """Geometric/free-space loss in dB."""
    # Free space path loss: L_geo = 20*log10(4πl/λ)
    l_m_arr = np.asarray(l_m)
    return 20 * np.log10(4 * np.pi * l_m_arr / LAMBDA_M)

def L_point_db(l_m: np.ndarray, sigma_p_rad: float, receiver_radius_m: float) -> np.ndarray:
    """Pointing loss in dB: L_point = -10*log10(η_dif(l))."""
    eta_dif_vals = eta_dif(l_m, sigma_p_rad, receiver_radius_m)
    # Avoid log(0) by setting minimum value
    eta_dif_vals = np.maximum(eta_dif_vals, 1e-10)
    return -10 * np.log10(eta_dif_vals)

def L_sys_db(eta_mem: float, eta_det: float) -> float:
    """System loss in dB: L_sys = -10*log10(η_mem * η_det)."""
    eta_sys = eta_mem * eta_det
    return -10 * np.log10(eta_sys)

def L_total_db(elev_deg: np.ndarray, l_m: np.ndarray, sigma_p_rad: float, 
               receiver_radius_m: float, eta_mem: float, eta_det: float) -> np.ndarray:
    """Total loss in dB: L_total = L_atm + L_geo + L_point + L_sys."""
    L_atm_vals = L_atm_db(elev_deg)
    L_geo_vals = L_geo_db(l_m)
    L_point_vals = L_point_db(l_m, sigma_p_rad, receiver_radius_m)
    L_sys_val = L_sys_db(eta_mem, eta_det)
    
    return L_atm_vals + L_geo_vals + L_point_vals + L_sys_val

def eta_from_db_loss(L_total_db_vals: np.ndarray) -> np.ndarray:
    """End-to-end transmission efficiency: η = 10^(-L_total/10)."""
    return 10 ** (-L_total_db_vals / 10)

def eta_atm(elev_deg: np.ndarray) -> np.ndarray:
    """Beer–Lambert atmospheric transmission η_atm(θ)."""
    eta = np.zeros_like(elev_deg, dtype=float)
    
    # Create a mask for valid elevation angles (elev_deg > 0)
    mask_positive_elev = elev_deg > 0
    
    # Initialize sin_theta array; calculate for positive elevations only
    sin_theta_values = np.zeros_like(elev_deg, dtype=float)
    if np.any(mask_positive_elev):
        sin_theta_values[mask_positive_elev] = np.sin(np.deg2rad(elev_deg[mask_positive_elev]))
    
    # Update mask to include condition sin_theta >= 1e-6
    # This combined mask ensures we only operate on valid inputs
    final_mask = mask_positive_elev & (sin_theta_values >= 1e-6)
    
    # Calculate transmission for elements satisfying the final_mask
    if np.any(final_mask):
        eta[final_mask] = ETA_ATM_Z ** (1.0 / sin_theta_values[final_mask])
        
    return eta


def eta_dif(l_m: np.ndarray, sigma_p_rad: float, receiver_radius_m: float) -> np.ndarray:
    """Diffraction + pointing loss η_dif(l) for a Gaussian beam."""

    l_m_arr = np.asarray(l_m)
    eta = np.zeros_like(l_m_arr, dtype=float) # Initialize results array based on l_m_arr shape

    if THETA_D_RAD <= 0:
        return eta # Return array of zeros

    # diffraction‑limited waist at launch
    w0 = LAMBDA_M / (np.pi * THETA_D_RAD) # scalar
    
    # beam radius at distance l_m_arr
    # This will be an array if l_m_arr is an array
    w_l = w0 * np.sqrt(1 + (THETA_D_RAD * l_m_arr / w0) ** 2)

    # effective spot size after convolving with pointing jitter sigma_p_rad
    # This will also be an array if l_m_arr is an array
    w_eff2 = w_l ** 2 + 4 * (sigma_p_rad * l_m_arr) ** 2
    
    # Create a mask for valid w_eff2 values (w_eff2 > 0)
    # And also for l_m_arr >= 0, as negative distances are not physical
    mask = (w_eff2 > 0) & (l_m_arr >= 0)
    
    if np.any(mask):
        # Calculate eta_dif only for elements satisfying mask
        eta[mask] = 1.0 - np.exp(-2 * receiver_radius_m ** 2 / w_eff2[mask])
        
    return eta

# Link Type

def uplink_prob(elev_deg: np.ndarray, sigma_p_rad: float) -> np.ndarray:
    """Ground → satellite using direct multiplication."""
    return (
        ETA_MEM_SAT
        * ETA_DET_SAT_MEM
        * eta_atm(elev_deg)  # eta_atm now returns an array
        * eta_dif(slant_range_from_elevation(elev_deg), sigma_p_rad, R_REC_SAT_M) 
    )

def downlink_prob(elev_deg: np.ndarray, sigma_p_rad: float) -> np.ndarray:
    """Satellite → ground using direct multiplication."""
    return (
        ETA_MEM_GROUND
        * ETA_DET_GROUND
        * eta_atm(elev_deg) # eta_atm now returns an array
        * eta_dif(slant_range_from_elevation(elev_deg), sigma_p_rad, R_REC_GROUND_M)
    )


def isl_prob(distance_m: np.ndarray, sigma_p_rad: float) -> np.ndarray:
    """Inter‑satellite optical link (above the atmosphere) using direct multiplication."""
    return (
        ETA_MEM_SAT
        * ETA_DET_SAT_MEM
        * 1.0  # no atmosphere
        * eta_dif(distance_m, sigma_p_rad, R_REC_SAT_M) # distance_m is an array, eta_dif handles this
    )

# Geometry Helper Functions 

def slant_range_from_elevation(elev_deg: np.ndarray) -> np.ndarray:
    """Return slant range l for given elevation angle θ (flat‑Earth approximation)."""
    # Ensure input is treated as a NumPy array for consistent processing
    elev_deg_arr = np.asarray(elev_deg)
    sin_theta = np.sin(np.deg2rad(elev_deg_arr))
    # Use np.where for element-wise conditional assignment
    slant_ranges = np.where(sin_theta < 1e-6, np.inf, H_SAT_M / sin_theta)
    return slant_ranges

def slant_range_from_ground_track(L_g: np.ndarray, h: float = H_SAT_M, R_E: float = EARTH_RADIUS_M) -> np.ndarray:
    """
    Calculate slant range using full spherical geometry from appendix.
    
    l² = R_E² + (R_E+h)² - 2R_E(R_E+h)·cos(L_g/R_E)
    
    Args:
        L_g: Ground track separation [m]
        h: Satellite altitude [m]
        R_E: Earth radius [m]
    
    Returns:
        Slant range l [m]
    """
    L_g_arr = np.asarray(L_g)
    
    # Convert ground track distance to angular separation
    angular_sep = L_g_arr / R_E
    
    # Calculate slant range using spherical law of cosines
    l_squared = (R_E**2 + (R_E + h)**2 - 
                2 * R_E * (R_E + h) * np.cos(angular_sep))
    
    return np.sqrt(np.maximum(l_squared, 0))  # Ensure non-negative

def elevation_from_ground_track(L_g: np.ndarray, h: float = H_SAT_M, R_E: float = EARTH_RADIUS_M) -> np.ndarray:
    """
    Calculate elevation angle using full spherical geometry from appendix.
    
    θ = π/2 - L_g/R_E - arcsin[(R_E/l)·sin(L_g/R_E)]
    
    Args:
        L_g: Ground track separation [m]
        h: Satellite altitude [m] 
        R_E: Earth radius [m]
    
    Returns:
        Elevation angle θ [degrees]
    """
    L_g_arr = np.asarray(L_g)
    
    # Calculate slant range first
    l = slant_range_from_ground_track(L_g_arr, h, R_E)
    
    # Convert ground track distance to angular separation
    angular_sep = L_g_arr / R_E
    
    # Calculate elevation angle
    sin_term = (R_E / l) * np.sin(angular_sep)
    # Clamp to valid range for arcsin
    sin_term = np.clip(sin_term, -1, 1)
    
    theta_rad = np.pi/2 - angular_sep - np.arcsin(sin_term)
    
    # Convert to degrees and ensure non-negative
    theta_deg = np.rad2deg(theta_rad)
    return np.maximum(theta_deg, 0)

# Heatmap Creation 

def generate_heatmaps(output_dir: str) -> None:
    apply_plot_style()

    n = 400  # resolution per axis

    jitter_urad = np.linspace(0, 3, n)
    jitter_rad = jitter_urad * 1e-6

    slant_km = np.linspace(500, 3500, n)
    slant_m = slant_km * 1e3

    Z_downlink = np.zeros((n, n))

    for i, sigma in enumerate(jitter_rad):
        # vectorised elevation angles for speed
        elev_deg = np.rad2deg(np.arcsin(np.clip(H_SAT_M / slant_m, 0, 1)))
        Z_downlink[i, :] = downlink_prob(elev_deg, sigma)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.viridis
    # Define vmin and vmax for LogNorm
    log_norm_vmin = 1e-4  # Smallest value for log scale, must be > 0
    current_vmax = 0.05   # Max probability for the scale

    contour_lvls = [1e-3, 1e-2, 3e-2]

    ax.set_box_aspect(1) # Ensure the plot area is square
    Xm, Ym = np.meshgrid(slant_km, jitter_urad)
    # Use LogNorm for pcolormesh. vmin and vmax are now part of the norm.
    im = ax.pcolormesh(Xm, Ym, Z_downlink, cmap=cmap, shading="gouraud",
                       norm=LogNorm(vmin=log_norm_vmin, vmax=current_vmax))
    cs = ax.contour(Xm, Ym, Z_downlink, levels=contour_lvls, colors="white", linewidths=2.0)
    ax.clabel(cs, inline=True, fontsize=18, fmt="%.1e")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success Probability η")

    ax.set_title("Satellite-to-Ground (Downlink)")
    ax.set_xlabel("Slant Range (km)")
    ax.set_ylabel("Pointing Jitter (µrad)")



    out_path_png = os.path.join(output_dir, "downlink_probability.png")
    plt.savefig(out_path_png)
    print(f"[+] Downlink probability plot saved ➜ {out_path_png}")
    
    out_path_pdf = os.path.join(output_dir, "downlink_probability.pdf")
    plt.savefig(out_path_pdf)
    print(f"[+] Downlink probability plot saved ➜ {out_path_pdf}")
    
    plt.close(fig)


def main() -> None:
    # Create output dir to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, "plots", "downlink_probability")
    os.makedirs(output_dir, exist_ok=True)
    generate_heatmaps(output_dir)

if __name__ == "__main__":
    main()
