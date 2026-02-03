
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS
from src.physics.worker_get_plot_curve import worker_get_plot_curve 

def plot_theoretical_eos(baselines, n_curves):
    """
    Generates the Theoretical EoS Prior plot.
    
    Parameters:
    - baselines: Dictionary of hadronic max masses (for scaling).
    - n_curves: Number of curves to generate per class.
    """
    set_paper_style()
    print(f"\n--- Generating Theoretical EoS Plot ({n_curves} curves) ---")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 1. Generate Curves in Parallel
    # We use a distinct seed for each curve to ensure variety
    seeds = np.random.randint(0, 1e9, n_curves)
    
    # Run Workers
    print(f"  > Simulating {n_curves} Hadronic and {n_curves} Quark EoS...")
    res_h = Parallel(n_jobs=-1)(delayed(worker_get_plot_curve)('hadronic', baselines, s) for s in seeds)
    res_q = Parallel(n_jobs=-1)(delayed(worker_get_plot_curve)('quark', baselines, s) for s in seeds)
    
    # 2. Plotting
    # Use very low alpha to handle density pile-up visually
    alpha_val = 0.1 if n_curves >= 500 else 0.3
    
    # Plot Hadronic (Green)
    for res in res_h:
        if res is None: continue
        eos = res[1] # [Eps, P]
        
        # Filter purely for visualization ranges to avoid drawing artifacts
        mask = (eos[:,0] > 50) & (eos[:,0] < 6000)
        ax.plot(eos[mask, 0], eos[mask, 1], color=COLORS['H_main'], alpha=alpha_val, lw=0.5)

    # Plot Quark (Magenta)
    for res in res_q:
        if res is None: continue
        eos = res[1] # [Eps, P]
        mask = (eos[:,0] > 50) & (eos[:,0] < 6000)
        ax.plot(eos[mask, 0], eos[mask, 1], color=COLORS['Q_main'], alpha=alpha_val, lw=0.5)

    # 3. Physics Constraints
    x_guide = np.logspace(1.5, 4, 100)
    
    # Causal Limit (P = epsilon, c_s = 1)
    ax.plot(x_guide, x_guide, color='black', linestyle=':', lw=1.5, alpha=0.8, 
            label=r'Causal ($c_s=1$)')
    
    # Conformal Limit (P = epsilon/3, c_s^2 = 1/3)
    # This is the high-density limit for QCD
    ax.plot(x_guide, x_guide/3.0, color='gray', linestyle='--', lw=1.5, alpha=0.8, 
            label=r'Conformal ($c_s^2=1/3$)')

    # 4. Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Match limits to Grand Summary for consistency
    ax.set_xlim(CONSTANTS['PLOT_EPS_LOG'])
    ax.set_ylim(1e0, 3e3) # Cap at 3000 MeV/fm^3 matching the generation limit
    
    ax.set_xlabel(r"Energy Density $\varepsilon$ [MeV/fm$^3$]")
    ax.set_ylabel(r"Pressure $P$ [MeV/fm$^3$]")
    ax.set_title(r"Prior Model Space (Microphysics)")
    
    # Custom Legend
    lines = [
        Line2D([0], [0], color=COLORS['H_main'], lw=2), 
        Line2D([0], [0], color=COLORS['Q_main'], lw=2),
        Line2D([0], [0], color='black', linestyle=':'),
        Line2D([0], [0], color='gray', linestyle='--')
    ]
    labels = ['Hadronic Models', 'Quark Models (CFL)', r'Causal Limit ($c_s=1$)', r'Conformal Limit ($c_s^2=1/3$)']
    
    ax.legend(lines, labels, loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.savefig("plots/fig_theoretical_eos.pdf")
    plt.close()
    print("[Success] Saved Theoretical EoS Plot.")