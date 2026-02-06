

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS

def plot_slope_evolution(df):
    """
    Generates a 2x2 grid of Slope vs Speed of Sound diagnostics.
    """
    set_paper_style()
    print("\n--- Generating Slope Evolution Diagnostics (Paper Style) ---")
    
    # Define Targets: We analyze the slope at 4 distinct mass steps
    targets = [
        {'col': 'Slope14', 'mass': '1.4'},
        {'col': 'Slope16', 'mass': '1.6'},
        {'col': 'Slope18', 'mass': '1.8'},
        {'col': 'Slope20', 'mass': '2.0'}
    ]
    
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Drop duplicates to plot one point per EoS Curve
    unique_stars = df.drop_duplicates(subset=['Curve_ID'])
    
    for i, target in enumerate(targets):
        ax = axes[i]
        col_name = target['col']
        mass_label = target['mass']
        
        # Filter NaNs (Stars that collapsed before reaching this mass step)
        plot_data = unique_stars.dropna(subset=[col_name, 'CS2_at_14'])
        
        # 1. Plot Hadronic (H_main)
        h_data = plot_data[plot_data['Label'] == 0]
        ax.scatter(h_data['CS2_at_14'], h_data[col_name], 
                   color=COLORS['H_main'], s=15, alpha=0.5, 
                   label='Hadronic', edgecolors='none', rasterized=True)
        
        # 2. Plot Quark (Q_main)
        q_data = plot_data[plot_data['Label'] == 1]
        ax.scatter(q_data['CS2_at_14'], q_data[col_name], 
                   color=COLORS['Q_main'], s=15, alpha=0.5, 
                   label='Quark (CFL)', edgecolors='none', rasterized=True)
        

        ax.set_xlim(CONSTANTS['PLOT_CS2_LIM'])
        ax.set_ylim(CONSTANTS['PLOT_SLOPE_LIM'])
        
        # Zero line: Separates expanding stars (dR/dM > 0) from compressing stars (dR/dM < 0)
        ax.axhline(0, color='black', linestyle=':', alpha=0.6, lw=1)
        
        # Labels and Style
        ax.set_xlabel(r"$c_s^2(r=0)$ at $1.4 M_{\odot}$")
        ax.set_ylabel(r"Slope $dR/dM$")
        
        
        ax.text(0.05, 0.05, f"$M = {mass_label} M_{{\odot}}$", 
                transform=ax.transAxes, fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))
        
        
        if i == 0:
            ax.legend(loc='upper right', frameon=True, markerscale=2.0)

    plt.tight_layout()
    plt.savefig("plots/fig_slope_evolution_paper_style.pdf")
    plt.close()
    print("[Success] Saved Slope Evolution Plot (Paper Style).")
