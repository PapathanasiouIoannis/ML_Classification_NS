

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS

def plot_slope_vs_radius(df):
    """
    Generates a 2x2 grid of Slope vs Radius diagnostics at different mass steps.
    """
    set_paper_style()
    print("\n--- Generating Slope vs Radius Diagnostics ---")
    
    # Check if the specific diagnostic column exists
    if 'Radius_14' not in df.columns:
        print("[Warn] 'Radius_14' column missing. Skipping plot.")
        return

    targets = [
        {'col': 'Slope14', 'mass': '1.4'}, 
        {'col': 'Slope16', 'mass': '1.6'},
        {'col': 'Slope18', 'mass': '1.8'}, 
        {'col': 'Slope20', 'mass': '2.0'}
    ]
    
    # 2x2 Grid Layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Filter to unique EoS curves to avoid plotting 300 points per star
    unique_stars = df.drop_duplicates(subset=['Curve_ID'])
    
    for i, target in enumerate(targets):
        ax = axes[i]
        col = target['col']
        # Filter NaNs (Stars that collapsed before reaching this mass step)
        data = unique_stars.dropna(subset=[col, 'Radius_14'])
        
        # 1. Plot Hadronic Population
        h_data = data[data['Label']==0]
        ax.scatter(h_data['Radius_14'], h_data[col], 
                   color=COLORS['H_main'], s=15, alpha=0.5, 
                   label='Hadronic', edgecolors='none', rasterized=True)
        
        # 2. Plot Quark Population
        q_data = data[data['Label']==1]
        ax.scatter(q_data['Radius_14'], q_data[col], 
                   color=COLORS['Q_main'], s=15, alpha=0.5, 
                   label='Quark', edgecolors='none', rasterized=True)
        
        # --- Formatting ---
        # Match Axes to global plotting standards
        ax.set_xlim(CONSTANTS['PLOT_R_LIM'])
        ax.set_ylim(CONSTANTS['PLOT_SLOPE_LIM'])
        
        # Stability Reference Line
        ax.axhline(0, color='black', linestyle=':', alpha=0.6, lw=1)
        
        # Text Tag
        ax.text(0.05, 0.05, f"$M = {target['mass']} M_{{\odot}}$", 
                transform=ax.transAxes, fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))
        
        ax.set_xlabel(r"Radius $R_{1.4}$ [km]")
        ax.set_ylabel(r"Slope $dR/dM$")
        
        # Legend 
        if i == 0: 
            # Scale up markers in legend for visibility
            ax.legend(loc='upper right', frameon=True, markerscale=2.0)

    plt.tight_layout()
    plt.savefig("plots/fig_slope_vs_radius_paper_style.pdf")
    plt.close()
    print("[Success] Saved Slope vs Radius Plot.")
