

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS

def plot_correlations(df):
    """
    Generates Figures 16-18: Micro-Macro Correlation Plots.
    """
    set_paper_style()
    print("\n--- Generating Micro-Macro Correlation Plots ---")
    
    # Verify columns exist
    if 'Eps_Central' not in df.columns:
        print("[Warn] Microphysics columns not found. Skipping Figs 16-18.")
        return

    # Palette Map
    palette_map = {0: COLORS['H_main'], 1: COLORS['Q_main']}
    
    # ==============================================================
    # FIGURE 16: Central Density vs Radius
    # ==============================================================
    print("   Plotting Fig 16: Density vs Radius...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Subsample 10k points for performance (Scatter is slow on 1.5M)
    df_sample = df.sample(n=min(10000, len(df)), random_state=42)
    
    sns.scatterplot(data=df_sample, x='Eps_Central', y='Radius', hue='Label', 
                    palette=palette_map, alpha=0.4, s=10, ax=ax, edgecolor='none')
    
    ax.set_xlabel(r"Central Energy Density $\varepsilon_c$ [MeV/fm$^3$]")
    ax.set_ylabel(r"Radius $R$ [km]")
    ax.set_title(r"Micro-Macro: Core Density vs Surface Radius")
    ax.set_xscale('log') 
    
    # Limits
    ax.set_xlim(1e2, 5e3) # Matches CONSTANTS['PLOT_EPS_LOG'] roughly
    ax.set_ylim(CONSTANTS['PLOT_R_LIM'])

    # Reference Line: Saturation Density (~150 MeV/fm^3)
    ax.axvline(150, color='gray', linestyle=':', label=r'$\rho_{sat}$')
    
    # Custom Legend
    handles, labels = ax.get_legend_handles_labels()
    label_map = {'0': 'Hadronic', '1': 'Quark', 0: 'Hadronic', 1: 'Quark'}
    clean_labels = [label_map.get(l, l) for l in labels]
    ax.legend(handles, clean_labels, loc='upper right')
    
    plt.savefig("plots/fig_16_density_radius.pdf")
    plt.close()

    # ==============================================================
    # FIGURE 17: Stiffness (Sound Speed) vs Maximum Mass
    # ==============================================================
    print("   Plotting Fig 17: Stiffness vs Max Mass...")
    

    # 1. Sort by Mass Descending
    df_sorted = df.sort_values('Mass', ascending=False)
    
    # 2. Drop duplicates on Curve_ID, keeping the first (heaviest) one
    max_mass_stars = df_sorted.drop_duplicates(subset=['Curve_ID'])
    
    # 3. Subsample if too large
    if len(max_mass_stars) > 5000:
        max_mass_stars = max_mass_stars.sample(5000, random_state=42)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.scatterplot(data=max_mass_stars, x='CS2_Central', y='Mass', hue='Label', 
                    palette=palette_map, alpha=0.6, s=25, ax=ax, edgecolor='none')
    
    ax.set_xlabel(r"Central Sound Speed Squared $c_s^2$ (at $M_{max}$)")
    ax.set_ylabel(r"Maximum Mass $M_{max}$ [$M_{\odot}$]")
    ax.set_title(r"Stiffness Limit: Sound Speed vs Max Mass")
    
    # Physical Limits
    ax.set_xlim(CONSTANTS['PLOT_CS2_LIM'])
    ax.set_ylim(1.5, 3.5) # Focus on high mass region
    
    # Constraints
    ax.axhline(2.08, color='black', linestyle=':', label='J0740 Lower Bound')
    ax.axvline(1.0/3.0, color='gray', linestyle='--', alpha=0.7, label='Conformal Limit')
    ax.axvline(1.0, color='gray', linestyle='-', alpha=0.5, label='Causal Limit')
    
    ax.legend(loc='lower right')
    
    plt.savefig("plots/fig_17_stiffness_maxmass.pdf")
    plt.close()

    # ==============================================================
    # FIGURE 18: Topological Slope vs Tidal Deformability
    # ==============================================================
    print("   Plotting Fig 18: Slope vs Lambda...")
    
    # Focus on the canonical mass region (1.4 M_sun +/- 0.05)
    df_14 = df[np.abs(df['Mass'] - 1.4) < 0.05].copy()
    df_14 = df_14.dropna(subset=['Slope14'])
    
    # Subsample
    if len(df_14) > 10000:
        df_14 = df_14.sample(10000, random_state=42)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.scatterplot(data=df_14, x='Slope14', y='Lambda', hue='Label', 
                    palette=palette_map, alpha=0.4, s=15, ax=ax, edgecolor='none')
    
    ax.set_xlabel(r"Slope at $1.4 M_{\odot}$ ($dR/dM$)")
    ax.set_ylabel(r"Tidal Deformability $\Lambda_{1.4}$")
    ax.set_title(r"Topological Signature: Slope vs Deformability")
    
    # Reference Line (Slope=0)
    ax.axvline(0, color='black', linestyle='-', linewidth=1, label='Zero Slope')
    ax.set_xlim(CONSTANTS['PLOT_SLOPE_LIM'])
    ax.set_ylim(0, 1000) # Focus on Lambda < 1000 
    
    ax.legend(loc='upper right')
    
    plt.savefig("plots/fig_18_slope_lambda.pdf")
    plt.close()
    
    print("[Success] Saved Figures 16-18.")
