
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import PartialDependenceDisplay
from src.visualize.style_config import set_paper_style, COLORS
from src.const import CONSTANTS

def plot_partial_dependence(models_dict, X_test_all):
    """
    Generates Partial Dependence Plots (PDP) for Model A and Model D.
    """
    set_paper_style()
    print("\n--- Generating Partial Dependence Plots (PDP) ---")
    
    # Common settings for the PDP lines
    common_params = {
        "kind": "average",      
        "grid_resolution": 100,  
        "percentiles": (0.00, 1.00),
        "n_jobs": -1
    }
    
    line_style = {"color": COLORS['Q_main'], "linewidth": 3}

    # =======================================================
    # PLOT 1: OBSERVABLES (Model A)
    # =======================================================
    if 'A' in models_dict:
        model_A = models_dict['A']
        features_A = ['Mass', 'Radius', 'LogLambda']
        X_A = X_test_all[features_A]
        
        fig, ax = plt.subplots(figsize=(14, 4))
        
        display = PartialDependenceDisplay.from_estimator(
            model_A, X_A, features_A,
            ax=ax, line_kw=line_style, **common_params
        )
        
        fig.suptitle(r"Partial Dependence: Observables (Model A)", y=1.05)
        
        axes = display.axes_[0]
        
        # 1. Mass Axis
        axes[0].set_xlabel(r"Mass $M$ [$M_{\odot}$]")
        axes[0].set_xlim(CONSTANTS['PLOT_M_LIM'])

        # 2. Radius Axis
        axes[1].set_xlabel(r"Radius $R$ [km]")
        axes[1].set_xlim(CONSTANTS['PLOT_R_LIM'])
        
        # 3. Tidal Axis
        axes[2].set_xlabel(r"Log Tidal $\log_{10}\Lambda$")
        axes[2].set_xlim(CONSTANTS['PLOT_L_LIM'])

        # Global Style Updates for Plot 1
        for sub_ax in axes:
            # Force Y-axis to 0-1 to show absolute probability impact
            sub_ax.set_ylim(0, 1.05)
            # Add Decision Boundary
            sub_ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
            # Grid
            sub_ax.grid(True, alpha=0.15)

        axes[0].set_ylabel(r"Probability $P(\text{Quark})$")
        
        plt.savefig("plots/fig_pdp_observables.pdf", bbox_inches='tight')
        plt.close()
    
    # =======================================================
    # PLOT 2: MICROPHYSICS & TOPOLOGY (Model D)
    # =======================================================
    if 'D' in models_dict:
        model_D = models_dict['D']
        features_D = ['Eps_Central', 'CS2_Central', 'Slope14']
        
        # Ensure we have the full feature set required by Model D
        cols_D = ['Mass', 'Radius', 'LogLambda', 'Eps_Central', 'CS2_Central', 'Slope14']
        X_D = X_test_all[cols_D]
        
        fig, ax = plt.subplots(figsize=(14, 4))
        
        display = PartialDependenceDisplay.from_estimator(
            model_D, X_D, features_D,
            ax=ax, line_kw=line_style, **common_params
        )
        
        fig.suptitle(r"Partial Dependence: Microphysics (Model D)", y=1.05)
        
        axes = display.axes_[0]
        
        # 1. Density Axis
        axes[0].set_xlabel(r"Central Density $\varepsilon_c$ [MeV/fm$^3$]")
        axes[0].set_xlim(CONSTANTS['PLOT_EPS_LIM'])
        
        # 2. Sound Speed Axis
        axes[1].set_xlabel(r"Sound Speed Squared $c_s^2$")
        axes[1].set_xlim(CONSTANTS['PLOT_CS2_LIM'])

        # 3. Slope Axis (The Topological Phase Transition)
        axes[2].set_xlabel(r"Slope $dR/dM$ at $1.4 M_{\odot}$")
        axes[2].set_xlim(CONSTANTS['PLOT_SLOPE_LIM'])
        
        # Physical Context Shading
        # Left of 0.0: Hadronic-like (Collapsing branch)
        axes[2].axvspan(CONSTANTS['PLOT_SLOPE_LIM'][0], 0.0, color=COLORS['H_main'], alpha=0.1)
        # Right of 0.0: Quark-like (Stable branch)
        axes[2].axvspan(0.0, CONSTANTS['PLOT_SLOPE_LIM'][1], color=COLORS['Q_main'], alpha=0.1)
        
        for sub_ax in axes:
            sub_ax.set_ylim(0, 1.05)
            sub_ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
            sub_ax.grid(True, alpha=0.15)

        axes[0].set_ylabel(r"Probability $P(\text{Quark})$")
        
        plt.savefig("plots/fig_pdp_microphysics.pdf", bbox_inches='tight')
        plt.close()
    
    print("[Success] Saved PDP Figures.")