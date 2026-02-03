

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS

def plot_universal_relations(df):
    """
    Plots Tidal Deformability (Lambda) vs Compactness (C = GM/Rc^2).
    Checks adherence to the I-Love-Q universal relations.
    """
    set_paper_style()
    print("\n--- Generating Universal Relations Check (I-Love-Q) ---")
    
    # 1. Calculate Compactness
    # C = (G * M / c^2) / R
    # A_CONV = G * M_sun / c^2
    df['Compactness'] = CONSTANTS['A_CONV'] * (df['Mass'] / df['Radius'])
    
    if 'LogLambda' not in df.columns:
        df['LogLambda'] = np.log10(df['Lambda'])
        
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Subsample for plotting speed
    plot_data = df.sample(min(5000, len(df)), random_state=42)
    
    # 2. Plot Hadronic (Standard Track)
    h_data = plot_data[plot_data['Label'] == 0]
    ax.scatter(h_data['Compactness'], h_data['LogLambda'], 
               c=COLORS['H_main'], s=15, alpha=0.2, 
               label='Hadronic (Standard)', edgecolors='none')
    
    # 3. Plot Quark (CFL)
    q_data = plot_data[plot_data['Label'] == 1]
    ax.scatter(q_data['Compactness'], q_data['LogLambda'], 
               c=COLORS['Q_main'], s=15, alpha=0.2, 
               label='Quark (CFL)', edgecolors='none')
    
    # 4. Formatting
    ax.set_xlabel(r"Compactness $C = GM/Rc^2$")
    ax.set_ylabel(r"Log Tidal Deformability $\log_{10}\Lambda$")
    ax.set_title(r"Universal Relations: Compactness vs Deformability")
    
    # Physical Limits
    ax.axvline(0.5, color='black', linestyle='-', linewidth=1.5, label='Black Hole Limit')
    # Buchdahl Limit for C = M/R (geometric) -> 4/9 = 0.444
    ax.axvline(0.444, color='gray', linestyle=':', linewidth=1.5, label='Buchdahl Limit')
    
    ax.set_xlim(0.0, 0.6)
    ax.set_ylim(0, 5)
    
    ax.legend(loc='upper right', frameon=True)
    
    plt.savefig("plots/fig_universal_relations.pdf")
    plt.close()
    print("[Success] Saved Universal Relations Plot.")


def plot_misclassification_map(models_dict, X_test_all, y_test):
    """
    Visualizes the 'Geography of Failure' across the Model Hierarchy (Geo -> D).
    Uses a 2x3 Grid to show how adding physics features reduces error rates.
    """
    set_paper_style()
    print("\n--- Generating 5-Panel Misclassification Evolution (Including Geo) ---")
    
    # Feature definitions
    feature_sets = {
        'Geo': ['Mass', 'Radius'],
        'A': ['Mass', 'Radius', 'LogLambda'],
        'B': ['Mass', 'Radius', 'LogLambda', 'Eps_Central'],
        'C': ['Mass', 'Radius', 'LogLambda', 'Eps_Central', 'CS2_Central'],
        'D': ['Mass', 'Radius', 'LogLambda', 'Eps_Central', 'CS2_Central', 'Slope14']
    }
    
    model_names = {
        'Geo': 'Model Geo: M-R Only',
        'A': 'Model A: +Tidal (Obs)',
        'B': 'Model B: +Density',
        'C': 'Model C: +Sound Speed',
        'D': 'Model D: +Topology'
    }
    
    # Order of evolution
    model_keys = ['Geo', 'A', 'B', 'C', 'D']

    # Setup 2x3 Grid (Width=18, Height=12)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    
    # Common Physics Data
    m_buch = np.linspace(0, 4.5, 100)
    # R_buch = (9/4) * A_CONV * M
    r_buch = CONSTANTS['BUCHDAHL_FACTOR'] * CONSTANTS['A_CONV'] * m_buch

    # Loop through models
    for i, key in enumerate(model_keys):
        ax = axes_flat[i]
        
        # Skip if model wasn't trained
        if key not in models_dict:
            ax.text(0.5, 0.5, f"{key} Not Trained", ha='center')
            continue
            
        model = models_dict[key]
        features = feature_sets[key]
        
        # 1. Predict
        X_slice = X_test_all[features]
        y_pred = model.predict(X_slice)
        
        # 2. Masks
        mask_correct_h = (y_test == 0) & (y_pred == 0)
        mask_correct_q = (y_test == 1) & (y_pred == 1)
        mask_fp = (y_test == 0) & (y_pred == 1) # H -> Q (Blue)
        mask_fn = (y_test == 1) & (y_pred == 0) # Q -> H (Orange)
        
        # 3. Plot Background (Correct)
        ax.scatter(X_test_all.loc[mask_correct_h, 'Radius'], X_test_all.loc[mask_correct_h, 'Mass'],
                   c=COLORS['H_main'], s=8, alpha=0.15, edgecolors='none', rasterized=True)
        
        ax.scatter(X_test_all.loc[mask_correct_q, 'Radius'], X_test_all.loc[mask_correct_q, 'Mass'],
                   c=COLORS['Q_main'], s=8, alpha=0.15, edgecolors='none', rasterized=True)
        
        # 4. Plot Errors (Foreground)
        ax.scatter(X_test_all.loc[mask_fp, 'Radius'], X_test_all.loc[mask_fp, 'Mass'],
                   marker='x', c='#004488', s=40, lw=1.5, label='False Pos (H→Q)')
        
        ax.scatter(X_test_all.loc[mask_fn, 'Radius'], X_test_all.loc[mask_fn, 'Mass'],
                   marker='x', c='#DDAA33', s=40, lw=1.5, label='False Neg (Q→H)')
        
        # 5. Physics Overlays
        ax.fill_betweenx(m_buch, 0, r_buch, color='gray', alpha=0.15, zorder=-5)
        ax.axhline(2.08, color='black', linestyle='-.', lw=1.5, alpha=0.5)
        
        # 6. Stats Box
        n_errors = mask_fp.sum() + mask_fn.sum()
        acc = 100 * (1 - n_errors/len(y_test))
        stats = f"Errors: {n_errors}\nAcc: {acc:.2f}%"
        ax.text(0.05, 0.95, stats, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.9, boxstyle='round'))
        
        ax.set_title(model_names[key], y=1.02)
        ax.grid(True, alpha=0.2)
        
        # Labels only on edges
        if i >= 2: ax.set_xlabel(r"Radius $R$ [km]")
        if i % 3 == 0: ax.set_ylabel(r"Mass $M$ [$M_{\odot}$]")
        
        ax.set_xlim(CONSTANTS['PLOT_R_LIM'])
        ax.set_ylim(CONSTANTS['PLOT_M_LIM'])

    # --- CLEANUP SLOT 6 ---
    # Hide the empty 6th subplot
    axes_flat[5].axis('off')
    
    # --- LEGEND IN SLOT 6 ---
    # We use the empty space for the legend instead of cramming it below
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['H_main'], label='Correct Hadronic'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['Q_main'], label='Correct Quark'),
        Line2D([0], [0], marker='x', color='#004488', linestyle='None', label='False Positive (H classified as Q)'),
        Line2D([0], [0], marker='x', color='#DDAA33', linestyle='None', label='False Negative (Q classified as H)'),
        Patch(facecolor='gray', alpha=0.2, label='GR Forbidden')
    ]
    
    # Place legend in the empty axes[5]
    axes_flat[5].legend(handles=legend_elements, loc='center', fontsize=12, 
                        frameon=False, title="Classification Legend")
    
    plt.tight_layout()
    plt.savefig("plots/fig_misclassification_map.pdf", dpi=300)
    plt.close()
    print("[Success] Saved 5-Panel Misclassification Map.")