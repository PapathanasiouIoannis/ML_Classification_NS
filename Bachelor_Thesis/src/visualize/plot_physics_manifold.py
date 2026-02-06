
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from src.visualize.style_config import set_paper_style, COLORS
from src.const import CONSTANTS

def plot_physics_manifold(df):
    """
    Generates the Triptych Manifold plot using KDE Contours (Statistical View).
    """
    set_paper_style()
    print(f"\n--- Generating Manifold Triptych (KDE) ---")
    
    # 1. Setup Data
    grouped = df.groupby('Curve_ID')
    curve_ids = list(grouped.groups.keys())
    
    # Subsample for Line Plotting (Background context only)
    # Using a fixed seed ensures consistent figures between runs
    np.random.seed(42)
    if len(curve_ids) > 1000:
        selected_ids = set(np.random.choice(curve_ids, 1000, replace=False))
    else:
        selected_ids = set(curve_ids)
    
    # 2. Initialize Triptych (1 Row, 3 Cols)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, sharex=True)
    plt.subplots_adjust(wspace=0.05) 
    
    # Accumulate points for KDE
    h_points_r, h_points_m = [], []
    q_points_r, q_points_m = [], []

    print("Rendering curves (Background)...")
    for name, group in tqdm(grouped, desc="Processing Curves"):
        # Sort by Density to ensure lines are drawn in physical order
        g = group.sort_values(by='Eps_Central')
        label = g['Label'].iloc[0]
        
        # Plot Faint Lines (Background Context)
        if name in selected_ids:
            if label == 0:
                axes[0].plot(g['Radius'], g['Mass'], color=COLORS['H_main'], 
                             alpha=0.05, lw=0.5, rasterized=True)
                axes[2].plot(g['Radius'], g['Mass'], color=COLORS['H_main'], 
                             alpha=0.01, lw=0.3, rasterized=True)
            if label == 1:
                axes[1].plot(g['Radius'], g['Mass'], color=COLORS['Q_main'], 
                             alpha=0.05, lw=0.5, rasterized=True)
                axes[2].plot(g['Radius'], g['Mass'], color=COLORS['Q_main'], 
                             alpha=0.01, lw=0.3, rasterized=True)
        
        # Collect Points for KDE (Stable branch only)
        m_max = g['Mass'].max()
        stable_g = g[g['Mass'] <= m_max] 
        
        if label == 0:
            h_points_r.extend(stable_g['Radius'].values)
            h_points_m.extend(stable_g['Mass'].values)
        else:
            q_points_r.extend(stable_g['Radius'].values)
            q_points_m.extend(stable_g['Mass'].values)

    # 3. Generate KDE Contours
    print("Generating Phase Space Contours...")
    
    # Define evaluation grid based on CONSTANTS plotting limits
    r_min, r_max = CONSTANTS['PLOT_R_LIM']
    m_min, m_max = CONSTANTS['PLOT_M_LIM']
    
    xx, yy = np.mgrid[r_min:r_max:300j, m_min:m_max:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    def draw_contours(ax, r_pts, m_pts, color, is_hatched=False):
        if len(r_pts) < 100: return
        
        # Subsample for KDE speed if dataset is massive
        if len(r_pts) > 30000:
            idx = np.random.choice(len(r_pts), 30000, replace=False)
            r_pts = np.array(r_pts)[idx]
            m_pts = np.array(m_pts)[idx]
            
        kernel = gaussian_kde(np.vstack([r_pts, m_pts]))
        # Adjust bandwidth for smoothness
        kernel.set_bandwidth(kernel.factor)
        
        f = np.reshape(kernel(positions).T, xx.shape)
        f_max = f.max()
        
        # Define levels relative to peak density
        levels = [0.01 * f_max, 0.1 * f_max, 0.5 * f_max, f_max]
        
        if is_hatched:
            # Quark Style: Hatched with transparent fill
            cntr = ax.contourf(xx, yy, f, levels=levels, colors='none', hatches=['////'])
            for c in cntr.collections:
                c.set_edgecolor(color)
                c.set_linewidth(0.0)
                c.set_alpha(0.5)
            ax.contourf(xx, yy, f, levels=levels, colors=[color], alpha=0.1)
        else:
            # Hadronic Style: Solid fill
            ax.contourf(xx, yy, f, levels=levels, colors=[color], alpha=0.3)
            
        ls = '--' if is_hatched else '-'
        ax.contour(xx, yy, f, levels=levels[:1], colors=[color], linewidths=2, linestyles=ls)

    # --- DRAW ON PANELS ---
    draw_contours(axes[0], h_points_r, h_points_m, COLORS['H_main'], is_hatched=False)
    draw_contours(axes[1], q_points_r, q_points_m, COLORS['Q_main'], is_hatched=True)
    draw_contours(axes[2], h_points_r, h_points_m, COLORS['H_main'], is_hatched=False)
    draw_contours(axes[2], q_points_r, q_points_m, COLORS['Q_main'], is_hatched=True)

    # 4. Add Constraints & Formatting
    _apply_common_formatting(axes)

    # Specific Labels
    axes[0].set_title(r"(a) Hadronic Models (KDE)", y=1.02)
    axes[1].set_title(r"(b) Quark Models (KDE)", y=1.02)
    axes[2].set_title(r"(c) Phase Space Intersection", y=1.02)

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS['H_main'], alpha=0.3, label='Hadronic (95% CI)'),
        Patch(facecolor=COLORS['Q_main'], hatch='////', alpha=0.2, 
              edgecolor=COLORS['Q_main'], label='Quark (95% CI)')
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig("plots/fig_1_manifold_triptych.pdf")
    plt.close()
    print("[Success] Saved KDE Manifold.")


def plot_manifold_curves(df):
    """
    Generates the Triptych Manifold plot using All Curves (Morphological View).
    Highlights outliers and the full morphological flow.
    """
    set_paper_style()
    print(f"\n--- Generating Manifold Curves (Full Data) ---")
    
    # 1. Setup Data
    grouped = df.groupby('Curve_ID')
    
    # 2. Initialize Triptych
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, sharex=True)
    plt.subplots_adjust(wspace=0.05)
    
    print(f"Rendering {len(grouped)} curves (Rasterized)...")
    
    # 3. Plot Every Single Curve
    for name, group in tqdm(grouped, desc="Plotting Lines"):
        # Sort points by density/radius to ensure clean lines
        g = group.sort_values(by='Eps_Central')
        label = g['Label'].iloc[0]
        
        # Hadronic: Green
        if label == 0:
            # Panel 1 (Hadronic Only)
            axes[0].plot(g['Radius'], g['Mass'], 
                         color=COLORS['H_main'], alpha=0.2, lw=0.8, rasterized=True)
            # Panel 3 (Combined)
            axes[2].plot(g['Radius'], g['Mass'], 
                         color=COLORS['H_main'], alpha=0.15, lw=0.6, rasterized=True)
            
        # Quark: Magenta
        elif label == 1:
            # Panel 2 (Quark Only)
            axes[1].plot(g['Radius'], g['Mass'], 
                         color=COLORS['Q_main'], alpha=0.2, lw=0.8, rasterized=True)
            # Panel 3 (Combined)
            axes[2].plot(g['Radius'], g['Mass'], 
                         color=COLORS['Q_main'], alpha=0.15, lw=0.6, rasterized=True)

    # 4. Apply Common Constraints
    _apply_common_formatting(axes)
    
    # Labels
    axes[0].set_title(r"(a) Hadronic Population (All)", y=1.02)
    axes[1].set_title(r"(b) Quark Population (All)", y=1.02)
    axes[2].set_title(r"(c) Full Population Overlay", y=1.02)
    
    # Custom Legend for Lines
    legend_elements = [
        Line2D([0], [0], color=COLORS['H_main'], lw=2, label='Hadronic EoS'),
        Line2D([0], [0], color=COLORS['Q_main'], lw=2, label='Quark EoS')
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', frameon=True)

    # Save
    outfile = "plots/fig_1_manifold_curves.pdf"
    plt.tight_layout()
    plt.savefig(outfile, dpi=300) 
    plt.close()
    print(f"[Success] Saved Curve Manifold to {outfile}")


def _apply_common_formatting(axes):
    """
    Helper to apply physical constraints (Buchdahl, J0740) and standard axis labels.
    """
    m_buch = np.linspace(0, 4.5, 100)
    # Buchdahl Limit: R < 9/4 G M / c^2
    # R_buch = (9/4) * A_CONV * M [M_sun]
    r_buch = CONSTANTS['BUCHDAHL_FACTOR'] * CONSTANTS['A_CONV'] * m_buch
    
    for ax in axes:
        # J0740 Limit (2.08 M_sun)
        ax.axhline(2.08, color=COLORS['Constraint'], linestyle='-.', linewidth=1.5, zorder=10)
        
        # Buchdahl Forbidden Wedge
        ax.fill_betweenx(m_buch, 0, r_buch, color='gray', alpha=0.1, zorder=-1)
        
        # Grid
        ax.grid(True, which='major', alpha=0.2)
        
        # Limits from CONSTANTS
        ax.set_xlim(CONSTANTS['PLOT_R_LIM'])
        ax.set_ylim(CONSTANTS['PLOT_M_LIM'])
        ax.set_xlabel(r"Radius $R$ [km]")

    axes[0].set_ylabel(r"Mass $M$ [$M_{\odot}$]")
    axes[0].text(8.0, 2.15, r"PSR J0740", fontsize=9, color=COLORS['Constraint'])
    axes[0].text(6.0, 2.8, "GR Forbidden", fontsize=8, color='gray', rotation=45)
