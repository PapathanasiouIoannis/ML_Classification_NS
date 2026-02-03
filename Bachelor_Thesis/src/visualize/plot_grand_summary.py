
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from tqdm import tqdm
from matplotlib.patches import Patch
from src.visualize.style_config import set_paper_style, COLORS
from src.const import CONSTANTS

def plot_grand_summary(df):
    """
    Generates the 3-panel Grand Summary plot.
    """
    set_paper_style()
    print(f"\n--- Generating 'Grand Summary' (Final Corrected) ---")

    # ==========================================
    # 0. CONFIGURATION & GRIDS
    # ==========================================
    # Extended grids to ensure no clipping during interpolation
    eps_grid = np.logspace(np.log10(CONSTANTS['PLOT_EPS_LOG'][0]), 
                           np.log10(CONSTANTS['PLOT_EPS_LOG'][1]), 500)
    
    mass_grid = np.linspace(CONSTANTS['PLOT_M_LIM'][0] + 0.1, 
                            CONSTANTS['PLOT_M_LIM'][1], 300)
    
    # Storage arrays
    eos_h_stack, eos_q_stack = [], []
    lam_h_stack, lam_q_stack = [], []
    
    grouped = df.groupby('Curve_ID')
    curve_ids = list(grouped.groups.keys())
    
    # Subsample curves if dataset is too large (optimization)
    np.random.seed(42)
    limit = 3000
    if len(curve_ids) > limit:
        selected_ids = set(np.random.choice(curve_ids, limit, replace=False))
    else:
        selected_ids = set(curve_ids)

    # ==========================================
    # 1. DATA PROCESSING
    # ==========================================
    print("Interpolating curves...")
    
    for name, group in tqdm(grouped, desc="Processing"):
        if name not in selected_ids: continue
        
        g = group.sort_values(by='Eps_Central')
        idx_max = g['Mass'].idxmax()
        label = g['Label'].iloc[0]
        
        # --- A. EoS Processing ---
        e_vals = g['Eps_Central'].values
        p_vals = g['P_Central'].values
        
        if len(e_vals) > 5:
            # Interpolate Log-Log
            f_eos = interp1d(np.log10(e_vals), np.log10(p_vals), bounds_error=False, fill_value=np.nan)
            p_interp_log = f_eos(np.log10(eps_grid))
            p_interp = 10**p_interp_log
            
            if label == 0:
                eos_h_stack.append(p_interp)
            else:
                eos_q_stack.append(p_interp)

        # --- B. Lambda Processing ---
        # Only use the stable branch up to M_max
        g_stable = g.loc[:idx_max].drop_duplicates(subset='Mass').sort_values('Mass')
        
        if len(g_stable) > 5:
            m_vals = g_stable['Mass'].values
            l_vals = g_stable['Lambda'].values
            l_vals = np.maximum(l_vals, 1e-5) # Avoid log(0)
            
            f_lam = interp1d(m_vals, np.log10(l_vals), bounds_error=False, fill_value=np.nan)
            l_interp = f_lam(mass_grid)
            
            if label == 0:
                lam_h_stack.append(l_interp)
            else:
                lam_q_stack.append(l_interp)

    eos_h_stack = np.array(eos_h_stack)
    eos_q_stack = np.array(eos_q_stack)
    lam_h_stack = np.array(lam_h_stack)
    lam_q_stack = np.array(lam_q_stack)

    # ==========================================
    # 2. PLOTTING
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.25)

    def plot_smooth_band(ax, x_grid, data_stack, color, label, hatch=None, ls='-', is_hatch_only=False, clip_to_causal=False):
        """Helper to plot smoothed median and percentile bands."""
        low  = np.nanpercentile(data_stack, 5, axis=0)
        med  = np.nanpercentile(data_stack, 50, axis=0)
        high = np.nanpercentile(data_stack, 95, axis=0)
        
        # Gaussian Smoothing
        sigma = 3.50
        low_smooth = gaussian_filter1d(low, sigma)
        med_smooth = gaussian_filter1d(med, sigma)
        high_smooth = gaussian_filter1d(high, sigma)
        
        # Visual Causality Clamp (P <= Epsilon)
        if clip_to_causal:
            med_smooth = np.minimum(med_smooth, x_grid)
            high_smooth = np.minimum(high_smooth, x_grid)
            low_smooth = np.minimum(low_smooth, x_grid)

        ax.plot(x_grid, med_smooth, color=color, linestyle=ls, linewidth=2.0, label=label)
        ax.plot(x_grid, low_smooth, color=color, linestyle=ls, linewidth=0.5, alpha=0.6)
        ax.plot(x_grid, high_smooth, color=color, linestyle=ls, linewidth=0.5, alpha=0.6)
        
        if is_hatch_only:
            ax.fill_between(x_grid, low_smooth, high_smooth, 
                            facecolor='none', edgecolor=color, 
                            hatch=hatch, linewidth=0.0, alpha=0.5)
        else:
            ax.fill_between(x_grid, low_smooth, high_smooth, 
                            facecolor=color, alpha=0.25, 
                            edgecolor=None, linewidth=0.0)

    # --- PANEL A: Equation of State ---
    ax = axes[0]
    plot_smooth_band(ax, eps_grid, eos_h_stack, COLORS['H_main'], 'Hadronic', ls='-', clip_to_causal=True)
    plot_smooth_band(ax, eps_grid, eos_q_stack, COLORS['Q_main'], 'Quark (CFL)', hatch='////', ls='--', is_hatch_only=True, clip_to_causal=True)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(CONSTANTS['PLOT_EPS_LOG'])
    ax.set_ylim(1e0, 2e3)
    ax.set_xlabel(r"Energy Density $\varepsilon$ [MeV/fm$^3$]")
    ax.set_ylabel(r"Pressure $P$ [MeV/fm$^3$]")
    ax.set_title(r"(a) Equation of State")
    
    # Physics Constraints
    x_guide = np.logspace(1, 4, 50)
    # Causal (c_s = 1) -> P = epsilon
    ax.plot(x_guide, x_guide, 'k:', alpha=0.6, lw=1.5, label='Causal ($c_s=1$)')
    # Conformal (c_s^2 = 1/3) -> P = epsilon/3
    ax.plot(x_guide, x_guide/3.0, color='gray', linestyle=':', alpha=0.5, lw=1.5, label='Conformal ($c_s^2=1/3$)')

    # --- PANEL B: Mass-Radius ---
    ax = axes[1]
    
    h_r = df[df['Label']==0]['Radius']
    h_m = df[df['Label']==0]['Mass']
    q_r = df[df['Label']==1]['Radius']
    q_m = df[df['Label']==1]['Mass']
    
    xx, yy = np.mgrid[4:22:300j, 0:4.5:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Hadronic KDE
    if len(h_r) > 1000:
        if len(h_r) > 20000:
            idx = np.random.choice(len(h_r), 20000, replace=False)
            h_r_s, h_m_s = h_r.iloc[idx], h_m.iloc[idx]
        else:
            h_r_s, h_m_s = h_r, h_m
            
        kernel_h = gaussian_kde(np.vstack([h_r_s, h_m_s]), bw_method='scott')
        kernel_h.set_bandwidth(kernel_h.factor * 1.5) 
        
        f_h = np.reshape(kernel_h(positions).T, xx.shape)
        levels_h = [0.05 * f_h.max(), f_h.max()]
        ax.contourf(xx, yy, f_h, levels=levels_h, colors=[COLORS['H_main']], alpha=0.3)
        ax.contour(xx, yy, f_h, levels=levels_h[:1], colors=[COLORS['H_main']], linewidths=2)
    
    # Quark KDE
    if len(q_r) > 1000:
        if len(q_r) > 20000:
            idx = np.random.choice(len(q_r), 20000, replace=False)
            q_r_s, q_m_s = q_r.iloc[idx], q_m.iloc[idx]
        else:
            q_r_s, q_m_s = q_r, q_m
            
        kernel_q = gaussian_kde(np.vstack([q_r_s, q_m_s]))
        kernel_q.set_bandwidth(kernel_q.factor * 1.3)
        
        f_q = np.reshape(kernel_q(positions).T, xx.shape)
        levels_q = [0.05 * f_q.max(), f_q.max()]
        
        # Hatched contour
        cntr = ax.contourf(xx, yy, f_q, levels=levels_q, colors='none', hatches=['////'])
        for collection in cntr.collections:
            collection.set_edgecolor(COLORS['Q_main'])
            collection.set_linewidth(0.0)
            collection.set_alpha(0.5)
        ax.contourf(xx, yy, f_q, levels=levels_q, colors=[COLORS['Q_main']], alpha=0.1)
        ax.contour(xx, yy, f_q, levels=levels_q[:1], colors=[COLORS['Q_main']], linewidths=2, linestyles='--')

    # Constraints
    ax.axhline(2.08, color='black', linestyle='-.', lw=1.5, label='J0740')
    
    # Buchdahl Limit: R < 9/4 G M / c^2
    m_buch = np.linspace(0, 4.5, 100)
    r_buch = CONSTANTS['BUCHDAHL_FACTOR'] * CONSTANTS['A_CONV'] * m_buch
    
    ax.fill_betweenx(m_buch, 0, r_buch, color='gray', alpha=0.1, zorder=-10)
    ax.text(6.0, 3.8, "GR Forbidden\n(Buchdahl)", fontsize=9, color='gray', ha='center')

    ax.set_xlim(CONSTANTS['PLOT_R_LIM'])
    ax.set_ylim(CONSTANTS['PLOT_M_LIM'])
    ax.set_xlabel(r"Radius $R$ [km]")
    ax.set_ylabel(r"Mass $M$ [$M_{\odot}$]")
    ax.set_title(r"(b) Mass-Radius Relation")

    # --- PANEL C: Tidal Deformability ---
    ax = axes[2]
    
    lam_h_lin = 10**lam_h_stack
    lam_q_lin = 10**lam_q_stack
    
    plot_smooth_band(ax, mass_grid, lam_h_lin, COLORS['H_main'], 'Hadronic', ls='-', is_hatch_only=False)
    plot_smooth_band(ax, mass_grid, lam_q_lin, COLORS['Q_main'], 'Quark', hatch='////', ls='--', is_hatch_only=True)
    
    ax.set_yscale('log')
    ax.set_xlim(CONSTANTS['PLOT_M_LIM'])
    ax.set_ylim(1, 5000)
    ax.set_xlabel(r"Mass $M$ [$M_{\odot}$]")
    ax.set_ylabel(r"Tidal Deformability $\Lambda$")
    ax.set_title(r"(c) Tidal Deformability")
    
    # GW170817 Constraint (Upper Bound at 1.4 M_sun)
    # Visualizing Lambda(1.4) < 580
    ax.vlines(1.36, 1, 580, colors='black', lw=2)
    ax.hlines(580, 1.36, 1.6, colors='black', lw=2) 
    ax.text(1.45, 650, "GW170817", fontsize=10)

    # Global Legend
    h_handle = Patch(facecolor=COLORS['H_main'], alpha=0.3, label='Hadronic')
    q_handle = Patch(facecolor=COLORS['Q_main'], hatch='////', alpha=0.2, edgecolor=COLORS['Q_main'], label='Quark (CFL)')
    
    fig.legend(handles=[h_handle, q_handle], loc='upper center', 
               bbox_to_anchor=(0.5, 0.08), ncol=2, frameon=False, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18) 
    plt.savefig("plots/fig_grand_summary.pdf")
    plt.close()
    print("[Success] Saved Grand Summary Plot (Final).")