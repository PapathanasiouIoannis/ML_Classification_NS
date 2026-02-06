import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from src.visualize.style_config import set_paper_style, COLORS

def plot_decision_mechanics(df):
    """
    Visualizes the Random Forest's decision boundaries as smooth probability fields.
    Covers Observables (M-R, M-L), Microphysics (Eps-Cs2), and Topology (R-Slope).
    """
    set_paper_style()
    print("\n--- Generating Decision Mechanism Visualizations (Smoothed & High Contrast) ---")
    
    # 1. Setup Custom Diverging Colormap
    # Hadronic (Green) -> Uncertain (White) -> Quark (Magenta)
    # We use a slightly off-white middle to keep the plot looking rich
    div_cmap = LinearSegmentedColormap.from_list(
        "DivProb", 
        [COLORS['H_main'], "#FFFFFF", COLORS['Q_main']]
    )

    # 2. Data Prep
    if 'LogLambda' not in df.columns:
        df['LogLambda'] = np.log10(df['Lambda'])

    # --- PLOTTING HELPER ---
    def plot_boundary(x_col, y_col, x_label, y_label, x_lim, y_lim, filename, 
                      use_unique_curves=False, overlay_func=None):
        
        # A. Data Selection
        if use_unique_curves:
            data = df.drop_duplicates(subset=['Curve_ID']).copy()
        else:
            data = df.copy()
            
        data = data.dropna(subset=[x_col, y_col])
        X = data[[x_col, y_col]]
        y = data['Label']
        
        # B. Train "Toy" Model
        # A focused RF to visualize 2D decision logic
        rf = RandomForestClassifier(n_estimators=1000, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # C. Generate Grid for Contours
        res = 400 # High Resolution
        xx = np.linspace(x_lim[0], x_lim[1], res)
        yy = np.linspace(y_lim[0], y_lim[1], res)
        XX, YY = np.meshgrid(xx, yy)
        grid_points = np.c_[XX.ravel(), YY.ravel()]
        
        # D. Predict Probability Field
        Z = rf.predict_proba(grid_points)[:, 1]
        Z = Z.reshape(XX.shape)
        
        
        # Apply Gaussian filter to melt "blocky" RF artifacts into a smooth gradient
        Z_smooth = gaussian_filter(Z, sigma=2.0)
        
        # E. Plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 1. The Probability Gradient (No black contour lines)
        contour = ax.contourf(XX, YY, Z_smooth, levels=100, cmap=div_cmap, vmin=0, vmax=1, alpha=0.85)
        
        # 2. Sparse Data Scatter (High Contrast)
        # Plot a small random sample
        sample = data.sample(min(4000, len(data)), random_state=42)
        
        # Hadronic: Blue Circle with White Halo
        h_data = sample[sample['Label']==0]
        ax.scatter(h_data[x_col], h_data[y_col],
                   c='#0077BB', edgecolors='white', linewidths=0.2,
                   s=10, alpha=0.9, marker='o', label='Hadronic')
        
        # Quark: Red Diamond with White Halo
        q_data = sample[sample['Label']==1]
        ax.scatter(q_data[x_col], q_data[y_col],
                   c='#CC3311', edgecolors='white', linewidths=0.2,
                   s=10, alpha=0.9, marker='D', label='Quark')

        # 3. Physics Overlays (Opaque Masks)
        if overlay_func:
            overlay_func(ax)
            
        # 4. Formatting
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Decision Manifold: {x_label} vs {y_label}")
        
        # Colorbar
        cbar = plt.colorbar(contour, ax=ax, ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(['Hadronic', 'Uncertain', 'Quark'])
        
        # Legend (Upper Right)
        ax.legend(loc='upper right', frameon=True, framealpha=0.9)
        
        plt.savefig(f"plots/{filename}")
        plt.close()

    # ==============================================================
    # PLOT 1: Mass vs Radius (Macroscopic)
    # ==============================================================
    def overlay_mr(ax):
        # Buchdahl Limit (R < 3.32 M) -> Masked Opaque
        m_vals = np.linspace(0, 4.5, 100)
        r_buch = 3.32 * m_vals
        # Opaque light grey to hide ML artifacts in forbidden region
        ax.fill_betweenx(m_vals, 0, r_buch, color='#E0E0E0', alpha=1.0, zorder=5)
        ax.text(6.5, 3.5, "GR Forbidden", color='gray', fontsize=10, ha='center', zorder=6)
        
        # J0740
        ax.axhline(2.08, color='black', linestyle='-.', alpha=0.5, zorder=6)

    plot_boundary(
        x_col='Radius', y_col='Mass',
        x_label=r'Radius $R$ [km]', y_label=r'Mass $M$ [$M_{\odot}$]',
        x_lim=(5.0, 20.0), y_lim=(0.0, 4.0),
        filename='fig_10a_mechanism_MR.pdf',
        overlay_func=overlay_mr
    )

    # ==============================================================
    # PLOT 2: Mass vs Log Lambda (Gravitational Waves)
    # ==============================================================
    plot_boundary(
        x_col='Mass', y_col='LogLambda',
        x_label=r'Mass $M$ [$M_{\odot}$]', y_label=r'Log Tidal $\log_{10}\Lambda$',
        x_lim=(0.0, 4.0), y_lim=(0.0, 5.0),
        filename='fig_10b_mechanism_MLambda.pdf'
    )

    # ==============================================================
    # PLOT 3: Central Density vs Sound Speed (Microphysics)
    # ==============================================================
    def overlay_micro(ax):
        # Causal Limit (cs2 = 1)
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5)
        ax.text(2000, 1.02, "Causal Limit", fontsize=9)
        # Conformal Limit (cs2 = 1/3)
        ax.axhline(1.0/3.0, color='gray', linestyle=':', linewidth=1.5)
        ax.text(2000, 0.36, "Conformal", fontsize=9, color='gray')

    plot_boundary(
        x_col='Eps_Central', y_col='CS2_Central',
        x_label=r'Central Density $\varepsilon_c$ [MeV/fm$^3$]', y_label=r'Sound Speed $c_s^2$',
        x_lim=(100, 3000), y_lim=(0.0, 1.1),
        filename='fig_10c_mechanism_Microphysics.pdf',
        overlay_func=overlay_micro
    )

    # ==============================================================
    # PLOT 4: Radius_14 vs Slope_14 (Topology)
    # ==============================================================
    def overlay_topo(ax):
        # Stability Line (Slope = 0)
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
        ax.text(14, 0.2, "Stability Threshold", fontsize=9)

    plot_boundary(
        x_col='Radius_14', y_col='Slope14',
        x_label=r'Radius $R_{1.4}$ [km]', y_label=r'Slope $dR/dM|_{1.4}$',
        x_lim=(9.0, 16.0), y_lim=(-8, 6),
        filename='fig_10d_mechanism_Topology.pdf',
        use_unique_curves=True, 
        overlay_func=overlay_topo
    )

    print("[Success] Saved 4 Decision Manifold plots.")
