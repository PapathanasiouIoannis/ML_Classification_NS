
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from src.visualize.style_config import set_paper_style, COLORS
from src.const import CONSTANTS

# Optional: Interactive plotting
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def plot_3d_separation(df):
    """
    Generates a 3D Manifold Plot with projected density contours (Shadows).
    """
    set_paper_style()
    print("\n--- Generating 3D Manifold Plot (Intersection Shadows) ---")
    
    # 1. Data Prep
    if 'LogLambda' not in df.columns:
        df['LogLambda'] = np.log10(df['Lambda'])
        
    h_data = df[df['Label'] == 0]
    q_data = df[df['Label'] == 1]
    
    # Load Limits from Constants
    r_lim = CONSTANTS['PLOT_R_LIM']
    m_lim = CONSTANTS['PLOT_M_LIM']
    l_lim = CONSTANTS['PLOT_L_LIM']

    # Helper for Density Calculation
    def get_density_grid(x, y, x_lim, y_lim, grid_size=60):
        """Calculates 2D KDE on a grid."""
        # Subsample for speed
        if len(x) > 5000:
            idx = np.random.choice(len(x), 5000, replace=False)
            x_s, y_s = x.iloc[idx], y.iloc[idx]
        else:
            x_s, y_s = x, y
            
        # Create grid
        xx = np.linspace(x_lim[0], x_lim[1], grid_size)
        yy = np.linspace(y_lim[0], y_lim[1], grid_size)
        XX, YY = np.meshgrid(xx, yy)
        positions = np.vstack([XX.ravel(), YY.ravel()])
        
        # KDE
        try:
            kernel = gaussian_kde(np.vstack([x_s, y_s]))
            Z = np.reshape(kernel(positions).T, XX.shape)
            # Normalize
            Z = Z / Z.max()
            return XX, YY, Z
        except:
            return XX, YY, np.zeros_like(XX)

    # ==========================================
    # PART A: MATPLOTLIB (Static PDF)
    # ==========================================
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- 1. The Central Volume (Scatter) ---
    n_scatter = 3000
    h_sample = h_data.sample(min(n_scatter, len(h_data)), random_state=42)
    q_sample = q_data.sample(min(n_scatter, len(q_data)), random_state=42)
    
    ax.scatter(h_sample['Radius'], h_sample['Mass'], h_sample['LogLambda'], 
               c=COLORS['H_main'], s=5, alpha=0.1, edgecolors='none', label='Hadronic')
    
    ax.scatter(q_sample['Radius'], q_sample['Mass'], q_sample['LogLambda'], 
               c=COLORS['Q_main'], s=5, alpha=0.1, edgecolors='none', label='Quark')

    # --- 2. The Wall Projections (Matplotlib) ---
    print("Calculating Wall Intersections (PDF)...")
    
    def plot_mpl_wall(ax, x_h, y_h, x_q, y_q, z_dir, z_offset, x_lim, y_lim):
        XX, YY, Z_h = get_density_grid(x_h, y_h, x_lim, y_lim)
        _, _, Z_q = get_density_grid(x_q, y_q, x_lim, y_lim)
        
        # Threshold to remove background noise
        Z_h[Z_h < 0.05] = np.nan
        Z_q[Z_q < 0.05] = np.nan
        
        # Plot Hadronic
        ax.contourf(XX, YY, Z_h, zdir=z_dir, offset=z_offset, levels=5, colors=[COLORS['H_main']], alpha=0.50)
        ax.contour(XX, YY, Z_h, zdir=z_dir, offset=z_offset, levels=[0.1], colors=[COLORS['H_main']], linewidths=1.5)
        
        # Plot Quark
        ax.contourf(XX, YY, Z_q, zdir=z_dir, offset=z_offset, levels=5, colors=[COLORS['Q_main']], alpha=0.50)
        ax.contour(XX, YY, Z_q, zdir=z_dir, offset=z_offset, levels=[0.1], colors=[COLORS['Q_main']], linewidths=1.5, linestyles='--')

    # Floor (z=0): Radius vs Mass
    plot_mpl_wall(ax, h_data['Radius'], h_data['Mass'], 
                  q_data['Radius'], q_data['Mass'], 'z', l_lim[0], r_lim, m_lim)
    
    # Back Wall (y=Max): Radius vs Tidal
    plot_mpl_wall(ax, h_data['Radius'], h_data['LogLambda'], 
                  q_data['Radius'], q_data['LogLambda'], 'y', m_lim[1], r_lim, l_lim)
    
    # Side Wall (x=Min): Mass vs Tidal
    plot_mpl_wall(ax, h_data['Mass'], h_data['LogLambda'], 
                  q_data['Mass'], q_data['LogLambda'], 'x', r_lim[0], m_lim, l_lim)

    # Aesthetics
    ax.set_xlim(r_lim); ax.set_ylim(m_lim); ax.set_zlim(l_lim)
    ax.set_xlabel(r'Radius $R$ [km]', labelpad=12)
    ax.set_ylabel(r'Mass $M$ [$M_{\odot}$]', labelpad=12)
    ax.set_zlabel(r'Log Tidal $\log_{10}\Lambda$', labelpad=12)
    ax.set_title(r"Topological Phase Space (With Projections)", y=1.02)
    ax.view_init(elev=30, azim=135)
    
    # Transparent Panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['H_main'], label='Hadronic'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['Q_main'], label='Quark')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig("plots/fig_12_3d_manifold.pdf")
    plt.close()
    
    # ==========================================
    # PART B: PLOTLY (Interactive HTML)
    # ==========================================
    if PLOTLY_AVAILABLE:
        print("Generating Interactive HTML with KDE Surfaces...")
        fig_html = go.Figure()
        
        # 1. Main Volume Scatter
        n_web = 5000
        h_web = h_data.sample(min(n_web, len(h_data)), random_state=42)
        q_web = q_data.sample(min(n_web, len(q_data)), random_state=42)
        
        fig_html.add_trace(go.Scatter3d(
            x=h_web['Radius'], y=h_web['Mass'], z=h_web['LogLambda'],
            mode='markers', marker=dict(size=2, color=COLORS['H_main'], opacity=0.3), name='Hadronic'
        ))
        fig_html.add_trace(go.Scatter3d(
            x=q_web['Radius'], y=q_web['Mass'], z=q_web['LogLambda'],
            mode='markers', marker=dict(size=2, color=COLORS['Q_main'], opacity=0.3), name='Quark'
        ))
        
        # 2. Add KDE Surfaces on Walls
        def add_plotly_wall(fig, x_h, y_h, x_q, y_q, wall_type, x_range, y_range):
            """
            Adds KDE heatmap surfaces to the 3D plot walls.
            wall_type: 'floor' (z=min), 'back' (y=max), 'side' (x=min)
            """
            XX, YY, Z_h = get_density_grid(x_h, y_h, x_range, y_range, grid_size=50)
            _, _, Z_q = get_density_grid(x_q, y_q, x_range, y_range, grid_size=50)
            
            # Mask low density for HTML transparency
            Z_h[Z_h < 0.05] = np.nan
            Z_q[Z_q < 0.05] = np.nan
            
            # Determine 3D coordinates based on wall type
            if wall_type == 'floor':
                # Floor: Z is constant (l_lim[0])
                # We apply a tiny offset to separate H and Q layers (Z-fighting fix)
                z_h = np.full_like(XX, l_lim[0])
                z_q = np.full_like(XX, l_lim[0] + 0.02) 
                
                fig.add_trace(go.Surface(x=XX, y=YY, z=z_h, surfacecolor=Z_h, 
                                         colorscale='Greens', showscale=False, opacity=0.5))
                fig.add_trace(go.Surface(x=XX, y=YY, z=z_q, surfacecolor=Z_q, 
                                         colorscale='RdPu', showscale=False, opacity=0.5))
                
            elif wall_type == 'back':
                # Back Wall: Y is constant (m_lim[1])
                y_h = np.full_like(XX, m_lim[1])
                y_q = np.full_like(XX, m_lim[1] - 0.02)
                
                # Note: For back wall, inputs were (Radius, Lambda) -> (X, Z)
                # So XX maps to X (Radius), YY maps to Z (Lambda)
                fig.add_trace(go.Surface(x=XX, y=y_h, z=YY, surfacecolor=Z_h, 
                                         colorscale='Greens', showscale=False, opacity=0.5))
                fig.add_trace(go.Surface(x=XX, y=y_q, z=YY, surfacecolor=Z_q, 
                                         colorscale='RdPu', showscale=False, opacity=0.5))
                
            elif wall_type == 'side':
                # Side Wall: X is constant (r_lim[0])
                x_h = np.full_like(XX, r_lim[0])
                x_q = np.full_like(XX, r_lim[0] + 0.05)
                
                # Inputs were (Mass, Lambda) -> (Y, Z)
                # XX maps to Y (Mass), YY maps to Z (Lambda)
                fig.add_trace(go.Surface(x=x_h, y=XX, z=YY, surfacecolor=Z_h, 
                                         colorscale='Greens', showscale=False, opacity=0.5))
                fig.add_trace(go.Surface(x=x_q, y=XX, z=YY, surfacecolor=Z_q, 
                                         colorscale='RdPu', showscale=False, opacity=0.5))

        # Add Surfaces
        # Floor (Radius, Mass)
        add_plotly_wall(fig_html, h_data['Radius'], h_data['Mass'], 
                        q_data['Radius'], q_data['Mass'], 'floor', r_lim, m_lim)
        
        # Back (Radius, Lambda) -> Maps to (X, Z) on wall
        add_plotly_wall(fig_html, h_data['Radius'], h_data['LogLambda'], 
                        q_data['Radius'], q_data['LogLambda'], 'back', r_lim, l_lim)
        
        # Side (Mass, Lambda) -> Maps to (Y, Z) on wall
        add_plotly_wall(fig_html, h_data['Mass'], h_data['LogLambda'], 
                        q_data['Mass'], q_data['LogLambda'], 'side', m_lim, l_lim)

        fig_html.update_layout(
            title="Interactive Manifold with Density Contours",
            scene=dict(
                xaxis_title='Radius (km)',
                yaxis_title='Mass (M_sun)',
                zaxis_title='Log Tidal',
                xaxis=dict(range=r_lim),
                yaxis=dict(range=m_lim),
                zaxis=dict(range=l_lim),
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        fig_html.write_html("plots/fig_12_3d_interactive.html")
    
    print("[Success] Saved 3D PDF and HTML.")