
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

def plot_microphysics_3d(df):
    """
    Generates a 3D Scatter plot of Microphysics parameters with KDE wall projections.
    """
    set_paper_style()
    print("\n--- Generating 3D Microphysics Manifold (Intersection Shadows) ---")
    
    # 1. Data Prep
    plot_df = df.dropna(subset=['Eps_Central', 'CS2_Central', 'Slope14']).copy()
    h_data = plot_df[plot_df['Label'] == 0]
    q_data = plot_df[plot_df['Label'] == 1]
    
    # Load Limits from Constants
    eps_lim = CONSTANTS['PLOT_EPS_LIM']
    cs2_lim = CONSTANTS['PLOT_CS2_LIM']
    slope_lim = CONSTANTS['PLOT_SLOPE_LIM']

    # Helper for Density
    def get_density_grid(x, y, x_lim, y_lim, grid_size=60):
        if len(x) > 5000:
            idx = np.random.choice(len(x), 5000, replace=False)
            x_s, y_s = x.iloc[idx], y.iloc[idx]
        else:
            x_s, y_s = x, y
        
        xx = np.linspace(x_lim[0], x_lim[1], grid_size)
        yy = np.linspace(y_lim[0], y_lim[1], grid_size)
        XX, YY = np.meshgrid(xx, yy)
        positions = np.vstack([XX.ravel(), YY.ravel()])
        
        try:
            kernel = gaussian_kde(np.vstack([x_s, y_s]))
            Z = np.reshape(kernel(positions).T, XX.shape)
            return XX, YY, Z / Z.max()
        except:
            return XX, YY, np.zeros_like(XX)

    # ==========================================
    # PART A: MATPLOTLIB (Static PDF)
    # ==========================================
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    n_scatter = 3000
    h_sample = h_data.sample(min(n_scatter, len(h_data)), random_state=42)
    q_sample = q_data.sample(min(n_scatter, len(q_data)), random_state=42)
    
    ax.scatter(h_sample['Eps_Central'], h_sample['CS2_Central'], h_sample['Slope14'], 
               c=COLORS['H_main'], s=5, alpha=0.1, edgecolors='none', label='Hadronic')
    
    ax.scatter(q_sample['Eps_Central'], q_sample['CS2_Central'], q_sample['Slope14'], 
               c=COLORS['Q_main'], s=5, alpha=0.1, edgecolors='none', label='Quark')

    # --- Wall Projections ---
    print("Calculating Wall Intersections (PDF)...")
    
    def plot_mpl_wall(ax, x_h, y_h, x_q, y_q, z_dir, z_offset, x_lim, y_lim):
        XX, YY, Z_h = get_density_grid(x_h, y_h, x_lim, y_lim)
        _, _, Z_q = get_density_grid(x_q, y_q, x_lim, y_lim)
        
        Z_h[Z_h < 0.05] = np.nan
        Z_q[Z_q < 0.05] = np.nan
        
        ax.contourf(XX, YY, Z_h, zdir=z_dir, offset=z_offset, levels=5, colors=[COLORS['H_main']], alpha=0.50)
        ax.contour(XX, YY, Z_h, zdir=z_dir, offset=z_offset, levels=[0.1], colors=[COLORS['H_main']], linewidths=1.5)
        
        ax.contourf(XX, YY, Z_q, zdir=z_dir, offset=z_offset, levels=5, colors=[COLORS['Q_main']], alpha=0.50)
        ax.contour(XX, YY, Z_q, zdir=z_dir, offset=z_offset, levels=[0.1], colors=[COLORS['Q_main']], linewidths=1.5, linestyles='--')

    # Floor (z=Min Slope): Eps vs CS2
    plot_mpl_wall(ax, h_data['Eps_Central'], h_data['CS2_Central'], 
                  q_data['Eps_Central'], q_data['CS2_Central'], 'z', slope_lim[0], eps_lim, cs2_lim)
    
    # Back Wall (y=Max CS2): Eps vs Slope
    plot_mpl_wall(ax, h_data['Eps_Central'], h_data['Slope14'], 
                  q_data['Eps_Central'], q_data['Slope14'], 'y', cs2_lim[1], eps_lim, slope_lim)
    
    # Side Wall (x=Min Eps): CS2 vs Slope
    plot_mpl_wall(ax, h_data['CS2_Central'], h_data['Slope14'], 
                  q_data['CS2_Central'], q_data['Slope14'], 'x', eps_lim[0], cs2_lim, slope_lim)

    # Aesthetics
    ax.set_xlim(eps_lim); ax.set_ylim(cs2_lim); ax.set_zlim(slope_lim)
    ax.set_xlabel(r'$\varepsilon_c$ [MeV/fm$^3$]', labelpad=12)
    ax.set_ylabel(r'$c_s^2$', labelpad=12)
    ax.set_zlabel(r'Slope $dR/dM$', labelpad=12)
    ax.set_title(r"Microphysics Manifold (With Projections)", y=1.02)
    ax.view_init(elev=30, azim=-60)
    
    # Constraints
    ax.plot([eps_lim[0], eps_lim[1]], [1.0, 1.0], [slope_lim[0], slope_lim[0]], 
            color='black', linestyle='--', lw=1.5, zorder=10)
    ax.plot([eps_lim[0], eps_lim[0]], [cs2_lim[0], cs2_lim[1]], [0, 0], 
            color='black', linestyle=':', lw=1.5, zorder=10)

    # Panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['H_main'], label='Hadronic'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['Q_main'], label='Quark')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig("plots/fig_microphysics_3d.pdf")
    plt.close()
    
    # ==========================================
    # PART B: PLOTLY (Interactive HTML)
    # ==========================================
    if PLOTLY_AVAILABLE:
        print("Generating Interactive Microphysics Plot with KDE Surfaces...")
        fig_html = go.Figure()
        
        n_web = 5000
        h_web = h_data.sample(min(n_web, len(h_data)), random_state=42)
        q_web = q_data.sample(min(n_web, len(q_data)), random_state=42)
        
        fig_html.add_trace(go.Scatter3d(
            x=h_web['Eps_Central'], y=h_web['CS2_Central'], z=h_web['Slope14'],
            mode='markers', marker=dict(size=2, color=COLORS['H_main'], opacity=0.3), name='Hadronic'
        ))
        fig_html.add_trace(go.Scatter3d(
            x=q_web['Eps_Central'], y=q_web['CS2_Central'], z=q_web['Slope14'],
            mode='markers', marker=dict(size=2, color=COLORS['Q_main'], opacity=0.3), name='Quark'
        ))
        
        def add_plotly_wall(fig, x_h, y_h, x_q, y_q, wall_type, x_range, y_range):
            XX, YY, Z_h = get_density_grid(x_h, y_h, x_range, y_range, grid_size=50)
            _, _, Z_q = get_density_grid(x_q, y_q, x_range, y_range, grid_size=50)
            
            Z_h[Z_h < 0.05] = np.nan
            Z_q[Z_q < 0.05] = np.nan
            
            if wall_type == 'floor': # Z is constant (min slope)
                z_h = np.full_like(XX, slope_lim[0])
                z_q = np.full_like(XX, slope_lim[0] + 0.2)
                fig.add_trace(go.Surface(x=XX, y=YY, z=z_h, surfacecolor=Z_h, colorscale='Greens', showscale=False, opacity=0.5))
                fig.add_trace(go.Surface(x=XX, y=YY, z=z_q, surfacecolor=Z_q, colorscale='RdPu', showscale=False, opacity=0.5))
                
            elif wall_type == 'back': # Y is constant (max CS2)
                y_h = np.full_like(XX, cs2_lim[1])
                y_q = np.full_like(XX, cs2_lim[1] - 0.02)
                # Inputs were (Eps, Slope) -> (X, Z)
                fig.add_trace(go.Surface(x=XX, y=y_h, z=YY, surfacecolor=Z_h, colorscale='Greens', showscale=False, opacity=0.5))
                fig.add_trace(go.Surface(x=XX, y=y_q, z=YY, surfacecolor=Z_q, colorscale='RdPu', showscale=False, opacity=0.5))
                
            elif wall_type == 'side': # X is constant (min Eps)
                x_h = np.full_like(XX, eps_lim[0])
                x_q = np.full_like(XX, eps_lim[0] + 20)
                # Inputs were (CS2, Slope) -> (Y, Z)
                fig.add_trace(go.Surface(x=x_h, y=XX, z=YY, surfacecolor=Z_h, colorscale='Greens', showscale=False, opacity=0.5))
                fig.add_trace(go.Surface(x=x_q, y=XX, z=YY, surfacecolor=Z_q, colorscale='RdPu', showscale=False, opacity=0.5))

        # Add Surfaces
        # Floor (Eps, CS2)
        add_plotly_wall(fig_html, h_data['Eps_Central'], h_data['CS2_Central'], 
                        q_data['Eps_Central'], q_data['CS2_Central'], 'floor', eps_lim, cs2_lim)
        # Back (Eps, Slope)
        add_plotly_wall(fig_html, h_data['Eps_Central'], h_data['Slope14'], 
                        q_data['Eps_Central'], q_data['Slope14'], 'back', eps_lim, slope_lim)
        # Side (CS2, Slope)
        add_plotly_wall(fig_html, h_data['CS2_Central'], h_data['Slope14'], 
                        q_data['CS2_Central'], q_data['Slope14'], 'side', cs2_lim, slope_lim)

        fig_html.update_layout(
            title="Interactive Model D Space with Density Contours",
            scene=dict(
                xaxis_title='Density',
                yaxis_title='Sound Speed',
                zaxis_title='Slope',
                xaxis=dict(range=eps_lim),
                yaxis=dict(range=cs2_lim),
                zaxis=dict(range=slope_lim),
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        fig_html.write_html("plots/fig_microphysics_3d_interactive.html")
    
    print("[Success] Saved Microphysics 3D Plots.")