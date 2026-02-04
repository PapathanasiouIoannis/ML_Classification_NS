
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS

def plot_stability_window(df):
    """
    Generates the QCD Stability Window plot (B vs Delta).
    """
    set_paper_style()
    print("\n--- Generating Stability Window (QCD Check) ---")
    
    # 1. Filter Data
    q_data = df[df['Label'] == 1].drop_duplicates(subset=['Curve_ID'])
    
    if 'Bag_B' not in q_data.columns: 
        print("Missing Bag_B columns. Skipping.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # ==========================================
    # 2. THE SCATTER DATA
    # ==========================================
    # Scatter points colored by m_s
    sc = ax.scatter(q_data['Gap_Delta'], q_data['Bag_B'], 
                    c=q_data['Mass_Strange'], cmap='viridis', 
                    s=20, alpha=0.7, edgecolors='none', label='Generated Models')
    
    # ==========================================
    # 3. DYNAMIC STABILITY BOUNDARIES (The Band)
    # ==========================================
    hc = CONSTANTS['HC']   
    m_n = CONSTANTS['M_N']
    
    # Retrieve range from constants
    ms_min, ms_max = CONSTANTS['Q_MS_RANGE']
    
    # Grid for Delta
    delta_vals = np.linspace(CONSTANTS['Q_DELTA_RANGE'][0], 
                             CONSTANTS['Q_DELTA_RANGE'][1] + 50, 200)
    mu_limit = m_n / 3.0
    
    def calculate_b_max(delta_arr, ms_val):
        """Calculates B_max vector for a specific ms."""
        # B_max = [3/(4pi^2)*mu^4 + 3/pi^2 * (Delta^2 - ms^2/4)*mu^2] / hc^3
        term1 = (3.0 / (4.0 * np.pi**2)) * (mu_limit**4)
        eff_gap_sq = delta_arr**2 - (ms_val**2 / 4.0)
        term2 = (3.0 / np.pi**2) * eff_gap_sq * (mu_limit**2)
        return (term1 + term2) / (hc**3)

    # Calculate boundaries

    b_max_ms_min = calculate_b_max(delta_vals, ms_min) # ms = 80
    b_max_ms_max = calculate_b_max(delta_vals, ms_max) # ms = 120
    
    # Plot the Band
    ax.fill_between(delta_vals, b_max_ms_min, b_max_ms_max, 
                    color='black', alpha=0.15, label=f'Stability Uncertainty ($m_s={int(ms_min)}$-${int(ms_max)}$)')
    
    # Plot the edges
    ax.plot(delta_vals, b_max_ms_min, 'k--', linewidth=1.0)
    ax.plot(delta_vals, b_max_ms_max, 'k-', linewidth=1.5, label=f'Max Stability Limit')

    # ==========================================
    # 4. IRON STABILITY (Lower Bound)
    # ==========================================
    b_min = CONSTANTS['Q_B_MIN']
    ax.axhline(b_min, color='red', linewidth=2.0, linestyle='--', 
               label=r'Iron Stability ($^{56}$Fe)')
    
    # ==========================================
    # 5. SHADING (Forbidden Regions)
    # ==========================================
    # Unstable to Neutrons (Above the band)
    # We shade above the highest possible stable B
    ax.fill_between(delta_vals, b_max_ms_min, 450, color='gray', alpha=0.1)
    
    # Unstable to 2-flavor (Below Iron line)
    ax.fill_between(delta_vals, 0, b_min, color='gray', alpha=0.1)
    
    # ==========================================
    # 6. FORMATTING
    # ==========================================
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"Strange Quark Mass $m_s$ [MeV]")
    
    ax.set_xlim(CONSTANTS['Q_DELTA_RANGE'][0], CONSTANTS['Q_DELTA_RANGE'][1])
    ax.set_ylim(50, 350)
    
    ax.set_xlabel(r"Gap Energy $\Delta$ [MeV]")
    ax.set_ylabel(r"Vacuum Pressure $B$ [MeV/fm$^3$]")
    ax.set_title(r"QCD Stability Window (Dynamic $m_s$)")
    
    # Annotations
    ax.text(60, 250, r"Unstable (Decays to Neutrons)", 
            color='black', alpha=0.6, fontsize=10)
    ax.text(180, 25, r"Unstable (2-Flavor Stable)", 
            color='red', alpha=0.6, fontsize=10)

    ax.legend(loc='best', frameon=True)
    
    plt.savefig("plots/fig_stability_triangle.pdf")
    plt.close()
    print("[Success] Saved Stability Triangle.")
