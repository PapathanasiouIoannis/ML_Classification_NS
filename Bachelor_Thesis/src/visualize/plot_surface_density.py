
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.visualize.style_config import set_paper_style, COLORS

def plot_surface_density(df):
    """
    Generates the Surface Density distribution plot, highlighting the 
    'Forbidden Gap' between Hadronic and Quark matter surfaces.
    """
    set_paper_style()
    print("\n--- Generating Surface Density Proof (Singularity vs Distribution) ---")
    
    # 1. Data Prep
    if 'Eps_Surface' not in df.columns:
        print("[Error] 'Eps_Surface' column missing. Skipping.")
        return
        
    q_vals = df[df['Label']==1]['Eps_Surface']
    

    if len(q_vals) < 10:
        print("[Warn] Not enough Quark samples for density plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    

    #  plot the Quark distribution using KDE
    sns.kdeplot(x=q_vals, ax=ax, 
                fill=True, color=COLORS['Q_main'], alpha=0.2, 
                linewidth=2, label='Quark Surface Density')
    

    # Hadronic stars effectively have 0 density at the surface compared to nuclear scales.
    #  represent this with a vertical line.
    ax.axvline(0, color=COLORS['H_main'], linewidth=3, linestyle='-', 
               label='Hadronic Boundary ($\epsilon \\approx 0$)')
    

    gap_limit = q_vals.min()
    
    ax.axvspan(0, gap_limit, color='gray', alpha=0.1, hatch='///')
    ax.text(gap_limit/2, ax.get_ylim()[1]*0.5, "Forbidden\nRegion", 
            ha='center', va='center', color='gray', fontsize=10, fontweight='bold')


    # Hadronic Annotation
    ax.annotate(r"Iron Crust ($\epsilon \approx 0$)", 
                xy=(0, ax.get_ylim()[1]*0.8), xytext=(150, ax.get_ylim()[1]*0.8),
                arrowprops=dict(facecolor=COLORS['H_main'], arrowstyle='->', lw=1.5),
                color=COLORS['H_main'], fontweight='bold')
    
    # Quark Annotation
    ax.annotate(r"Self-Bound Surface ($\epsilon \approx 4B$)", 
                xy=(gap_limit, ax.get_ylim()[1]*0.2), 
                xytext=(gap_limit+200, ax.get_ylim()[1]*0.3),
                arrowprops=dict(facecolor=COLORS['Q_main'], arrowstyle='->', lw=1.5),
                color=COLORS['Q_main'], fontweight='bold')


    ax.set_title(r"Surface Density Gap: Gravity-Bound vs Self-Bound")
    ax.set_xlabel(r"Surface Energy Density $\varepsilon_{surf}$ [MeV/fm$^3$]")
    ax.set_ylabel(r"Probability Density")
 
    ax.set_xlim(-50, 1000)
    
    ax.legend(loc='upper right', frameon=True)
    
    plt.savefig("plots/fig_surface_density.pdf")
    plt.close()
    print("[Success] Saved Surface Density Plot.")
