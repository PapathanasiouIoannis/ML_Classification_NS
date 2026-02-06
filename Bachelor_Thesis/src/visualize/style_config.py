
import matplotlib.pyplot as plt


COLORS = {
    'H_main': 'green',     
    'Q_main': 'magenta',   
    'H_fade': '#228B22',   # Hadronic
    'Q_fade': '#BA55D3',   # Quark
    'Constraint': 'black', # Observational limits 
    'Guide': 'gray'        # Guide lines / Grids
}

def set_paper_style():
    """
    Configures Matplotlib for  plots.
    
    Settings:
    - Font: Sans-serif body, Computer Modern math ($...$).
    - DPI: 300 for high-resolution output.
    - Ticks: Inward facing, present on all sides.
    - Grid: Subtle dotted lines for readability.
    """
    plt.rcParams.update({
        # --- LAYOUT & RESOLUTION ---
        "figure.figsize": (10, 7),
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "font.family": "sans-serif",      
        "mathtext.fontset": "cm",         
        
        # --- SIZES ---
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        
        # --- TICKS ---
        "xtick.direction": "in",          
        "ytick.direction": "in",
        "xtick.top": True,                
        "ytick.right": True,              
        "xtick.minor.visible": True,      
        "ytick.minor.visible": True,
        
        # --- LINES & GEOMETRY ---
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.linewidth": 1.0,
        
        # --- GRID ---
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        
        # --- LEGEND ---
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "gray",
        "legend.fancybox": False,         
        "legend.loc": "best"
    })
