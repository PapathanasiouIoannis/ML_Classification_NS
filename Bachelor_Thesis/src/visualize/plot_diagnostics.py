
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from src.visualize.style_config import set_paper_style, COLORS

def plot_diagnostics(models_dict, X_test_all, y_test):
    """
    Generates ML diagnostic plots: ROC, Calibration, and Radius Violin plots.
    
    Parameters:
    - models_dict: Dictionary of trained classifiers.
    - X_test_all: Test features DataFrame.
    - y_test: Test labels Series.
    """
    set_paper_style()
    print("\n--- Generating ML Diagnostic Plots ---")
    
    # Feature definitions for each model
    feature_sets = {
        'A': ['Mass', 'Radius', 'LogLambda'],
        'B': ['Mass', 'Radius', 'LogLambda', 'Eps_Central'],
        'C': ['Mass', 'Radius', 'LogLambda', 'Eps_Central', 'CS2_Central'],
        'D': ['Mass', 'Radius', 'LogLambda', 'Eps_Central', 'CS2_Central', 'Slope14'],
        'Geo': ['Mass', 'Radius']
    }
    
    # Consistent color mapping for models
    model_colors = {
        'A': '#1f77b4',   # Blue
        'B': '#17becf',   # Cyan
        'C': '#ff7f0e',   # Orange
        'D': '#2ca02c',   # Green
        'Geo': '#7f7f7f'  # Gray
    }

    # ==========================================
    # 1. ROC CURVE COMPARISON
    # ==========================================
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot([0, 1], [0, 1], color='black', linestyle=':', alpha=0.5)
    
    # iterate in logical complexity order
    for name in ['Geo', 'A', 'B', 'C', 'D']:
        if name not in models_dict: continue
        
        model = models_dict[name]
        features = feature_sets[name]
        
        y_probs = model.predict_proba(X_test_all[features])[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.4f})", 
                    color=model_colors[name])

    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve Comparison')
    ax_roc.legend(loc="lower right")
    
    plt.savefig("plots/fig_6_roc_combined.pdf")
    plt.close()

    # ==========================================
    # 2. CALIBRATION CURVE (RELIABILITY)
    # ==========================================
    fig_cal, ax_cal = plt.subplots(figsize=(8, 6))
    ax_cal.plot([0, 1], [0, 1], linestyle=':', color='black', label='Ideal')
    
    for name in ['Geo', 'A', 'B', 'C', 'D']:
        if name not in models_dict: continue
        
        model = models_dict[name]
        y_probs = model.predict_proba(X_test_all[feature_sets[name]])[:, 1]
        
        # strategy='quantile' handles histograms with variable bin widths,
        # which is robust for highly accurate models (probs clustered near 0/1).
        prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10, strategy='quantile')
        
        ax_cal.plot(prob_pred, prob_true, marker='s', markersize=4, 
                    label=f'{name}', color=model_colors[name])
        
    ax_cal.set_xlabel('Mean Predicted Probability')
    ax_cal.set_ylabel('True Positive Fraction')
    ax_cal.set_title('Reliability Diagram (Quantile Binned)')
    ax_cal.legend()
    
    plt.savefig("plots/fig_7b_calibration.pdf")
    plt.close()

    # ==========================================
    # 3. SPLIT VIOLIN PLOT (Radius Distribution)
    # ==========================================
    # Visualizes how the Radius distributions of Hadronic vs Quark stars
    # diverge across different mass regimes.
    
    plot_df = X_test_all.copy()
    plot_df['Label'] = y_test 
    
    # Define Mass Bins
    bins = [0.0, 1.1, 1.7, 4.0]
    labels = [r'Low ($<1.1$)', r'Canonical ($1.1-1.7$)', r'High ($>1.7$)']
    plot_df['Mass_Bin'] = pd.cut(plot_df['Mass'], bins=bins, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=plot_df, x='Mass_Bin', y='Radius', hue='Label',
        split=True, inner=None, 
        palette={0: COLORS['H_main'], 1: COLORS['Q_main']},
        linewidth=1.0, alpha=0.8, ax=ax
    )
    
    ax.set_title('Radius Distribution by Mass Regime')
    ax.set_xlabel(r'Mass Regime [$M_{\odot}$]')
    ax.set_ylabel(r'Radius [km]')
    
    # Custom Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=COLORS['H_main'], lw=4, label='Hadronic'),
               Line2D([0],[0], color=COLORS['Q_main'], lw=4, label='Quark')]
    ax.legend(handles=handles, loc='best')
    
    plt.savefig("plots/fig_8_violin_radius.pdf")
    plt.close()
    
    print("[Success] Saved Diagnostics (ROC, Calibration, Violins).")