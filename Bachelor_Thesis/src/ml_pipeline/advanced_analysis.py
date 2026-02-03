
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.model_selection import learning_curve, GroupKFold

from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS


try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[Info] 'shap' library not found. Skipping SHAP plots.")

def run_advanced_analysis(df, models_dict, X_test, y_test):
    """
    Master function to run ML Rigor tests and Physics Correlation checks.
    """
    set_paper_style()
    print("\n========================================")
    print("   RUNNING ADVANCED ML DIAGNOSTICS")
    print("   (Observational Models Only: Geo, A)")
    print("========================================")
    
    # Define feature requirements for the models being tested
    model_features = {
        'Geo': ['Mass', 'Radius'],
        'A':   ['Mass', 'Radius', 'LogLambda']
    }
    
    # ------------------------------------------------------
    # 1. Learning Curves (Data Efficiency)
    # ------------------------------------------------------
    if 'Geo' in models_dict:
        plot_learning_curve(models_dict['Geo'], df, 'Geo', model_features['Geo'])
    
    if 'A' in models_dict:
        plot_learning_curve(models_dict['A'], df, 'A', model_features['A'])
    
    # ------------------------------------------------------
    # 2. Noise Robustness (Stress Test)
    # ------------------------------------------------------
    if 'A' in models_dict:
        plot_noise_robustness(models_dict['A'], X_test, y_test)

    # ------------------------------------------------------
    # 3. Physics Correlations (KDE Contours)
    # ------------------------------------------------------
    print("\n--- Generating Physics Correlation Plots (KDE Contours) ---")
    
    # Use exact test set rows to match predictions
    df_test_slice = df.loc[X_test.index]
    
    # Parameters to correlate against P(Quark)
    physics_params = [
        ('LogLambda', r'Log$_{10}$ Tidal Deformability $\Lambda$', 'lambda'),
        ('Eps_Central', r'Central Energy Density $\epsilon_c$ [MeV/fm$^3$]', 'epsilon'),
        ('CS2_Central', r'Squared Sound Speed $c_s^2(r=0)$', 'cs2'),
        ('Slope14', r'Slope $dR/dM$ at $1.4 M_\odot$', 'slope')
    ]
    
    for model_name in ['Geo', 'A']:
        if model_name not in models_dict: continue
        model = models_dict[model_name]
        
        # Select specific columns to avoid ValueError during prediction
        cols_needed = model_features[model_name]
        X_input = X_test[cols_needed]
        
        # Get Probabilities
        probs = model.predict_proba(X_input)[:, 1]
        
        for col, label, tag in physics_params:
            if col not in df_test_slice.columns or df_test_slice[col].isna().any():
                continue
            
            plot_probability_kde(
                x_data=df_test_slice[col],
                y_probs=probs,
                x_label=label,
                model_name=model_name,
                tag=tag
            )

    # ------------------------------------------------------
    # 4. SHAP Analysis (Model A Only)
    # ------------------------------------------------------
    if SHAP_AVAILABLE and 'A' in models_dict:
        plot_shap_analysis(models_dict['A'], X_test, 'A')


# =========================================================
# PLOTTING FUNCTIONS
# =========================================================

def plot_probability_kde(x_data, y_probs, x_label, model_name, tag):
    """
    Generates a KDE (Kernel Density Estimation) Contour plot.
    Now uses a White-to-Color gradient ('BuPu') to avoid ugly black blocks.
    """
    plt.figure(figsize=(8, 6))
    
    # 1. SUBSAMPLING
    MAX_SAMPLES = 50000
    data = pd.DataFrame({'x': x_data, 'y': y_probs}).dropna()
    if len(data) > MAX_SAMPLES:
        data = data.sample(n=MAX_SAMPLES, random_state=42)
    
    x = data['x'].values
    y = data['y'].values

    # 2. CALCULATE KDE
    try:
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        
        # Sort points by density
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        
        # 3. PLOT FILLED CONTOURS
        xmin, xmax = x.min(), x.max()
        ymin, ymax = -0.05, 1.05 
        
        Xgrid, Ygrid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([Xgrid.ravel(), Ygrid.ravel()])
        
        kernel = gaussian_kde(xy)
        Zgrid = np.reshape(kernel(positions).T, Xgrid.shape)
        
        # Mask very low density areas to force white background
        Zgrid[Zgrid < 0.01 * Zgrid.max()] = np.nan
        
        # Use 'BuPu' (Blue-Purple) or 'OrRd' (Orange-Red) which start light/white
        plt.contourf(Xgrid, Ygrid, Zgrid, levels=20, cmap='BuPu')
        
    except Exception:
        print(f"   [Warning] KDE failed for {tag}. Fallback to scatter.")
        plt.scatter(x, y, c='gray', s=1, alpha=0.1)

    # 4. DECORATIONS
    cb = plt.colorbar()
    cb.set_label('Probability Density')
    
    plt.xlabel(x_label)
    plt.ylabel(r"Predicted Probability $P(\text{Quark})$")
    plt.title(f"Model {model_name}: Correlation with {tag.capitalize()}")
    
    # Reference Lines (Darker for visibility on light background)
    plt.axhline(0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='Decision Boundary')
    
    if tag == 'cs2':
        plt.axvline(1.0/3.0, color='gray', linestyle=':', linewidth=2, label=r'Conformal ($1/3$)')
    if tag == 'slope':
        plt.axvline(0.0, color='gray', linestyle=':', linewidth=2, label='Zero Slope')
    
    # =========================================================
    # 5. PHYSICS LIMITS (Centralized)
    # =========================================================
    if tag == 'slope':
        plt.xlim(CONSTANTS['PLOT_SLOPE_LIM'])
    elif tag == 'epsilon':
        plt.xlim(CONSTANTS['PLOT_EPS_LIM'])
    elif tag == 'lambda':
        plt.xlim(CONSTANTS['PLOT_L_LIM'])
    elif tag == 'cs2':
        plt.xlim(CONSTANTS['PLOT_CS2_LIM'])

    plt.ylim(-0.05, 1.05) 
    
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    
    filename = f"plots/fig_corr_{model_name}_{tag}.pdf"
    plt.savefig(filename)
    plt.close()
    print(f"   Saved: {filename}")


def plot_learning_curve(model, df, model_name, features):
    # (Same as before - standard plotting code)
    print(f"   Generating Learning Curve for Model {model_name}...")
    X = df[features]
    y = df['Label']
    groups = df['Curve_ID']
    cv_splitter = GroupKFold(n_splits=5)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv_splitter, groups=groups, n_jobs=1,            
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, 'o-', color='gray', label="Training Score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='gray')
    color = COLORS['Q_main'] 
    ax.plot(train_sizes, test_mean, 'o-', color=color, linewidth=2, label="Cross-Validation Score")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color=color)
    ax.set_title(f"Learning Curve: Model {model_name}")
    ax.set_xlabel("Training Set Size (Samples)")
    ax.set_ylabel("Accuracy")
    if test_mean[-1] > 0.98: ax.set_ylim(0.95, 1.001)
    else: ax.set_ylim(0.80, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2)
    gap = train_mean[-1] - test_mean[-1]
    msg = "Converged" if gap < 0.05 else "Needs Data"
    ax.text(0.05, 0.5, msg, transform=ax.transAxes, fontsize=12, fontweight='bold', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    plt.savefig(f"plots/fig_ml_learning_curve_{model_name}.pdf")
    plt.close()

def plot_noise_robustness(model, X_test, y_test):
    # (Same as before - standard plotting code)
    print("   Generating Noise Robustness Stress Test (Model A)...")
    required_cols = ['Mass', 'Radius', 'LogLambda']
    X_base = X_test[required_cols].copy()
    noise_levels = np.linspace(0.0, 2.0, 20) 
    accuracies = []
    for sigma in noise_levels:
        X_noisy = X_base.copy()
        noise = np.random.normal(0, sigma, size=len(X_noisy))
        X_noisy['Radius'] += noise
        acc = model.score(X_noisy[required_cols], y_test)
        accuracies.append(acc)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(noise_levels, accuracies, 'D-', color='#DDAA33', linewidth=2, markersize=6)
    ax.set_title(r"Robustness to Observational Noise (Radius)")
    ax.set_xlabel(r"Injected Noise $\sigma_{Radius}$ [km]")
    ax.set_ylabel(r"Model Accuracy")
    ax.axvline(0.5, color='gray', linestyle='--', label='Typical NICER Error')
    ax.axhline(0.90, color='red', linestyle=':', label='90% Reliability')
    ax.legend()
    plt.savefig("plots/fig_ml_noise_robustness.pdf")
    plt.close()

def plot_shap_analysis(model, X_test, model_name):
    
    print(f"   Generating SHAP Beeswarm for Model {model_name}...")
    if hasattr(model, 'calibrated_classifiers_'):
        explainer_model = model.calibrated_classifiers_[0].estimator
    else:
        explainer_model = model
    cols_A = ['Mass', 'Radius', 'LogLambda']
    if len(X_test) > 500: X_shap = X_test[cols_A].sample(500, random_state=42)
    else: X_shap = X_test[cols_A]
    name_map = {'Mass': r'$M$', 'Radius': r'$R$', 'LogLambda': r'$\log\Lambda$'}
    X_display = X_shap.rename(columns=name_map)
    explainer = shap.TreeExplainer(explainer_model)
    shap_values = explainer.shap_values(X_shap, check_additivity=False)
    if isinstance(shap_values, list): vals_to_plot = shap_values[1]
    else: vals_to_plot = shap_values
    plt.figure() 
    shap.summary_plot(vals_to_plot, X_display, show=False, plot_type="dot", cmap='coolwarm')
    plt.title(f"SHAP Feature Importance (Model {model_name})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"plots/fig_ml_shap_beeswarm_{model_name}.pdf", bbox_inches='tight')
    plt.close()