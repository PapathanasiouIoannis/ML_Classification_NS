
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV

def train_model(df):
    """
    Trains the hierarchy of Calibrated Random Forest Classifiers.
    
    Parameters:
    - df: The master DataFrame containing physics data.
    
    Returns:
    - models: Dictionary of trained models {'A': model, 'Geo': model, ...}
    - X_test_all: DataFrame of features for the held-out test set.
    - y_test: Series of labels for the held-out test set.
    """
    print("\n--- Training Hierarchy of Random Forest Models ---")
    
    # ==========================================
    # 1. PRE-PROCESSING & SAFETY CHECKS
    # ==========================================
    # Feature Engineering: Log-Transform Lambda if not present
    if 'LogLambda' not in df.columns:
        df['LogLambda'] = np.log10(df['Lambda'])
    
    # Common Denominator Filter:
    # Model D relies on 'Slope14' (derivative at 1.4 M_sun). This feature is undefined
    # for stars that collapse before reaching 1.4 M_sun. To compare Model A and Model D
    # fairly, both must be trained/tested on the exact same subset of stars.
    initial_len = len(df)
    
    # Drop rows where critical physics features are NaN
    df_clean = df.dropna(subset=['Slope14', 'Eps_Central', 'CS2_Central', 'Radius', 'Mass']).copy()
    dropped_len = initial_len - len(df_clean)
    
    if dropped_len > 0:
        print(f"[Info] Dropped {dropped_len} low-mass stars (NaN features) to ensure fair model comparison.")
        print(f"[Info] Training set size: {len(df_clean)} curves.")

    y = df_clean['Label']
    groups = df_clean['Curve_ID']

    # ==========================================
    # 2. CROSS-VALIDATION SPLIT 
    # ==========================================
    # Split by Curve_ID to ensure the model generalizes to new EoS physics,
    # rather than memorizing points along a known curve.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(df_clean, y, groups=groups))
    
    # ==========================================
    # 3. HYPERPARAMETER CONFIGURATION
    # ==========================================
    
    # Base Configuration (Models A, B, C, D)
    # max_features='sqrt' is used to force tree decorrelation, ensuring that 
    # microphysical parameters are used even when correlated with observables.
    rf_params_base = {
        'n_estimators': 1000,        # High estimator count for stability
        'max_depth': 15,             # Sufficient depth for complex M-R topology
        'min_samples_leaf': 10,      # Regularization to prevent overfitting single curves
        'max_features': 'sqrt',      # Force feature diversity
        'bootstrap': True,           
        'class_weight': 'balanced',  
        'n_jobs': -1,
        'random_state': 42
    }

    # Strict Configuration (Model Geo)
    # Limited to small depth as it only uses Mass and Radius.
    rf_params_geo = rf_params_base.copy()
    rf_params_geo.update({
        'max_depth': 10,             
        'min_samples_leaf': 50       
    })
    
    # Define Feature Sets
    cols_A = ['Mass', 'Radius', 'LogLambda']
    cols_B = ['Mass', 'Radius', 'LogLambda', 'Eps_Central']
    cols_C = ['Mass', 'Radius', 'LogLambda', 'Eps_Central', 'CS2_Central']
    cols_D = ['Mass', 'Radius', 'LogLambda', 'Eps_Central', 'CS2_Central', 'Slope14']
    cols_geo = ['Mass', 'Radius']

    # ==========================================
    # 4. TRAINING LOOP (WITH CALIBRATION)
    # ==========================================
    models = {}
    
    train_setup = [
        ('A', cols_A, rf_params_base),
        ('B', cols_B, rf_params_base),
        ('C', cols_C, rf_params_base),
        ('D', cols_D, rf_params_base),
        ('Geo', cols_geo, rf_params_geo)
    ]

    for name, cols, params in train_setup:
        print(f"Training Model {name} (Calibrated)...")
        
        # 1. Instantiate Base Random Forest
        base_rf = RandomForestClassifier(**params)
        
        # 2. Wrap in Isotonic Calibration
        # Essential for interpreting outputs as true probabilities (Reliability Curves)
        cal_rf = CalibratedClassifierCV(base_rf, method='isotonic', cv=3)
        
        # 3. Fit the Calibrated Stack
        cal_rf.fit(df_clean.iloc[train_idx][cols], y.iloc[train_idx])
        
        models[name] = cal_rf

    # ==========================================
    # 5. OVERFITTING DIAGNOSTICS
    # ==========================================
    print("\n--- Overfitting Diagnostics (Train vs Test) ---")
    print(f"{'Model':<5} | {'Train Acc':<10} | {'Test Acc':<10} | {'Gap':<10} | {'Status'}")
    print("-" * 60)

    for name, cols, _ in train_setup:
        model = models[name]
        
        # Calculate Scores
        train_acc = model.score(df_clean.iloc[train_idx][cols], y.iloc[train_idx])
        test_acc = model.score(df_clean.iloc[test_idx][cols], y.iloc[test_idx])
        gap = train_acc - test_acc
        
        status = "OK"
        if gap > 0.05: status = "OvrFit" # >5% gap implies variance error
        if train_acc < 0.70: status = "UndFit" # <70% implies bias error
        
        print(f"{name:<5} | {train_acc:.4f}     | {test_acc:.4f}     | {gap:.4f}     | [{status}]")

    # ==========================================
    # 6. COMPATIBILITY PATCHING
    # ==========================================
    # Sklearn's CalibratedClassifierCV does not expose feature_importances_ directly.
    # We aggregate them from the underlying base estimators to allow SHAP/Importance plotting.
    
    if 'D' in models:
        rf_d_cal = models['D']
        
        # A. Patch Feature Importances (Average across calibration folds)
        importances = np.mean([
            clf.estimator.feature_importances_ 
            for clf in rf_d_cal.calibrated_classifiers_
        ], axis=0)
        
        std_importances = np.std([
            clf.estimator.feature_importances_ 
            for clf in rf_d_cal.calibrated_classifiers_
        ], axis=0)
        
        # Attach attributes to the calibrated object for downstream use
        rf_d_cal.feature_importances_ = importances
        rf_d_cal.feature_importances_std_ = std_importances
        
        # B. Patch Estimators for SHAP
        rf_d_cal.base_model_for_shap = rf_d_cal.calibrated_classifiers_[0].estimator

    # ==========================================
    # 7. FINAL PACKAGING
    # ==========================================
    # Return features for the test set to allow independent evaluation in main.py
    all_needed_cols = list(set(cols_D + cols_geo)) 
    X_test_all = df_clean.iloc[test_idx][all_needed_cols]
    y_test = y.iloc[test_idx]

    return models, X_test_all, y_test