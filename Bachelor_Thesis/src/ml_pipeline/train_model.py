

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV

def train_model(df):
    print("\n--- Training Hierarchy of Random Forest Models ---")
    
    # 1. Feature Engineering & Cleaning
    if 'LogLambda' not in df.columns:
        df['LogLambda'] = np.log10(df['Lambda'])
    
    # Drop rows where physics features are NaN (usually low mass stars for Slope14)
    df_clean = df.dropna(subset=['Slope14', 'Eps_Central', 'CS2_Central', 'Radius', 'Mass']).copy()
    y = df_clean['Label']
    groups = df_clean['Curve_ID']

    # 2. Cross-Validation Split
    # GroupShuffleSplit ensures no single EoS curve spans both Train and Test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(df_clean, y, groups=groups))
    
    # ==========================================
    # 3. HYPERPARAMETER CONFIGURATION (DEEP TUNED)
    # ==========================================
    
    # CONFIG 1: PHYSICS MODELS (B, C, D)
    # Result: 99.999% Accuracy.
    # - 'sqrt': Forces trees to use subtle features (Slope/Density).
    # - leaf=20: Robust against minor numerical noise.
    rf_params_phys = {
        'n_estimators': 1000,        # Scale up from grid search (200) to 1000 for stability
        'max_depth': 15,
        'min_samples_leaf': 20,      # Optimized value
        'max_features': 'sqrt',      # Constraint: Force decorrelation
        'max_samples': None,         # Use full dataset per tree
        'bootstrap': True,
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42
    }

    # CONFIG 2: OBSERVATIONAL MODELS (Geo, A)
    # Result: 95.0% Accuracy.
    # - max_samples=0.7: Bagging (70% data per tree) prevents memorizing specific curves.
    # - max_features=None: Always see Radius to fix low-mass errors.
    # - leaf=10: High resolution allowed because Bagging handles the noise.
    rf_params_obs = rf_params_phys.copy()
    rf_params_obs.update({
        'max_features': None,        # Grid Search Winner
        'max_samples': 0.7,          # Grid Search Winner (Aggressive Bagging)
        'min_samples_leaf': 10,      # Higher resolution
        'max_depth': 15
    })
    
    # Define Feature Sets
    cols_A = ['Mass', 'Radius', 'LogLambda']
    cols_B = ['Mass', 'Radius', 'LogLambda', 'Eps_Central']
    cols_C = ['Mass', 'Radius', 'LogLambda', 'Eps_Central', 'CS2_Central']
    cols_D = ['Mass', 'Radius', 'LogLambda', 'Eps_Central', 'CS2_Central', 'Slope14']
    cols_geo = ['Mass', 'Radius']

    # 4. Training Loop
    models = {}
    
    train_setup = [
        ('A', cols_A, rf_params_obs),      # Use Tuned Obs Config
        ('B', cols_B, rf_params_phys),     # Use Tuned Phys Config
        ('C', cols_C, rf_params_phys),     # Use Tuned Phys Config
        ('D', cols_D, rf_params_phys),     # Use Tuned Phys Config
        ('Geo', cols_geo, rf_params_obs)   # Use Tuned Obs Config
    ]

    for name, cols, params in train_setup:
        print(f"Training Model {name} (Calibrated)...")
        base_rf = RandomForestClassifier(**params)
        cal_rf = CalibratedClassifierCV(base_rf, method='isotonic', cv=3)
        cal_rf.fit(df_clean.iloc[train_idx][cols], y.iloc[train_idx])
        models[name] = cal_rf

    # 5. Diagnostics
    print("\n--- Overfitting Diagnostics (Train vs Test) ---")
    print(f"{'Model':<5} | {'Train Acc':<10} | {'Test Acc':<10} | {'Gap':<10} | {'Status'}")
    print("-" * 60)

    for name, cols, _ in train_setup:
        model = models[name]
        train_acc = model.score(df_clean.iloc[train_idx][cols], y.iloc[train_idx])
        test_acc = model.score(df_clean.iloc[test_idx][cols], y.iloc[test_idx])
        gap = train_acc - test_acc
        
        status = "OK"
        if gap > 0.05: status = "OvrFit"
        if train_acc < 0.70: status = "UndFit"
        
        print(f"{name:<5} | {train_acc:.4f}     | {test_acc:.4f}     | {gap:.4f}     | [{status}]")

    # 6. Patching for SHAP/Importance
    if 'D' in models:
        rf_d_cal = models['D']
        importances = np.mean([
            clf.estimator.feature_importances_ 
            for clf in rf_d_cal.calibrated_classifiers_
        ], axis=0)
        
        std_importances = np.std([
            clf.estimator.feature_importances_ 
            for clf in rf_d_cal.calibrated_classifiers_
        ], axis=0)
        
        rf_d_cal.feature_importances_ = importances
        rf_d_cal.feature_importances_std_ = std_importances
        rf_d_cal.base_model_for_shap = rf_d_cal.calibrated_classifiers_[0].estimator

    # 7. Return Data
    all_needed_cols = list(set(cols_D + cols_geo)) 
    X_test_all = df_clean.iloc[test_idx][all_needed_cols]
    y_test = y.iloc[test_idx]

    return models, X_test_all, y_test
