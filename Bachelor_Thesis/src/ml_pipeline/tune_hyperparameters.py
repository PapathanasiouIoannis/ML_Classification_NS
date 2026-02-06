

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
import os

def tune_hyperparameters():
    print("=========================================================")
    print("   HYPERPARAMETER OPTIMIZATION (GRID SEARCH)")
    print("=========================================================")
    
    # 1. Load Data
    data_path = os.path.join("data", "thesis_dataset.csv")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Run main.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Feature Engineering
    if 'LogLambda' not in df.columns:
        df['LogLambda'] = np.log10(df['Lambda'])
    
    # Cleaning (Must match train_model.py)
    df_clean = df.dropna(subset=['Slope14', 'Eps_Central', 'CS2_Central', 'Radius', 'Mass']).copy()
    y = df_clean['Label']
    groups = df_clean['Curve_ID']
    
    # Define Splitter
    # We use 3 folds to keep it reasonably fast
    cv_splitter = GroupKFold(n_splits=3)

    # =======================================================
    # SCENARIO 1: TUNING MODEL A (OBSERVATIONAL)
    # =======================================================
    print("\n--- Tuning Model A (Observational) ---")
    print("Goal: Maximize Accuracy. Testing if max_features=None is statistically superior.")
    
    X_A = df_clean[['Mass', 'Radius', 'LogLambda']]
    
    param_grid_obs = {
        'n_estimators': [200], # Lower than 1000 for speed
        'max_depth': [10, 15],
        'min_samples_leaf': [10, 30, 50, 100],
        'max_features': ['sqrt', None], # The big question
        'bootstrap': [True]
    }
    
    grid_A = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        param_grid=param_grid_obs,
        cv=cv_splitter,
        scoring='accuracy',
        verbose=1,
        n_jobs=1 
    )
    
    grid_A.fit(X_A, y, groups=groups)
    
    print(f"Best Params (Model A): {grid_A.best_params_}")
    print(f"Best Accuracy: {grid_A.best_score_:.4f}")

    # =======================================================
    # SCENARIO 2: TUNING MODEL D (PHYSICS)
    # =======================================================
    print("\n--- Tuning Model D (Microphysics) ---")
    print("Goal: Maximize Accuracy while enforcing feature diversity.")
    print("Constraint: max_features is LOCKED to 'sqrt'.")
    
    X_D = df_clean[['Mass', 'Radius', 'LogLambda', 'Eps_Central', 'CS2_Central', 'Slope14']]
    
    param_grid_phys = {
        'n_estimators': [200],
        'max_depth': [10, 15, 20], 
        'min_samples_leaf': [5, 10, 20, 30], 
        'max_features': ['sqrt'], # LOCKED
        'bootstrap': [True]
    }
    
    grid_D = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        param_grid=param_grid_phys,
        cv=cv_splitter,
        scoring='accuracy',
        verbose=1,
        n_jobs=1
    )
    
    grid_D.fit(X_D, y, groups=groups)
    
    print(f"Best Params (Model D): {grid_D.best_params_}")
    print(f"Best Accuracy: {grid_D.best_score_:.4f}")
    

    print("Suggested values")

if __name__ == "__main__":
    tune_hyperparameters()