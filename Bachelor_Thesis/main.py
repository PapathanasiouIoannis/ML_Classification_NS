
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# --- CONFIGURATION & CONSTANTS ---
from src.const import CONSTANTS

# --- PHYSICS ENGINE ---
from src.physics.calculate_baselines import calculate_baselines
from src.physics.run_worker_wrapper import run_worker_wrapper

# --- MACHINE LEARNING PIPELINE ---
from src.ml_pipeline.train_model import train_model
from src.ml_pipeline.analyze_candidates import analyze_candidates
from src.ml_pipeline.advanced_analysis import run_advanced_analysis

# --- VISUALIZATION SUITE ---
from src.visualize.plot_theoretical_eos import plot_theoretical_eos
from src.visualize.plot_grand_summary import plot_grand_summary
from src.visualize.plot_physics_manifold import plot_physics_manifold, plot_manifold_curves
from src.visualize.plot_diagnostics import plot_diagnostics
from src.visualize.plot_3d_separation import plot_3d_separation
from src.visualize.plot_physical_insights import plot_physical_insights
from src.visualize.plot_corner import plot_corner
from src.visualize.plot_slope_evolution import plot_slope_evolution
from src.visualize.plot_slope_vs_radius import plot_slope_vs_radius
from src.visualize.plot_stability_window import plot_stability_window
from src.visualize.plot_advanced_diagnostics import plot_misclassification_map
from src.visualize.plot_pdp import plot_partial_dependence
from src.visualize.plot_surface_density import plot_surface_density
from src.visualize.plot_statistical_bands import plot_statistical_bands
from src.visualize.plot_microphysics_3d import plot_microphysics_3d
from src.visualize.plot_correlations import plot_correlations



# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "thesis_dataset.csv")

def main():
    print("===============================================================")
    print("   Neutron star classification model   ")
    print("===============================================================")

    # 1. Directory Setup
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # 2. Physics Initialization
    # Calculate max masses of parent hadronic models for homologous scaling
    baselines = calculate_baselines()
    
    # 3. Data Management Logic
    should_generate = True
    
    # Load Master Schema from Constants
    cols = CONSTANTS['COLUMN_SCHEMA']

    if os.path.exists(DATA_FILE):
        print(f"\n[INFO] Found existing dataset: {DATA_FILE}")
        try:
            df = pd.read_csv(DATA_FILE)
            
            # Check if existing data matches the current schema
            if all(c in df.columns for c in cols):
                print(f"[INFO] Schema validation passed. Loaded {len(df)} samples.")
                should_generate = False
            else:
                print("[WARN] Dataset schema mismatch. Regenerating...")
                should_generate = True
        except Exception as e:
            print(f"[ERROR] Could not read dataset: {e}")
            should_generate = True
    
    if should_generate:
        print("\n--- Step 1: Generating Training Data (Parallel) ---")
        
        # --- RUN CONFIGURATION ---
        # Adjust TOTAL_CURVES as needed (e.g., 2000 for Test, 20000 for Production)
        TOTAL_CURVES = 20000
        CURVES_PER_BATCH = 50 
        N_JOBS = -1 # Use all available CPU cores
        
        tasks = []
        num_batches = max(1, TOTAL_CURVES // CURVES_PER_BATCH)
        
        for i in range(num_batches):
            # Interleave Hadronic and Quark tasks for load balancing
            t_type = 'hadronic' if i % 2 == 0 else 'quark'
            tasks.append((t_type, CURVES_PER_BATCH, i, i))

        print(f"Spawning {len(tasks)} tasks to generate {TOTAL_CURVES} EoS curves...")
        
        # Execute Parallel Workers
        res = Parallel(n_jobs=N_JOBS)(delayed(run_worker_wrapper)(t, baselines) for t in tqdm(tasks))
        
        # Flatten results and create DataFrame
        valid_rows = [item for sublist in res if sublist is not None for item in sublist]
        df = pd.DataFrame(valid_rows, columns=cols)
        
        # --- AUTOMATIC CLASS BALANCING ---
        print("\n[Balancing] Checking Class Distribution...")
        counts = df['Label'].value_counts()
        print(counts)
        
        # Undersample the majority class to prevent ML bias
        min_count = counts.min()
        print(f"[Balancing] Downsampling to {min_count} samples per class.")
        
        df = df.groupby('Label').sample(n=min_count, random_state=42)
        
        # Shuffle to mix classes
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        print("\n[Info] Final Class Distribution:")
        print(df['Label'].value_counts())
        
        # Save Dataset
        df.to_csv(DATA_FILE, index=False)
        print(f"[SUCCESS] Saved balanced dataset ({len(df)} samples) to {DATA_FILE}.")

    # Reload df to ensure data types are consistent
    df = pd.read_csv(DATA_FILE)

    # 4. Machine Learning Phase
    # Returns dictionary of models and the held-out test set
    models_dict, X_test, y_test = train_model(df)
    
    # 5. Visualization Phase
    print("\n--- Step 4: Running Visualization Suite ---")
    
    # ML Diagnostics
    plot_diagnostics(models_dict, X_test, y_test)
    run_advanced_analysis(df, models_dict, X_test, y_test)
    
    # Physics Manifold (Data Driven)
    plot_grand_summary(df)
    plot_statistical_bands(df)
    plot_physics_manifold(df)
    plot_manifold_curves(df)
    plot_theoretical_eos(baselines, n_curves=100)
    
    # Feature Interpretation
    plot_3d_separation(df)
    plot_microphysics_3d(df)
    plot_corner(df)
    
    # Microphysics Verification
    plot_physical_insights(models_dict, df)
    plot_correlations(df)
    
    # Thesis-Specific Diagnostics
    plot_slope_evolution(df)     # Slope vs Sound Speed
    plot_slope_vs_radius(df)     # Slope vs Radius
    plot_stability_window(df)    # B vs Delta (QCD Check)
    
    # 6. Inference Phase
    analyze_candidates(models_dict)
    
    # Model Interpretability
    plot_misclassification_map(models_dict, X_test, y_test)
    plot_partial_dependence(models_dict, X_test)
    
    # Final Physics Checks
    plot_surface_density(df)
    
    print("\n===============================================================")
    print("             Classification completed succesfully                ")
    print("===============================================================")

if __name__ == "__main__":
    main()
