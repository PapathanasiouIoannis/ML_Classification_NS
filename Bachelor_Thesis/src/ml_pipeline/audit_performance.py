

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from src.visualize.style_config import set_paper_style

def run_performance_audit(models_dict, X_test, y_test):
    set_paper_style()
    print("\n=========================================================")
    print("   PERFORMANCE AUDIT: STRESS TESTING THE MODELS")
    print("=========================================================")

    # Define the models we care about comparing
    target_models = ['Geo', 'A', 'D']
    valid_models = [m for m in target_models if m in models_dict]
    

    feature_sets = {
        'Geo': ['Mass', 'Radius'],
        'A':   ['Mass', 'Radius', 'LogLambda'],
        'D':   ['Mass', 'Radius', 'LogLambda', 'Eps_Central', 'CS2_Central', 'Slope14']
    }
    
    # Define Mass Bins
    bins = [0.1, 1.0, 1.4, 1.8, 2.0, 2.2, 2.4, 3.0]
    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    
    # Store results
    results = {name: [] for name in valid_models}
    counts = []


    # TEST 1: MASS-DEPENDENT ACCURACY

    print("\n[Audit 1] Calculating Accuracy per Mass Bin...")
    
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        
        # Filter Test Set for this Bin
        mask = (X_test['Mass'] >= low) & (X_test['Mass'] < high)
        X_sub = X_test[mask]
        y_sub = y_test[mask]
        
        counts.append(len(y_sub))
        
        if len(y_sub) == 0:
            for name in valid_models: results[name].append(np.nan)
            continue
            
        for name in valid_models:
            model = models_dict[name]
            cols = feature_sets[name] 
            
            # Ensure columns exist in X_sub before scoring
            if not all(col in X_sub.columns for col in cols):
                print(f"Warning: Missing columns for Model {name}. Skipping.")
                results[name].append(np.nan)
                continue

            acc = model.score(X_sub[cols], y_sub)
            results[name].append(acc)

    # --- PLOT 1: ACCURACY VS MASS ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    x_pos = np.arange(len(bin_labels))
    
    # Plot Accuracy lines
    colors = {'Geo': 'gray', 'A': '#1f77b4', 'D': '#2ca02c'}
    markers = {'Geo': 'x', 'A': 'o', 'D': 's'}
    
    for name in valid_models:
        ax1.plot(x_pos, results[name], marker=markers[name], color=colors[name], 
                 linewidth=2.5, label=f"Model {name}", alpha=0.9)

    ax1.set_ylabel("Classification Accuracy")
    ax1.set_title("The 'Zone of Confidence': Accuracy vs. Neutron Star Mass")
    ax1.set_ylim(0.5, 1.02)
    ax1.axhline(0.90, color='red', linestyle=':', label='90% Reliability Threshold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left')
    
    # Plot Sample Count Bar Chart 
    ax2.bar(x_pos, counts, color='gray', alpha=0.3)
    ax2.set_ylabel("N Samples")
    ax2.set_xlabel(r"Mass Range [$M_{\odot}$]")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bin_labels)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show() 


    # TEST 2: CONFIDENCE OF ERRORS (Calibration Check)

    print("\n[Audit 2] Analyzing Confidence of Misclassifications (Model A)...")
    
    if 'A' in models_dict:
        model = models_dict['A']
        cols = feature_sets['A']
        
        # Get predictions
        probs = model.predict_proba(X_test[cols])[:, 1]
        preds = (probs > 0.5).astype(int)
        
        # Isolate Errors
        error_mask = (preds != y_test)
        error_probs = probs[error_mask]
        
        # Calculate "Confidence" (Distance from 0.5)
        # 0.5 = Uncertain, 1.0 or 0.0 = Confident
        error_conf = 2 * np.abs(error_probs - 0.5)
        
        print(f"Total Errors in Model A: {len(error_probs)}")
        print(f"Mean Confidence on Errors: {np.mean(error_conf):.2f} (0=Unsure, 1=Certain)")
        
        # Plot Histogram
        plt.figure(figsize=(8, 5))
        sns.histplot(error_probs, bins=20, color='red', alpha=0.6, kde=True)
        plt.axvline(0.5, color='black', linestyle='--', label='Ideal Error (Uncertain)')
        plt.title("Model A: Predicted Probability for WRONG Answers")
        plt.xlabel(r"Predicted $P(\text{Quark})$")
        plt.ylabel("Count of Errors")
        plt.xlim(0, 1)
        plt.legend()
        plt.show()
        
        if np.mean(error_conf) < 0.4:
            print(">> VERDICT: GOOD. The model is uncertain when it is wrong.")
        else:
            print(">> VERDICT: WARNING. The model is confidently wrong on some points.")


    # TEST 3: THE HIGH-MASS BREAKDOWN
    print("\n[Audit 3] High-Mass Performance (M > 2.0 M_sun)")
    
    mask_high = X_test['Mass'] > 2.0
    y_high = y_test[mask_high]
    X_high = X_test[mask_high]
    
    print(f"High-Mass Test Set Size: {len(y_high)}")
    print("-" * 40)
    print(f"{'Model':<10} | {'Accuracy':<10} | {'Status'}")
    print("-" * 40)
    
    for name in valid_models:
        cols = feature_sets[name]
        acc = models_dict[name].score(X_high[cols], y_high)
        status = "CRITICAL FAILURE" if acc < 0.6 else "ROBUST" if acc > 0.9 else "DEGRADED"
        print(f"{name:<10} | {acc:.4f}     | {status}")
