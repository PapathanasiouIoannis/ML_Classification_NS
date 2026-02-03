
import numpy as np
import pandas as pd
from src.const import CONSTANTS

def analyze_candidates(models):
    """
    Infers the nature (Hadronic vs Quark) of real astrophysical objects.
    
    Parameters:
    - models: Dictionary containing trained classifiers (Expected: 'A' and 'Geo').
    """
    # Retrieve classifiers
    rf_full = models['A']   # Expects [Mass, Radius, LogLambda]
    rf_geo  = models['Geo'] # Expects [Mass, Radius]
    
    print("\n" + "="*90)
    print(f"{'INFERENCE ON REAL ASTROPHYSICAL DATA':^90}")
    print("="*90)
    
    # Real Data Catalog (Mean and 1-sigma uncertainties)
    # L=0 is a placeholder for objects where Lambda is unmeasured.
    # Data Sources: LIGO/Virgo (GW), NICER (PSR)
    candidates = [
        {"Name": "GW170817",     "M": 1.40, "sM": 0.10, "R": 11.90, "sR": 1.40, "L": 190, "sL": 120},
        {"Name": "PSR J0740+66", "M": 2.08, "sM": 0.07, "R": 12.35, "sR": 0.75, "L": 0,   "sL": 0},
        {"Name": "PSR J0030+04", "M": 1.44, "sM": 0.15, "R": 13.02, "sR": 1.06, "L": 0,   "sL": 0},
        {"Name": "HESS J1731",   "M": 0.77, "sM": 0.17, "R": 10.40, "sR": 0.78, "L": 0,   "sL": 0},
        {"Name": "GW190814(sec)","M": 2.59, "sM": 0.09, "R": 12.00, "sR": 3.00, "L": 0,   "sL": 0}
    ]
    
    print(f"{'Candidate':<20} | {'Mass':<8} | {'Radius':<8} | {'Model Used':<15} | {'P(Quark)':<10} | {'Verdict':<10}")
    print("-" * 100)

    for star in candidates:
        # ==========================================
        # 1. MONTE CARLO SAMPLING
        # ==========================================
        # Generate 5000 realizations of the star from its Error Distribution
        n_mc = 5000 
        
        raw_m = np.random.normal(star['M'], star['sM'], n_mc)
        raw_r = np.random.normal(star['R'], star['sR'], n_mc)
        raw_l = np.random.normal(star['L'], star['sL'], n_mc)
        
        # ==========================================
        # 2. PHYSICALITY FILTERING
        # ==========================================
        # Truncate the Gaussian to physical values
        # Radius > 8km (Causal/BH limit proximity), Mass > 0.1 M_sun
        valid_mask = (raw_r > 8.0) & (raw_m > 0.1)
        
        # Check if this object has Tidal Data available (GW events)
        has_tidal = (star['Name'] == "GW170817")
        
        if has_tidal: 
            # If using Tidal model, Lambda must be positive to take log10
            valid_mask = valid_mask & (raw_l >= 1.0)
            
        m_s = raw_m[valid_mask]
        r_s = raw_r[valid_mask]
        l_s = raw_l[valid_mask]
        
        # Skip if sampling failed (e.g., extremely unphysical parameters)
        if len(m_s) == 0: 
            continue 

        # ==========================================
        # 3. MODEL SELECTION & PREDICTION
        # ==========================================
        if has_tidal:
            # PATH A: Use Full Physics Model
            # Input features must match Model A training: [Mass, Radius, LogLambda]
            X_mc = pd.DataFrame({
                'Mass': m_s, 
                'Radius': r_s, 
                'LogLambda': np.log10(l_s)
            })
            probs = rf_full.predict_proba(X_mc)[:, 1]
            m_name = "Model A (Full)"
        else:
            # PATH B: Use Geometric Model
            # Input features must match Model Geo training: [Mass, Radius]
            X_mc = pd.DataFrame({
                'Mass': m_s, 
                'Radius': r_s
            })
            probs = rf_geo.predict_proba(X_mc)[:, 1]
            m_name = "Model Geo"
        
        # ==========================================
        # 4. MARGINALIZATION (SOFT VOTING)
        # ==========================================
        # The final probability is the mean over all Monte Carlo samples
        mean_p = np.mean(probs)
        verdict = "QUARK" if mean_p > 0.5 else "HADRONIC"
        
        # Output Row
        print(f"{star['Name']:<20} | {star['M']:<8.2f} | {star['R']:<8.2f} | {m_name:<15} | {mean_p*100:5.1f}%     | {verdict:<10}")
        
    print("="*90 + "\n")