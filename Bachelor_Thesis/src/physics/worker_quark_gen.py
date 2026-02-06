

import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from src.const import CONSTANTS
from src.physics.solve_sequence import solve_sequence

def worker_quark_gen(n_curves_to_gen, seed_offset, batch_idx):
    """
    Worker process for generating Quark Star EoS curves.
    
    Parameters:
    - n_curves_to_gen: Number of valid curves required.
    - seed_offset: Random seed for reproducibility.
    - batch_idx: ID for data labeling.
    
    Returns:
    - valid_data: List of star configurations matching the COLUMN_SCHEMA.
    """
    np.random.seed(seed_offset)
    valid_data = []
    curves_found = 0
    attempts = 0
    small_step = 1e-5
    
    # Physical Constants
    hc = CONSTANTS['HC']   
    m_n = CONSTANTS['M_N'] 
    
    max_attempts = n_curves_to_gen * 5000 

    while curves_found < n_curves_to_gen and attempts < max_attempts:
        attempts += 1
        

        # 1. SAMPLE MICROPHYSICS (Delta, ms)

        # Sample from the physical prior ranges defined in CONSTANTS
        ms_MeV = np.random.uniform(*CONSTANTS['Q_MS_RANGE'])
        Delta_MeV = np.random.uniform(*CONSTANTS['Q_DELTA_RANGE'])
        

        # 2. CALCULATE STABILITY LIMIT (B_max)

        mu_limit = m_n / 3.0
        term1 = (3.0 / (4.0 * np.pi**2)) * (mu_limit**4)
        eff_gap_sq = Delta_MeV**2 - (ms_MeV**2 / 4.0)
        term2 = (3.0 / np.pi**2) * eff_gap_sq * (mu_limit**2)
        
        B_max = (term1 + term2) / (hc**3)
        B_min = CONSTANTS['Q_B_MIN']

        # If the parameter space is invalid (max < min), skip iteration
        if B_max <= B_min: continue 
            

        # 3. TARGET-DRIVEN SAMPLING (Inverse Method)

        
        # Target Mass Range: 1.0 to Q_M_MAX_UPPER ( 3.0)
        target_m_max = np.random.uniform(1.0, CONSTANTS['Q_M_MAX_UPPER'])
        
        # A. Inverse Scaling Law (Empirical approximation)
        # B ~ 1/M^2. Baseline guess assuming Delta=0.
        B_guess = 58.0 * (2.03 / target_m_max)**2
        
        # B. Delta Correction

        delta_stiffness_factor = 1.0 + (1.5 * (Delta_MeV / 500.0))
        B_guess_corrected = B_guess * delta_stiffness_factor

        # C. Add Noise / Perturbation

        noise = np.random.uniform(0.75, 1.30)
        B_target = B_guess_corrected * noise
        
        # D. Clamp to Physics Bounds
        real_upper = min(B_max, CONSTANTS['Q_B_ABS_MAX'])
        if B_target < B_min: B_target = B_min
        if B_target > real_upper: B_target = real_upper
        
        B = B_target
        
        # Convert to Geometric Units for Solver
        Delta_geom = Delta_MeV / hc
        ms_geom = ms_MeV / hc
        

        # 4. SOLVE TOV SEQUENCE

        curve, max_m = solve_sequence((B, Delta_geom, ms_geom), is_quark=True)
        

        # 5. VALIDATION FILTERS

        # Mass Filter
        if max_m < 1.0: continue
        if max_m > CONSTANTS['Q_M_MAX_UPPER']: continue 
        
        # Radius Filter 
        if np.max(np.array(curve)[:,1]) > CONSTANTS['Q_R_MAX']: continue
        
        try:
            # Sort curve by Mass for interpolation
            c_arr = np.array(curve)
            c_arr = c_arr[c_arr[:,0].argsort()]
            

            # Ensure the curve spans the relevant mass range
            if c_arr[0,0] > 1.0 or c_arr[-1,0] < 1.0: continue


            f_R = PchipInterpolator(c_arr[:,0], c_arr[:,1])
            f_CS2 = interp1d(c_arr[:,0], c_arr[:,5], kind='linear', fill_value="extrapolate")
            
            # Extract Features at 1.4 M_sun
            if max_m >= 1.4:
                cs2_at_14 = float(f_CS2(1.4))
                r_14 = float(f_R(1.4))
                
                # Calculate Slopes (dR/dM) at key mass steps
                slopes = {}
                for m_step in [1.4, 1.6, 1.8, 2.0]:
                    if max_m > m_step:
                        r_minus = f_R(m_step)
                        r_plus = f_R(m_step + small_step)
                        slopes[m_step] = (r_plus - r_minus) / small_step
                    else:
                        slopes[m_step] = np.nan
            else:

                cs2_at_14 = np.nan
                r_14 = np.nan
                slopes = {m: np.nan for m in [1.4, 1.6, 1.8, 2.0]}
            
        except Exception:
            continue

        # 6. SAVE DATA

        curves_found += 1 
        curve_id = f"Q_{batch_idx}_{attempts}"
        
        # Flatten curve points into dataset rows
        for pt in curve:
            m_val = pt[0]
            # Save only stable branch points within valid mass range
            if m_val > CONSTANTS['Q_M_MIN'] and m_val <= max_m:
                valid_data.append([
                    m_val,          # Mass
                    pt[1],          # Radius
                    pt[2],          # Lambda
                    1,              # Label (1 = Quark)
                    curve_id,       # Group ID
                    pt[3],          # P_Central
                    pt[4],          # Eps_Central
                    pt[6],          # Eps_Surface
                    pt[5],          # CS2_Central
                    
                    # Diagnostics
                    cs2_at_14,
                    r_14,
                    slopes.get(1.4, np.nan),
                    slopes.get(1.6, np.nan),
                    slopes.get(1.8, np.nan),
                    slopes.get(2.0, np.nan),
                    
                    # Microphysics Parameters
                    B,          # Bag_B
                    Delta_MeV,  # Gap_Delta
                    ms_MeV      # Mass_Strange
                ])
                
    return valid_data
