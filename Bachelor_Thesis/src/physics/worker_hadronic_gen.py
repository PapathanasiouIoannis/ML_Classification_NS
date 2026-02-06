

import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from src.const import CONSTANTS
from src.physics.get_eos_library import get_eos_library
from src.physics.solve_sequence import solve_sequence

def worker_hadronic_gen(n_curves_to_gen, baselines, seed_offset, batch_idx):
    """
    Worker process for generating Hadronic Star EoS curves.
    
    Parameters:
    - n_curves_to_gen: Number of valid curves required.
    - baselines: Dictionary of max masses for parent models (calculated previously).
    - seed_offset: Random seed for reproducibility.
    - batch_idx: ID for data labeling.
    
    Returns:
    - valid_data: List of star configurations matching the COLUMN_SCHEMA.
    """
    np.random.seed(seed_offset)
    
    # Load Physics Libraries
    core_lib, crust_funcs = get_eos_library()
    model_names = list(baselines.keys())
    
    # Map of transition pressures (MeV/fm^3)
    # Uses centralized values for standard models vs "PS" model
    transition_map = {
        name: CONSTANTS['P_TRANS_PS'] if name == "PS" else CONSTANTS['P_TRANS_DEFAULT'] 
        for name in model_names
    }
    
    valid_data = []
    curves_found = 0
    attempts = 0
    small_step = 1e-6
    
    # --- CONFIGURATION FROM CONSTANTS ---
    m_min_target = CONSTANTS['H_M_MIN_TARGET']
    m_max_target = CONSTANTS['H_M_MAX_TARGET']
    delta_limit = CONSTANTS['H_DELTA_LIMIT']
    
    max_attempts = n_curves_to_gen * 5000

    while curves_found < n_curves_to_gen and attempts < max_attempts:
        attempts += 1

        # Pick two parent models and a mixing weight
        nA, nB = np.random.choice(model_names, 2, replace=False)
        w = np.random.uniform(0.20, 0.80)
        
        # Calculate Weighted Average of parent maximum masses
        base_max_m = w * baselines[nA] + (1-w) * baselines[nB]
        
        # Inverse Sampling
        # Target = Base * (1 + delta)
        # Therefore: delta = (Target / Base) - 1
        
        # What delta do we need to hit the mass floor?
        req_delta_min = (m_min_target / base_max_m) - 1.0
        
        # What delta do we need to hit the mass ceiling?
        req_delta_max = (m_max_target / base_max_m) - 1.0
        
        # Intersect with our allowed Smoothness Limit [-DELTA_LIMIT, +DELTA_LIMIT]
        effective_min = max(req_delta_min, -delta_limit)
        effective_max = min(req_delta_max, delta_limit)
        
        # If the valid window is empty, this parent combination cannot produce
        # a star in the desired mass range within scaling limits. Skip.
        if effective_min >= effective_max:
            continue
            
        # Sample delta from the VALID intersection only
        delta = np.random.uniform(effective_min, effective_max)
        

        # 2. SETUP PHYSICS

        target_m = base_max_m * (1.0 + delta)
        alpha = (base_max_m / target_m)**2
        
        fA = core_lib[nA]
        fB = core_lib[nB]
        

        # Apply +/- 20% noise to blur the discrete parent transition pressures
        p_base_mix = w * transition_map[nA] + (1-w) * transition_map[nB]
        p_jitter = np.random.uniform(0.80, 1.20)
        p_trans_mix = p_base_mix * p_jitter
        
        # EoS Input Tuple: (fA, dfA, fB, dfB, w, crusts, alpha, p_trans)
        eos_input = (fA[0], fA[1], fB[0], fB[1], w, crust_funcs, alpha, p_trans_mix)
        
        
        # 3. SOLVE STRUCTURE

        curve, max_m = solve_sequence(eos_input, is_quark=False)

        # Final Verification Checks
        if max_m < m_min_target or max_m > m_max_target: continue
        
        c_arr = np.array(curve)
        

        # Reject if Speed of Sound hits 1.0 (clamped) at low densities 
        # Column 4: Eps_Central, Column 5: CS2_Central
        eps_limit = CONSTANTS['CAUSALITY_EPS_LIMIT']
        violation_mask = (c_arr[:, 5] >= 0.999) & (c_arr[:, 4] < eps_limit)
        
        if np.any(violation_mask): continue
        
        # --- FILTER: RADIUS CAP ---

        mask_canonical = c_arr[:, 0] > 1.4
        if np.any(mask_canonical):
            if np.max(c_arr[mask_canonical, 1]) > 14.0: 
                continue
        

        if c_arr[0,0] > 1.4: continue
        

        # 4. FEATURE EXTRACTION

        try:
            # Sort by Mass for interpolation
            c_arr = c_arr[c_arr[:,0].argsort()] 
            

            f_R = PchipInterpolator(c_arr[:,0], c_arr[:,1])
            f_CS2 = interp1d(c_arr[:,0], c_arr[:,5], kind='linear', fill_value="extrapolate")
            
            # Extract features at 1.4 M_sun
            r_14 = float(f_R(1.4))
            cs2_at_14 = float(f_CS2(1.4))
            
            # Calculate Slopes (dR/dM)
            slopes = {}
            for m_step in [1.4, 1.6, 1.8, 2.0]:
                if max_m > m_step:
                    r_minus = f_R(m_step)
                    r_plus = f_R(m_step + small_step)
                    slope = (r_plus - r_minus) / small_step
                    slopes[m_step] = slope
                else:
                    slopes[m_step] = np.nan
            
        except Exception: 
            continue


        # 5. SAVE DATA

        curves_found += 1
        curve_id = f"H_{batch_idx}_{attempts}"
        
        for pt in curve:
            # Save points up to Max Mass (Stable Branch Only)
            if pt[0] > 0.1 and pt[0] < max_m:
                valid_data.append([
                    pt[0],      # Mass
                    pt[1],      # Radius
                    pt[2],      # Lambda
                    0,          # Label (0 = Hadronic)
                    curve_id,   # Group ID
                    pt[3],      # P_Central
                    pt[4],      # Eps_Central
                    pt[6],      # Eps_Surface
                    pt[5],      # CS2_Central
                    
                    # Diagnostics
                    cs2_at_14,
                    r_14,
                    slopes.get(1.4, np.nan),
                    slopes.get(1.6, np.nan),
                    slopes.get(1.8, np.nan),
                    slopes.get(2.0, np.nan),
                    
                    # Quark Params (NaN for Hadronic)
                    np.nan, np.nan, np.nan
                ])
                
    return valid_data
