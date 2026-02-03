
import numpy as np
from src.const import CONSTANTS
from src.physics.get_eos_library import get_eos_library
from src.physics.solve_sequence import solve_sequence

def worker_get_plot_curve(mode, baselines, seed):
    """
    Generates a single high-resolution EoS curve and its corresponding P-Epsilon grid.
    
    Parameters:
    - mode: 'hadronic' or 'quark'
    - baselines: Dictionary of hadronic max masses (only used if mode='hadronic')
    - seed: Random seed
    
    Returns:
    - (curve_data, eos_grid_data)
    """
    np.random.seed(seed)
    
    # Physical Constants
    hc = CONSTANTS['HC']   
    m_n = CONSTANTS['M_N'] 
    
    if mode == 'hadronic':
        # ==========================================
        # HADRONIC LOGIC (Matches worker_hadronic_gen)
        # ==========================================
        core_lib, crust_funcs = get_eos_library()
        model_names = list(baselines.keys())
        
        # Transition Pressure Map
        transition_map = {
            name: CONSTANTS['P_TRANS_PS'] if name == "PS" else CONSTANTS['P_TRANS_DEFAULT'] 
            for name in model_names
        }
        
        # Configuration
        m_min_target = CONSTANTS['H_M_MIN_TARGET']
        m_max_target = CONSTANTS['H_M_MAX_TARGET']
        delta_limit = CONSTANTS['H_DELTA_LIMIT']
        
        # Loop until a valid curve is found
        while True:
            # 1. SMART PARAMETER SELECTION
            nA, nB = np.random.choice(model_names, 2, replace=False)
            w = np.random.uniform(0.20, 0.80)
            
            # Weighted Average Base Mass
            base_max_m = w * baselines[nA] + (1-w) * baselines[nB]
            
            # --- Inverse Sampling Trick ---
            req_delta_min = (m_min_target / base_max_m) - 1.0
            req_delta_max = (m_max_target / base_max_m) - 1.0
            
            effective_min = max(req_delta_min, -delta_limit)
            effective_max = min(req_delta_max, delta_limit)
            
            if effective_min >= effective_max:
                continue
                
            delta = np.random.uniform(effective_min, effective_max)
            
            # 2. SETUP PHYSICS
            target_m = base_max_m * (1.0 + delta)
            alpha = (base_max_m / target_m)**2
            
            fA = core_lib[nA]
            fB = core_lib[nB]
            
            # Transition Jitter
            p_base_mix = w * transition_map[nA] + (1-w) * transition_map[nB]
            p_jitter = np.random.uniform(0.80, 1.20)
            p_trans_mix = p_base_mix * p_jitter
            
            eos_input = (fA[0], fA[1], fB[0], fB[1], w, crust_funcs, alpha, p_trans_mix)
            
            # 3. SOLVE STRUCTURE
            curve, max_m = solve_sequence(eos_input, is_quark=False)
            
            # 4. VALIDATION FILTERS (Strict match to worker)
            if max_m < m_min_target or max_m > m_max_target: continue
            
            c = np.array(curve)
            eps_limit = CONSTANTS['CAUSALITY_EPS_LIMIT']
            # Causality Clamp Check
            violation_mask = (c[:, 5] >= 0.999) & (c[:, 4] < eps_limit)
            if np.any(violation_mask): continue
            
            # Radius Cap
            mask_canonical = c[:, 0] > 1.4
            if np.any(mask_canonical):
                if np.max(c[mask_canonical, 1]) > 14.0: 
                    continue
            
            # Must resolve low mass branch
            if c[0,0] > 1.4: continue

            # 5. GENERATE DENSE EOS GRID (For Plotting P vs Eps)
            try:
                # Dense grid for visualization
                p_max = 3000.0 
                p_grid = np.logspace(-4.0, np.log10(p_max), 100)
                
                eps_grid = []
                last_e = -1.0
                last_p = -1.0
                
                for p_val in p_grid:
                    if p_val > p_trans_mix:
                        # CORE REGION
                        p_base = p_val / alpha
                        vA = fA[0](p_base)
                        vB = fB[0](p_base)
                        if vA > 0 and vB > 0:
                            raw_e = ((vA**w) * (vB**(1.0-w))) * alpha
                            
                            # Apply Causality Clamp to the Grid points too
                            if last_e > 0:
                                d_p = p_val - last_p
                                d_e_raw = raw_e - last_e
                                if d_e_raw < d_p: 
                                    raw_e = last_e + d_p
                            
                            eps_grid.append(raw_e)
                            last_e = raw_e
                            last_p = p_val
                        else: 
                            eps_grid.append(np.nan)
                    else:
                        # CRUST REGION (Use constants)
                        if p_val > CONSTANTS['P_C1']: val = crust_funcs['c1'][0](p_val)
                        elif p_val > CONSTANTS['P_C2']: val = crust_funcs['c2'][0](p_val)
                        elif p_val > CONSTANTS['P_C3']: val = crust_funcs['c3'][0](p_val)
                        else: val = crust_funcs['c4'][0](p_val)
                        eps_grid.append(val)

                eps_grid = np.array(eps_grid)
                valid_mask = ~np.isnan(eps_grid)
                # Stack: [EnergyDensity, Pressure]
                eos_arr = np.column_stack((eps_grid[valid_mask], p_grid[valid_mask]))
                
                return (c, eos_arr)
            except: 
                continue

    else:
        # ==========================================
        # QUARK LOGIC (Matches worker_quark_gen)
        # ==========================================
        while True:
            # 1. Parameter Selection
            ms_MeV = np.random.uniform(*CONSTANTS['Q_MS_RANGE'])
            Delta_MeV = np.random.uniform(*CONSTANTS['Q_DELTA_RANGE'])
            
            # 2. Stability Window
            mu_limit = m_n / 3.0
            term1 = (3.0 / (4.0 * np.pi**2)) * (mu_limit**4)
            eff_gap_sq = Delta_MeV**2 - (ms_MeV**2 / 4.0)
            term2 = (3.0 / np.pi**2) * eff_gap_sq * (mu_limit**2)
            
            B_max = (term1 + term2) / (hc**3)
            B_min = CONSTANTS['Q_B_MIN']
            
            if B_max <= B_min: continue 
            
            # 3. Target-Driven Sampling
            target_m_max = np.random.uniform(1.0, CONSTANTS['Q_M_MAX_UPPER'])
            
            # Baseline scaling
            B_guess = 58.0 * (2.03 / target_m_max)**2
            
            # Stiffness correction for Gap
            delta_stiffness_factor = 1.0 + (1.5 * (Delta_MeV / 500.0))
            B_guess_corrected = B_guess * delta_stiffness_factor
    
            # Add Noise
            noise = np.random.uniform(0.75, 1.30)
            B_target = B_guess_corrected * noise
            
            # Bounds
            real_upper = min(B_max, CONSTANTS['Q_B_ABS_MAX'])
            if B_target < B_min: B_target = B_min
            if B_target > real_upper: B_target = real_upper
            
            B = B_target
            
            # Geometric Units
            Delta_geom = Delta_MeV / hc
            ms_geom = ms_MeV / hc
            
            # 4. Solve Structure
            curve, max_m = solve_sequence((B, Delta_geom, ms_geom), is_quark=True)
            
            # 5. Validate
            if max_m < 1.0 or max_m > CONSTANTS['Q_M_MAX_UPPER']: continue
            
            c = np.array(curve)
            if np.max(c[:,1]) > CONSTANTS['Q_R_MAX']: continue
            
            # 6. Generate Dense EoS Grid
            p_max = 3000.0 
            p_grid = np.logspace(-4.0, np.log10(p_max), 100)
            eps_grid = []
            
            # Algebra Coefficients (Geometric)
            coeff_a = 3.0 / (4.0 * np.pi**2)
            eff_gap_term = Delta_geom**2 - (ms_geom**2 / 4.0)
            coeff_b = 3.0 * eff_gap_term / (np.pi**2)
            B_geom = B / hc
            
            for p_val in p_grid:
                p_val_geom = p_val / hc
                
                # Solve algebraic EoS: P -> mu -> epsilon
                coeff_c = -(p_val_geom + B_geom)
                det = coeff_b**2 - 4*coeff_a*coeff_c
                
                if det < 0: 
                    eps_grid.append(np.nan)
                    continue
                    
                mu2 = (-coeff_b + np.sqrt(det)) / (2*coeff_a)
                val_eps_geom = 3.0 * coeff_a * (mu2**2) + coeff_b * mu2 + B_geom
                val_eps = val_eps_geom * hc
                eps_grid.append(val_eps)
            
            eps_grid = np.array(eps_grid)
            valid_mask = ~np.isnan(eps_grid)
            eos_arr = np.column_stack((eps_grid[valid_mask], p_grid[valid_mask]))
            
            return (c, eos_arr)