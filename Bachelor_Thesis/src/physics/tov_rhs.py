# ==============================================================================
# HEADER: src/physics/tov_rhs.py
# ==============================================================================
# Description:
#   Computes the derivatives for the Tolman-Oppenheimer-Volkoff (TOV) equations
#   coupled with the Tidal Deformability Riccati equation.
#
# Cleaned & Refactored:
#   - Replaced hardcoded physics constants with `src.const.CONSTANTS`.
#   - Replaced SLy crust transition thresholds with centralized values.
#   - Unified safe pressure floor using `TOV_P_MIN_SAFE`.
#   - Added comments explaining the geometric unit conversions and causality clamps.
# ==============================================================================

import numpy as np
from src.const import CONSTANTS

def tov_rhs(r, y_state, eos_data, is_quark):
    """
    Computes the derivatives for the TOV and Tidal Deformability equations.
    
    Parameters:
    - r: Current radius (integration variable)
    - y_state: [Mass, Pressure, y_tidal]
    - eos_data: Tuple containing EoS parameters:
        * Quark: (B, Delta_geom, ms_geom)
        * Hadronic: (fA_e, fA_de, fB_e, fB_de, w, crusts, alpha, P_trans)
    - is_quark: Boolean flag indicating the type of star.
    
    Returns:
    - [dm_dr, dP_dr, dy_dr]
    """
    m, P, y_tidal = y_state
    
    # Ensure P is never negative or zero to avoid log errors in crust formulas
    P_safe = max(P, CONSTANTS['TOV_P_MIN_SAFE'])

    # ==========================================
    # 1. MICROPHYSICS (Thermodynamics)
    # ==========================================
    if is_quark:
        # --- QUARK LOGIC (Generalized CFL) ---
        # Unpack parameters: 
        # B (MeV/fm^3), Delta (fm^-1), ms (fm^-1)
        # Note: Delta and ms are passed as geometric units from the worker.
        B, Delta_geom, ms_geom = eos_data
        
        # Physical Constant: hbar*c [MeV*fm]
        hc = CONSTANTS['HC']
        
        # 1. CONVERT TO GEOMETRIC UNITS (fm^-4)
        P_geom = P_safe / hc
        B_geom = B / hc
        
        # 2. SOLVE IN GEOMETRIC SPACE
        # Analytic relation: P = 3/(4pi^2)*mu^4 + 3/pi^2*(Delta^2 - ms^2/4)*mu^2 - B
        coeff_a = 3.0 / (4.0 * np.pi**2)
        
        # Effective Gap Term (Delta^2 - ms^2/4)
        eff_gap_term = Delta_geom**2 - (ms_geom**2 / 4.0)
        
        coeff_b = 3.0 * eff_gap_term / (np.pi**2)
        coeff_c = -(P_geom + B_geom)
        
        # Solve Quadratic for mu^2
        det = coeff_b**2 - 4*coeff_a*coeff_c
        mu2 = (-coeff_b + np.sqrt(max(0, det))) / (2*coeff_a)
        
        # 3. CALCULATE EPSILON (Geometric: fm^-4)
        eps_geom = 3.0 * coeff_a * (mu2**2) + coeff_b * mu2 + B_geom
        
        # 4. CONVERT BACK TO LAB UNITS (MeV/fm^3)
        epsilon = eps_geom * hc
        
        # 5. CALCULATE INVERSE SPEED OF SOUND (Dimensionless)
        # dedp = 1/cs2
        term_shift = 2.0 * eff_gap_term
        numerator = 3.0 * mu2 + term_shift
        denominator = mu2 + term_shift
        
        if abs(denominator) < 1e-10: 
            dedp = 3.0 
        else: 
            dedp = numerator / denominator

    else:
        # --- HADRONIC LOGIC (Mixed Core + Multi-Layer Crust) ---
        fA_e, fA_de, fB_e, fB_de, w, crusts, alpha, P_trans = eos_data
        
        # Crust Thresholds (MeV/fm^3) - Based on Douchin & Haensel (SLy)
        P_c1 = CONSTANTS['P_C1']
        P_c2 = CONSTANTS['P_C2']
        P_c3 = CONSTANTS['P_C3']
        
        if P_safe > P_trans:
            # --- CORE REGION ---
            # Apply Homologous Scaling and Mixing
            P_base = P_safe / alpha
            valA, valB = fA_e(P_base), fB_e(P_base)
            
            if valA <= 0 or valB <= 0: return [0, 0, 0]
            
            # Mix Energy Density
            epsilon_base = (valA**w) * (valB**(1.0-w))
            epsilon = epsilon_base * alpha
            
            # Mix Derivatives (Chain Rule for 1/cs2)
            dedpA, dedpB = fA_de(P_base), fB_de(P_base)
            termA = (w * dedpA / valA)
            termB = ((1.0-w) * dedpB / valB)
            
            # Theoretical Derivative
            dedp = (epsilon_base) * (termA + termB)
            
            # --- CAUSALITY CLAMP ---
            # If dEps/dP < 1, then v_sound > c. This violates causality.
            # We enforce the causal limit (stiffest possible matter: dP = dEps).
            if dedp < 1.0:
                dedp = 1.0
            
        else:
            # --- CRUST REGION (Cascade) ---
            if P_safe > P_c1:
                epsilon = crusts['c1'][0](P_safe); dedp = crusts['c1'][1](P_safe)
            elif P_safe > P_c2:
                epsilon = crusts['c2'][0](P_safe); dedp = crusts['c2'][1](P_safe)
            elif P_safe > P_c3:
                epsilon = crusts['c3'][0](P_safe); dedp = crusts['c3'][1](P_safe)
            else:
                epsilon = crusts['c4'][0](P_safe); dedp = crusts['c4'][1](P_safe)

    # Convert dEps/dP -> Sound Speed Squared
    if dedp <= 0: return [0, 0, 0]
    cs2_local = 1.0 / dedp 
    
    # --- GLOBAL PHYSICS CLAMPS ---
    # Enforce Causality (v <= c)
    if cs2_local > 1.0: cs2_local = 1.0 
    # Enforce Stability (v > 0)
    if cs2_local < 1e-5: cs2_local = 1e-5

    # Terminate if density is non-physical or radius is extremely small (singularity)
    if r < 1e-4 or epsilon <= 0: return [0, 0, 0]

    # ==========================================
    # 2. MACROPHYSICS (General Relativity)
    # ==========================================
    
    # --- A. TOV EQUATIONS (Structure) ---
    term_1 = (epsilon + P)
    # G_CONV includes 4*pi*G/c^4 factors
    term_2 = (m + (r**3 * P * CONSTANTS['G_CONV']))
    term_3 = r * (r - 2.0 * m * CONSTANTS['A_CONV'])
    
    if abs(term_3) < 1e-5: return [0, 0, 0]

    dP_dr = -CONSTANTS['A_CONV'] * (term_1 * term_2) / term_3
    dm_dr = (r**2) * epsilon * CONSTANTS['G_CONV']

    # --- B. RICCATI EQUATION (Tidal) ---
    # exp_lambda is the metric component e^lambda
    exp_lambda = 1.0 / (1.0 - 2.0 * CONSTANTS['A_CONV'] * m / r)
    
    # Q term involves speed of sound (cs2_local)
    Q = CONSTANTS['A_CONV'] * CONSTANTS['G_CONV'] * (5.0*epsilon + 9.0*P + (epsilon+P)/cs2_local) * (r**2)
    Q -= 6.0 * exp_lambda
    
    F = (1.0 - CONSTANTS['A_CONV'] * CONSTANTS['G_CONV'] * (r**2) * (epsilon - P)) * exp_lambda
    
    dy_dr = -(y_tidal**2 + y_tidal * F + Q) / r

    return [dm_dr, dP_dr, dy_dr]