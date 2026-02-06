

import numpy as np
from scipy.integrate import solve_ivp
from src.const import CONSTANTS
from src.physics.tov_rhs import tov_rhs

def solve_sequence(eos_input, is_quark):
    """
    Integrates TOV for a sequence of pressures.
    
    Parameters:
    - eos_input: Tuple of EoS parameters (model specific).
    - is_quark: Boolean flag for Quark (CFL) vs Hadronic models.
    
    Returns:
    - curve_data: List of [Mass, Radius, Lambda, Pc, Eps_c, Cs2_c, Eps_surf]
    - max_m: The maximum mass found in this sequence.
    """
    r_min = CONSTANTS['TOV_R_MIN']
    
    # Define pressure grid (log-spaced)

    n_points = 100
    if is_quark:
        pressures = np.logspace(0.0, 4.0, n_points) 
    else:
        pressures = np.logspace(-3, 3.8, n_points)
    
    curve_data = []
    max_m = 0.0
    
    # Trackers for Hadronic Causality Clamp (ensuring dP/dEps <= 1)
    last_pc_calc = 0.0
    last_eps_calc = 0.0

    for pc in pressures:

        # 1. INITIALIZATION (Finding Epsilon_c from P_c)

        if is_quark:
            # --- GENERALIZED CFL INITIALIZATION ---
            B, Delta_geom, ms_geom = eos_input
            hc = CONSTANTS['HC']
            
            # Convert to Geometric Units [fm^-4]
            pc_geom = pc / hc
            B_geom = B / hc
            
            # Coefficients for the quadratic equation in mu^2
            # P = 3/(4pi^2)*mu^4 + 3/pi^2*(Delta^2 - ms^2/4)*mu^2 - B
            coeff_a = 3.0 / (4.0 * np.pi**2)
            eff_gap_term = Delta_geom**2 - (ms_geom**2 / 4.0)
            coeff_b = 3.0 * eff_gap_term / (np.pi**2)
            coeff_c = -(pc_geom + B_geom)
            
            det = coeff_b**2 - 4*coeff_a*coeff_c
            if det < 0: continue 
            
            mu2 = (-coeff_b + np.sqrt(det)) / (2*coeff_a)
            eps_geom = 3.0 * coeff_a * (mu2**2) + coeff_b * mu2 + B_geom
            eps_init = eps_geom * hc
            
            # Sound Speed Calculation
            term_shift = 2.0 * eff_gap_term
            num = mu2 + term_shift
            den = 3.0 * mu2 + term_shift
            
            if abs(den) < 1e-10: 
                cs2_init = 0.33
            else: 
                cs2_init = num / den

        else:
            # --- HADRONIC INITIALIZATION ---
            fA_e, fA_de, fB_e, fB_de, w, crusts, alpha, P_trans = eos_input
            
            # Crust Transition Thresholds
            P_c1 = CONSTANTS['P_C1']
            P_c2 = CONSTANTS['P_C2']
            P_c3 = CONSTANTS['P_C3']
            
            if pc > P_trans:
                # Core Region: Apply Homologous Scaling
                p_base = pc / alpha
                vA = fA_e(p_base)
                vB = fB_e(p_base)
                
                if vA <= 0 or vB <= 0: continue
                
                eps_raw = ((vA**w) * (vB**(1.0-w))) * alpha
                
                # Causality Clamp check (Slope <= 1)
                # If dEps/dP < 1, causality is violated.
                if last_eps_calc > 0:
                    d_p = pc - last_pc_calc
                    d_e_raw = eps_raw - last_eps_calc
                    if d_e_raw < d_p: 
                        eps_init = last_eps_calc + d_p
                    else: 
                        eps_init = eps_raw
                else: 
                    eps_init = eps_raw
                
                # Derivative Mixing
                dedpA, dedpB = fA_de(p_base), fB_de(p_base)
                termA = (w * dedpA / vA)
                termB = ((1.0-w) * dedpB / vB)
                dedp = (eps_init / alpha) * (termA + termB)
                cs2_init = 1.0 / dedp if dedp > 0 else 0
                
            else:
                # Crust Region
                if pc > P_c1: 
                    eps_init = crusts['c1'][0](pc); dedp = crusts['c1'][1](pc)
                elif pc > P_c2: 
                    eps_init = crusts['c2'][0](pc); dedp = crusts['c2'][1](pc)
                elif pc > P_c3: 
                    eps_init = crusts['c3'][0](pc); dedp = crusts['c3'][1](pc)
                else: 
                    eps_init = crusts['c4'][0](pc); dedp = crusts['c4'][1](pc)
                
                cs2_init = 1.0 / dedp if dedp > 0 else 0
            
            last_pc_calc = pc
            last_eps_calc = eps_init
        
        # Physical Clamps for Speed of Sound
        if cs2_init > 1.0: cs2_init = 1.0
        if cs2_init < 1e-5: cs2_init = 1e-5


        # 2. SURFACE DENSITY CALCULATION (Quark Only)

        if is_quark:
            B, Delta_geom, ms_geom = eos_input
            hc = CONSTANTS['HC']
            B_geom = B / hc
            
            # Solve P=0 for Surface Density
            # At surface, P=0 implies coeff_c = -B_geom
            coeff_a = 3.0 / (4.0 * np.pi**2)
            eff_gap_term = Delta_geom**2 - (ms_geom**2 / 4.0)
            coeff_b = 3.0 * eff_gap_term / (np.pi**2)
            coeff_c_surf = -B_geom 
            
            det_surf = coeff_b**2 - 4*coeff_a*coeff_c_surf
            mu2_surf = (-coeff_b + np.sqrt(max(0, det_surf))) / (2*coeff_a)
            eps_geom_surf = 3.0 * coeff_a * (mu2_surf**2) + coeff_b * mu2_surf + B_geom
            eps_surf = eps_geom_surf * hc
        else:
            eps_surf = 0.0


        # 3. INTEGRATION (TOV Solver)
        # Initial Mass (Approximation for small r_min)
        m_init = (r_min**3) * eps_init * (CONSTANTS['G_CONV'] / 3.0)
        
        # State Vector: [Mass, Pressure, y_tidal]
        y0 = [m_init, pc, 2.0]

        # Event to detect surface (Pressure -> 0)
        def surface_event(t, y): return y[1]
        surface_event.terminal = True; surface_event.direction = -1

        try:
            sol = solve_ivp(fun=lambda r, y: tov_rhs(r, y, eos_input, is_quark), 
                            t_span=(r_min, CONSTANTS['TOV_R_MAX']), 
                            y0=y0, events=surface_event, 
                            method='RK45', rtol=1e-9, atol=1e-12)

            if sol.status == 1 and len(sol.t_events[0]) > 0:
                R = sol.t_events[0][0]
                M = sol.y_events[0][0][0]
                yR = sol.y_events[0][0][2]
                
                # Filter unphysical results
                if R < 3.0 or M < 0.1: continue
                
                # Calculate Tidal Deformability (Lambda)
                C = (M * CONSTANTS['A_CONV']) / R
                if C >= 0.5: continue # Black hole limit

                # Complex Tidal Love Number formula (Hinderer et al.)
                num = (8/5)*(1-2*C)**2 * C**5 * (2*C*(yR-1) - yR + 2)
                den = 2*C*(6-3*yR+3*C*(5*yR-8)) + 4*C**3*(13-11*yR+C*(3*yR-2)+2*C**2*(1+yR)) + 3*(1-2*C)**2*(2-yR+2*C*(yR-1))*np.log(1-2*C)
                
                if abs(den) < 1e-10: continue
                k2 = num/den
                Lam = (2/3)*k2*(C**-5)

                if M <= 0.0: break
                
                # Check for maximum mass peak (Stability limit)
                if M < max_m: break 
                if M > max_m: max_m = M
                
                curve_data.append([M, R, Lam, pc, eps_init, cs2_init, eps_surf])
        except Exception:
            continue
        
    return curve_data, max_m
