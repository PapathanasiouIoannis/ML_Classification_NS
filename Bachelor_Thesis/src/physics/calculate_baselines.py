# ==============================================================================
# HEADER: src/physics/calculate_baselines.py
# ==============================================================================
# Description:
#   Calculates the maximum mass for each unperturbed (pure) hadronic model
#   in the library.
#
#   Significance:
#   These "baseline" maximum masses are required by the mixing workers to 
#   apply homologous scaling (alpha) correctly. They define the reference 
#   mass scale for the "Smart Inverse Sampling" technique.
#
# Cleaned & Refactored:
#   - Integrated `src.const.CONSTANTS` for standard transition pressures.
#   - Removed hardcoded magic numbers.
#   - Standardized output formatting.
# ==============================================================================

from src.const import CONSTANTS
from src.physics.get_eos_library import get_eos_library
from src.physics.solve_sequence import solve_sequence

def calculate_baselines():
    """
    Calculates the maximum mass for each pure hadronic model.
    
    Returns:
    - baselines: Dictionary mapping {ModelName: MaxMass_Msun}
    """
    print("--- Phase 0: Calculating Hadronic Baselines ---")
    
    # Retrieve the library of analytic core models and crust functions
    core_lib, crust_funcs = get_eos_library()
    baselines = {}
    
    # Iterate through each 'parent' Hadronic EoS model
    for name in core_lib.keys():
        # Get the symbolic functions (energy and derivative)
        f = core_lib[name] 
        
        # Determine the specific Crust-Core transition pressure.
        # "PS" model requires a higher transition pressure than the SLy default.
        p_trans = CONSTANTS['P_TRANS_PS'] if name == "PS" else CONSTANTS['P_TRANS_DEFAULT']
        
        # Prepare the input tuple for the TOV solver.
        # Format: (fA, dfA, fB, dfB, mixing_weight, crusts, scaling_alpha, p_trans)
        # For a pure model: w=1.0 (no mixing), alpha=1.0 (no scaling).
        eos_input = (f[0], f[1], f[0], f[1], 1.0, crust_funcs, 1.0, p_trans)
        
        # Solve the TOV sequence to find the maximum stable mass
        _, max_m = solve_sequence(eos_input, is_quark=False)
        
        # Store valid results
        if max_m > 1.0: 
            baselines[name] = max_m
            print(f"  > {name:<10}: {max_m:.3f} M_sun")
            
    return baselines