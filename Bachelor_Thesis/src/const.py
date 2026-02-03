# src/const.py

CONSTANTS = {
    # ==========================================
    # 1. FUNDAMENTAL PHYSICS & UNITS
    # ==========================================
    'HC': 197.33,
    'M_N': 939.0,
    'G_CONV': 1.124e-5,
    'A_CONV': 1.4766,

    # ==========================================
    # 2. CRUST PHYSICS (SLy Model)
    # ==========================================
    'P_C1': 9.34375e-5,   
    'P_C2': 4.1725e-8,    
    'P_C3': 1.44875e-11,  
    'P_TRANS_DEFAULT': 0.184,
    'P_TRANS_PS': 0.696,  

    # ==========================================
    # 3. NUMERICAL SOLVER SETTINGS
    # ==========================================
    'TOV_R_MIN': 1e-4,    
    'TOV_R_MAX': 20.0,    
    'TOV_P_MIN_SAFE': 1e-12, 
    
    # ==========================================
    # 4. QUARK MODEL PARAMETERS (CFL)
    # ==========================================
    'Q_B_MIN': 57.0,      
    'Q_B_ABS_MAX': 400.0, 
    
    # Kept strict lower limit as per calculation
    'Q_DELTA_RANGE': (57.0, 250.0),
    
    # Widened range to allow Ms to compete with Delta
    'Q_MS_RANGE': (80.0, 120.0), 
    
    # ==========================================
    # 5. GENERATION CONSTRAINTS
    # ==========================================
    # Quark
    'Q_M_MIN': 0.0001,       
    'Q_R_MAX': 22.0,      
    'Q_M_MAX_UPPER': 4.0, 
    
    # Hadronic
    'H_M_MIN_TARGET': 2.0,
    'H_M_MAX_TARGET': 3.0,
    'H_DELTA_LIMIT': 1.0, 
    
    # Causality Violation Threshold [MeV/fm^3]
    # Set to 600.0 based on test ran on HLPS-3 (614) and APR-1 (819).
    'CAUSALITY_EPS_LIMIT': 600.0, 

    # ==========================================
    # 6. PLOTTING STANDARDS
    # ==========================================
    'PLOT_R_LIM': (5.0, 20.0),
    'PLOT_M_LIM': (0.0, 4.0),
    'PLOT_L_LIM': (0.0, 5.0),       
    'PLOT_EPS_LIM': (0, 2500),      
    'PLOT_EPS_LOG': (1e2, 5e3),     
    'PLOT_CS2_LIM': (0, 1.05),
    'PLOT_SLOPE_LIM': (-8, 6),
    'BUCHDAHL_FACTOR': 2.25,

    # ==========================================
    # 7. DATA SCHEMA
    # ==========================================
    'COLUMN_SCHEMA': [
        'Mass', 'Radius', 'Lambda', 'Label', 'Curve_ID', 
        'P_Central', 'Eps_Central', 'Eps_Surface', 'CS2_Central', 
        'CS2_at_14', 'Radius_14',       
        'Slope14', 'Slope16', 'Slope18', 'Slope20', 
        'Bag_B', 'Gap_Delta', 'Mass_Strange' 
    ]
}