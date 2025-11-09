# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
###########################
#------Turbie Module-------
# Contains functions to simulate a 2-DOF wind turbine system
#############################

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# *******************************
# Turbie parameters loader
# *******************************
def load_turbie_parameters(file_path):
    """
    Reads turbie_parameters.txt and returns system parameters
    """
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=1)
    # assign values manually as per the text file
    params = {}
    params['mb'] = df.iloc[0,0]
    params['mn'] = df.iloc[1,0]
    params['mh'] = df.iloc[2,0]
    params['mt'] = df.iloc[3,0]
    params['c1'] = df.iloc[4,0]
    params['c2'] = df.iloc[5,0]
    params['k1'] = df.iloc[6,0]
    params['k2'] = df.iloc[7,0]
    params['fb'] = df.iloc[8,0]
    params['ft'] = df.iloc[9,0]
    params['drb'] = df.iloc[10,0]
    params['drt'] = df.iloc[11,0]
    params['Dia'] = df.iloc[12,0]
    params['rho'] = df.iloc[13,0]

    # derived parameters, m1 = total mass of blades, m2 combined mass of hub, nacelle and tower, 
    # A rotor area
    params['m1'] = params['mb'] * 3
    params['m2'] = params['mh'] + params['mn'] + params['mt']
    params['A'] = np.pi * (params['Dia']/2)**2

    return params


# *****************************
# Wind data loader
# *****************************
def load_wind_data(root_dir):
    """
    Loads all wind files in the root directory.
    Returns a nested dict: {TI_folder: {wind_filename: dataframe}}
    """
    import os, glob
    TI_folders = sorted([d for d in os.listdir(root_dir) if d.startswith('wind_TI')])
    wind_data = {}
    for TI in TI_folders:
        wind_data[TI] = {}
        TI_path = os.path.join(root_dir, TI)
        TI_value = TI.split('_')[-1]
        files = sorted(glob.glob(os.path.join(TI_path, f"wind_*_ms_TI_{TI_value}.txt")))
        for f in files:
            fname = os.path.basename(f).replace('.txt','')
            df = pd.read_csv(f, delim_whitespace=True, header=0)  # Time(s) V(m/s)
            df.columns = ['Time(s)', 'u(m/s)']
            wind_data[TI][fname] = df
    return wind_data

# *****************************************
# Ct loader and Ct value for each simulation
# *****************************************
def load_Ct(file_path):
    """
    Load Ct vs wind speed table
    """
    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, names=['U','Ct'])
    Ct_interp = interp1d(df['U'], df['Ct'], kind='linear', fill_value='extrapolate')
    return Ct_interp

def get_Ct_for_wind(U_series, Ct_interp):
    """
    Obtain Ct value for a mean wind speed for a simulation
    """
    U_mean = U_series['u(m/s)'].mean()
    return float(Ct_interp(U_mean))

# *****************
# Dynamic System matrices
# ****************
def build_matrices(params):
    """
    create mass, damping and stiffnes matrices from turbie parameters
    """
    m1, m2 = params['m1'], params['m2']
    k1, k2 = params['k1'], params['k2']
    c1, c2 = params['c1'], params['c2']

    #mass matrix
    M = np.array([[m1, 0],
                  [0, m2]])
    
    # damping matrix
    C = np.array([[c1, -c1],
                  [-c1, c1+c2]])
    
    # stffness matrix
    K = np.array([[k1, -k1],
                  [-k1, k1+k2]])
    return M, C, K

# *****************
# Aerodynamic force on the blades
# *****************
def aero_force(U_t, x1_dot, Ct, rho, A):
    """
    Aerodynamic force on the blades as a function of the relative wind speed
    
    parameters :
        U_t - wind speed [m/s]
        x1_dot - blade velocity [m/s]
        Ct - thrust coefficient
        rho - air density [kg/m3]
        A - rotor area [m2]
    
    """
    
    return 0.5 * rho * A * Ct * (U_t - x1_dot) * abs(U_t - x1_dot)

# ******************
# Dynamic system equation function
#*******************
def two_dof_state_space(t, y, U_TS, Ct, M, C, K, rho, A):
    """
    Defining the equation of motion which will be used for solving the initial
    value probelm for each simulation
    
    parameters :
        t - time [s]
        y - state vector 
        U_TS - wind time  t, u        
        Ct - thrust coefficient
        M - mass matrix
        C - damping matrix 
        K - stiffness matrix
        rho - air density [kg/m3]
        A - rotor area [m2]
    
    """  
    
    x1, x2, x1_dot, x2_dot = y
    U_t = np.interp(t, U_TS['Time(s)'], U_TS['u(m/s)'])
    F = np.array([aero_force(U_t, x1_dot, Ct, rho, A), 0.0]).reshape(-1,1)
    Z = np.zeros((2,2))
    I = np.eye(2)
    A_mat = np.block([[Z, I],
                      [-np.linalg.inv(M) @ K, -np.linalg.inv(M) @ C]])
    B = np.vstack([np.zeros((2,1)), np.linalg.inv(M) @ F])
    y_dot = A_mat @ y.reshape(-1,1) + B
    return y_dot.flatten()
