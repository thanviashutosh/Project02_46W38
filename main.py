# -*- coding: utf-8 -*-
"""
Main script for running the dynamic system simulation. 
A total of 3 X 22 simulations i.e. 3 TI values (0.05, 0.10 and 0.15) and for
each TI value 22 wind time series for a mean speed of 4m/s to 25m/s

"""

import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import glob
import matplotlib.pyplot as plt

# import the turbie module
from turbie_mod import load_wind_data, load_Ct, two_dof_state_space, get_Ct_for_wind
from turbie_mod import load_turbie_parameters,build_matrices

#defining paths for wind data and CT files and turbie input parameters
wind_data_path = "C://Data//46W38-2025-Projects//Project_02//wind_files"
Ct_file = "C://Data//46W38-2025-Projects//Project_02//turbie_inputs//CT.txt"
turbie_inputs = "C://Data//46W38-2025-Projects//Project_02//turbie_inputs//turbie_parameters.txt"


# load wind data, Ct and Turbie parameters from the text files
wind_data = load_wind_data(wind_data_path)
Ct_interp = load_Ct(Ct_file)
params    = load_turbie_parameters(turbie_inputs)

# system matrices - Mass , Damping and Stiffness matrices

M,C,K = build_matrices(params)

# Area of the rotor 
A = params['A']

# air density
rho = params['rho']

## TI folders for running simulations
TI_folders = sorted([ d for d in os.listdir(wind_data_path) if d.startswith('wind_TI')])

## defining initial condition , zero initial displacement and velocity
y0 = [0,0,0,0] 

t_span = (0,600) # initial and final t in the simulation
t_eval = np.linspace(0,600,1000) # evaluation points

## output directory
os.makedirs("results",exist_ok=True)

## Simulation results storage

sim_results= {}
disp_stats ={}

for TI_folder, files in wind_data.items(): 

    sim_results[TI_folder] = {} 
    stats_list =[]

    for file_name, U_TS in files.items():
        Ct = get_Ct_for_wind(U_TS, Ct_interp)
        sol_sim = solve_ivp(two_dof_state_space, t_span, y0, t_eval=t_eval, args=(U_TS,Ct,M,C,K,rho,A))
        sim_results[TI_folder][file_name] = sol_sim
        
        # calculate displacements rounded to 3 decimals
        blade_disp= np.round(sol_sim.y[0]-sol_sim.y[1],3)
        tower_disp= np.round(sol_sim.y[1],3)
        
        # save results for this case
        results_df = pd.DataFrame({'Time(s)':np.round(sol_sim.t,3),
                                   'Relative Blade_disp(m)':blade_disp,
                                   'Tower_disp(m)':tower_disp})
        result_path = f"results/{TI_folder}_{file_name}_displacement.txt"
        results_df.to_csv(result_path, index=False, sep='\t')
        print(f"Have Patience, Completed:{TI_folder}-{file_name}")
        
        #*****************
        # compute mean and standard deviation of the blade and tower displacements
        #**********************
        mean_blade_disp = np.mean(blade_disp)
        std_blade_disp = np.std(blade_disp)
        mean_tower_disp = np.mean(tower_disp)
        std_tower_disp = np.std(tower_disp)
        
        #******************
        #extract mean wind speed from file names
        #******************
        U_mean = float(file_name.split('_')[1])
        
        #append the required values in the stats_list
        stats_list.append([U_mean, mean_blade_disp, std_blade_disp, mean_tower_disp, std_tower_disp])
        
        ## Simple comment :)
        print(f"Have Patience, Completed:{TI_folder}-{file_name}")
        
        #save all the statistics for this TI
        summary_df = pd.DataFrame(stats_list, columns= ['Mean Speed', 'Mean Blade Disp',
                                                       'Std_Dev Blade Disp', 'Mean Tower Disp',
                                                       'Std_Dev Tower Disp'])
        summary_df =summary_df.sort_values('Mean Speed')
        summary_df.to_csv(f"results/{TI_folder}_stats_summary.txt", sep='\t', index = False, float_format='%.3f')
        
        disp_stats[TI_folder] =summary_df
    

# *************************
# Sample plot as requested - time marching wind speed with blade and tower displacements
# *************************
first_TI = TI_folders[0]
first_file = list(sim_results[first_TI].keys())[0]
U_TS = wind_data[first_TI][first_file]
sol_example = sim_results[first_TI][first_file]

plt.figure(figsize=(10,5))
plt.plot(U_TS['Time(s)'],U_TS['u(m/s)'], 'k--', label='Wind speed(m/s)', alpha=0.5)
plt.plot(sol_example.t, sol_example.y[0]-sol_example.y[1], label='Relative Deflection_Blade (x1-x2)')
plt.plot(sol_example.t, sol_example.y[1], label='Hub+Nacelle+Tower (x2)')
plt.xlabel('Time [s]')
plt.ylabel('Displacement [m]')
plt.title(f"Time-marching Response :{first_TI} - {first_file}")
plt.legend()
plt.grid(True)
plt.show()

#****************************
# plotting mean and standard deviation vs wind speed for each TI
#******************************

plt.figure(figsize=(10,5))

for TI_folder, df in disp_stats.items():
    plt.plot(df['Mean Speed'], df['Mean Blade Disp'],'o-', label=f'{TI_folder}-Blade Disp Mean')
    plt.plot(df['Mean Speed'], df['Mean Tower Disp'], 's--',label=f'{TI_folder}-Tower Disp Mean')

plt.xlabel('Mean Wind Speed (m/s)')
plt.ylabel('Mean Displacements (m)')
plt.title('Mean Displacements vs Wind speed')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))

for TI_folder, df in disp_stats.items():
    plt.plot(df['Mean Speed'], df['Std_Dev Blade Disp'],'o-', label=f'{TI_folder}-Blade Disp Std_Dev')
    plt.plot(df['Mean Speed'], df['Std_Dev Tower Disp'],'s--',label=f'{TI_folder}-Tower Disp Std_Dev')

plt.xlabel('Mean Wind Speed (m/s)')
plt.ylabel('Std_Dev Displacement (m)')
plt.title('Std_Dev  Displacement vs Wind speed')
plt.legend()
plt.grid(True)
plt.show()



