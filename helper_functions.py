"""
 -*- coding: utf-8 -*-
This file contains all functions which are necessary for the project.
@Author : nkpanda
@FileName: helper_functions.py
@Git ：https://github.com/nkpanda97
"""
#%%
import numpy as np
import pandas as pd
import config
import matplotlib.pyplot as plt
import datetime as dt
# Supress all warnings
import warnings
warnings.filterwarnings("ignore")
import pyomo.environ as pyo
from itertools import combinations
import time
import tqdm.notebook as tq

# %%
'''
test_data = pd.DataFrame(data={'P_MAX': [20,10], 'P_MIN':[0,5], 'E_MAX':[25, 30], 'E_MIN':[15,20]})
model_direct = build_model_direct(test_data, [0,3], 1)
A_poly, b_poly = generate_system_matrix(test_data, [0,3], 1)
model_ul = build_model_ul(A_poly, b_poly, [0,3])
solver = pyo.SolverFactory('gurobi')
solver.options['DualReductions'] = 0
solver.solve(model_direct, tee=True)
solver.solve(model_ul, tee=False)
print(model_direct.obj())
print(model_ul.obj())
'''

def rescale_ev_data(ev_data, date_sample, del_t, restrict_stop_time=False, total_time_steps=36, verbose_=1000):
    ev_data = ev_data_all[ev_data_all['START'].dt.date == date_sample.date()]

    if verbose_ >=500:
        print(f'There are a total of {len(ev_data)} transactions on {date_sample.date()}')

    max_stop = ev_data['STOP'].max() # Farthest departure time of EV 

    # Make a datetime range with delta= del_t for the max_dur, which is the time axis for optimization
    if restrict_stop_time:
        time_range = total_time_steps
    else:
        time_range = np.ceil((max_stop-ev_data['START'].min().normalize()).total_seconds()/3600) # in integer timestep

    if verbose_>10:
        print(f'There are a total of {(time_range)} time steps of {del_t} hours each in the optimization horizon')

    # Changing START time and STOP time into the new integer timesteps of del_t timestep

    strt_int = np.floor((ev_data['START']-ev_data['START'].min().normalize()).dt.total_seconds()/(del_t*3600))  # Round down start time
    stop_int = np.ceil((ev_data['STOP']-ev_data['START'].min().normalize()).dt.total_seconds()/(del_t*3600))  # Round up stop time

    if restrict_stop_time:
        ev_data['STOP_int_1H_non_restrict'] = stop_int
        stop_int = np.minimum(stop_int, total_time_steps) # Restricting the stop time to the last time step

    ev_data[f'START_int_{del_t}H'] = strt_int
    ev_data[f'STOP_int_{del_t}H'] = stop_int

    # Dropping infeasible transactions based on (stop_int - strt_int)*Pmax < Vol
    idx_to_drop = ev_data[( ev_data[f'STOP_int_{del_t}H'].to_numpy()- ev_data[f'START_int_{del_t}H'].to_numpy())*ev_data['P_MAX']*del_t<ev_data['VOL']].index
    
    if len(idx_to_drop)>0:
        ev_data.drop(idx_to_drop, inplace=True)
        if verbose_>10:
            print(f'Dropped {len(idx_to_drop)} infeasible transactions')
            

    # The df where rows are time steps and columns are transactions. With each cell==1 if 
    # the EV is connected at that time step else 0
    connectivity_df = pd.DataFrame(index=np.arange((time_range)), columns=ev_data.index)  

    def fill_connectivity_df(col, ev_df_, time_range):
        '''
            This function fills the connectivity_df for a given transaction
            For a given col the values are 1 for the indexs which is between the START_int_{del_t}H and STOP_int_{del_t}H else 0.
        '''
        res_arr = np.zeros(int(time_range))
        res_arr[int(ev_df_.loc[col, f'START_int_{del_t}H']):int(ev_df_.loc[col, f'STOP_int_{del_t}H'])] = 1
        return res_arr
    

    for col in connectivity_df.columns:

        connectivity_df.loc[:,col] = fill_connectivity_df(col, ev_data, time_range)

    connectivity_df['Total_connectivity'] = connectivity_df.sum(axis=1)

    return connectivity_df, ev_data

def process_for_opt(ev_df, f_windw, del_t, verbose_=1000):
    '''
        This function takes the ev transactions after suitable time scaling
            and the flexibilty window and returns the df_for_opt which is used for optimization

        param ev_df: The dataframe of ev transactions after suitable time scaling
        param f_windw: The flexibility window
        param del_t: The time step for optimization
        param verbose_: The verbosity level

        return df_for_opt: The dataframe which is used for optimization with columns: E_MAX, E_MIN, P_MAX, P_MIN
    '''
    t_span = del_t*(f_windw[1]-f_windw[0]) # in hours
    ev_df_filtered = ev_df[(ev_df[f'START_int_{del_t}H']<=f_windw[0])& (ev_df[f'STOP_int_{del_t}H']>f_windw[1])].copy()

    if verbose_>=1000:
        print(f'There are a total of {len(ev_df_filtered)} transactions fully connected in the time window {f_windw[0]}-{f_windw[1]} out of {len(ev_df)} transactions')


    df_for_opt = pd.DataFrame(index=np.arange(len(ev_df_filtered)))

    df_for_opt['P_MAX'] = ev_df_filtered['P_MAX'].values
    df_for_opt['P_MIN'] = 0
    df_for_opt['E_MAX'] = np.minimum(ev_df_filtered['VOL'].to_numpy(),ev_df_filtered['P_MAX'].to_numpy()*t_span*del_t)  # min(P_max*t_span*del_t, Vol)

    energy_max_out_of_fwindow = ev_df_filtered['P_MAX'].to_numpy()*(-t_span*np.ones(len(ev_df_filtered))+(ev_df_filtered[f'STOP_int_{del_t}H'].to_numpy()-ev_df_filtered[f'START_int_{del_t}H'].to_numpy()))*del_t # P_max*(-t_span+stop_int - start_int)*del_t
    remaining_energy = ev_df_filtered['VOL'].to_numpy() - energy_max_out_of_fwindow # Vol - P_max*(-t_span+stop_int - start_int)*del_t
    df_for_opt['E_MIN'] = np.maximum(np.zeros(len(ev_df_filtered)), remaining_energy) # Vol_min = stop_int - start_int

    return df_for_opt

def build_model_direct(df_for_opt, f_windw, del_t):
    # Pyomo model

    model = pyo.ConcreteModel()

    # Sets
    T = int((f_windw[1]-f_windw[0])/del_t) # Total time steps
    model.T = pyo.Set(initialize=np.arange(T), doc='Set of time-steps') # Time steps
    model.N = pyo.Set(initialize=np.arange(len(df_for_opt)), doc='Set of EVs') # EV transactions
    model.T_1 = pyo.Set(initialize=np.arange(T+1), doc='Set of T+1 time steps ') # Time steps except the first one

    # Parameters
    model.p_max = pyo.Param(model.N, initialize={i :df_for_opt['P_MAX'].iloc[i] for i in range(len(df_for_opt))}, doc='Maximum charging power of EVs')
    model.p_min = pyo.Param(model.N, initialize={i :df_for_opt['P_MIN'].iloc[i] for i in range(len(df_for_opt))}, doc='Minimum charging power of EVs')
    model.e_max = pyo.Param(model.N, initialize={i :df_for_opt['E_MAX'].iloc[i] for i in range(len(df_for_opt))}, doc='Maximum charging energy of EVs')
    model.e_min = pyo.Param(model.N, initialize={i :df_for_opt['E_MIN'].iloc[i] for i in range(len(df_for_opt))}, doc='Minimum charging energy of EVs')
    model.del_t = pyo.Param(initialize=del_t, doc='Time step for optimization')

    # Variables

    model.p = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals, doc='Charging power of EVs')
    model.e = pyo.Var(model.N, model.T_1, within=pyo.NonNegativeReals, doc='State of energy of EVs (Always starts from 0 and has to end between [e_min, e_max])')
    model.p_agg = pyo.Var(model.T, within=pyo.NonNegativeReals, doc='Aggregated charging power of EVs')
    model.sigma = pyo.Var(within=pyo.NonNegativeReals, doc='Auxillary variable for aggregated charging power of EVs')


    # Constraints

    #Constraint for power
    def power_constraint_rule(model, n, t):
        return model.p_min[n], model.p[n,t], model.p_max[n]
    model.power_constraint = pyo.Constraint(model.N, model.T, rule=power_constraint_rule, doc='Power constraint of EVs')

    #Constraint on final energy
    def final_energy_constraint_rule(model, n):
        return model.e_min[n], model.e[n,T], model.e_max[n]
    model.final_energy_constraint = pyo.Constraint(model.N, rule=final_energy_constraint_rule, doc='Final energy constraint of EVs')

    #Constraint on energy and power
    def energy_power_constraint_rule(model, n, t):
        if t==0:
            return model.e[n,t] == 0
        else:
            return model.e[n,t] == model.e[n,t-1] + model.p[n,t-1]*model.del_t
    model.energy_power_constraint = pyo.Constraint(model.N, model.T_1, rule=energy_power_constraint_rule, doc='Energy and power constraint of EVs')
        

    #Constraint on aggregated power
    model.aggregated_power_constraint = pyo.Constraint(model.T, rule=lambda model, t: model.p_agg[t] == sum(model.p[n,t] for n in model.N), doc='Aggregated power constraint of EVs')

    # Constraint on aggregated energy
    model.constraint_cap_lim = pyo.Constraint(model.T, rule=lambda model, t: model.p_agg[t] >= model.sigma, doc='Aggregated power constraint of EVs')
    # Objective function (For capacity limitation)

    model.obj = pyo.Objective(expr=model.sigma, sense=pyo.maximize, doc='Objective function')

    return model

def calculate_u_l(p_min, p_max, e_min, e_max, del_t, t_steps):
    u = [0]
    l = [0]
    for n in range(1, t_steps+1):
        u.append(min(n*del_t*p_max, e_max-(t_steps-n)*del_t*p_min))
        l.append(max(n*del_t*p_min, e_min-(t_steps-n)*del_t*p_max))
    return u, l

def calculate_aggregate_ul(ev_data, f_windw, del_t):

    ul_dict = {}

    for rowx, one_ev_data in ev_data.iterrows():
        u,l = calculate_u_l(one_ev_data.P_MIN, one_ev_data.P_MAX, one_ev_data.E_MIN, one_ev_data.E_MAX, del_t, int((f_windw[1]-f_windw[0])/del_t))
        ul_dict[f'EV#{rowx+1} u']=u
        ul_dict[f'EV#{rowx+1} l'] = l
    ul_df = pd.DataFrame(ul_dict)
    # Add aggregate u and l, where Aggregate u = EV#1 u + EV#2 u + ... and Aggregate l = EV#1 l + EV#2 l + ...
    ul_df['Aggregate u'] = ul_df[[col for col in ul_df.columns if 'u' in col]].sum(axis=1)
    ul_df['Aggregate l'] = ul_df[[col for col in ul_df.columns if 'l' in col]].sum(axis=1)
    return ul_df

def generate_combinations_matrix(n):
    matrix_total = []
    for i in range(1, n+1):
        combinations_list = list(combinations(range(n), i))
        matrix = np.zeros((len(combinations_list), n), dtype=int)

        for row, comb in enumerate(combinations_list):
            for element in comb:
                matrix[row, element] = 1

        matrix_total.append(matrix)

    return np.vstack(matrix_total)

def generate_polytope_b_matrix(T):
   mymatrix = []
   for col in range(1,1+T):
      c_m = []
      for row in range(1,1+T):
         if row == col:
            c_m.append(np.ones(shape=(len(list(combinations(range(T), row))),1), dtype=int))
         else:
            c_m.append(np.zeros(shape=(len(list(combinations(range(T), row))),1), dtype=int))

      single_col_matrix = np.vstack(c_m)
   
      mymatrix.append(single_col_matrix)

   return np.hstack(mymatrix)

def generate_system_matrix(ev_data, f_windw, del_t):
    ul_aggregate = calculate_aggregate_ul(ev_data, f_windw, del_t)
    u = ul_aggregate['Aggregate u'].values
    l = ul_aggregate['Aggregate l'].values
    T = int((f_windw[1]-f_windw[0])/del_t)
    A = generate_combinations_matrix(T)
    b = generate_polytope_b_matrix(T)
    A_poly = np.vstack([A, -A])
    b_poly = np.vstack([np.matmul(b,np.array(u[1:])).reshape((len(b),1)),-np.matmul(b,np.array(l[1:])).reshape((len(b),1))])
    return A_poly, b_poly

def build_model_ul(A,b, f_windw, del_t):


    model = pyo.ConcreteModel()
    # Sets

    model.T = pyo.Set(initialize=np.arange(int((f_windw[1]-f_windw[0])/del_t)), doc='Set of time-steps') # Time steps
    model.row = pyo.Set(initialize=np.arange(A.shape[0]), doc='Set of rows of A') # Rows of A
    # Parameters
    model.A = pyo.Param(model.row, model.T, mutable=True, doc='Matrix A')
    model.b = pyo.Param(model.row, mutable=True, doc='Vector b')

    for row in model.row:
        model.b[row] = b[row][0]
        for t in model.T:
            model.A[row,t] = A[row,t]
    # Variables

    model.p_agg = pyo.Var(model.T, within=pyo.NonNegativeReals, doc='Aggregated charging power of EVs')
    model.sigma = pyo.Var(within=pyo.NonNegativeReals, doc='Auxillary variable for aggregated charging power of EVs')


    # Constraints
    # Constraint for A[row,t]*p_agg[t]<=b[row]
    def ul_constraint_rule(model, row):
        return sum(model.A[row,t]*model.p_agg[t]*del_t for t in model.T) <= model.b[row]
    model.ul_constraint = pyo.Constraint(model.row, rule=ul_constraint_rule, doc='Ul (Ax<b) constraint of EVs')

    # Constraint on aggregated energy
    model.constraint_cap_lim = pyo.Constraint(model.T, rule=lambda model, t: model.p_agg[t] >= model.sigma, doc='Aggregated power constraint of EVs')
    # Objective function (For capacity limitation)

    model.obj = pyo.Objective(expr=model.sigma, sense=pyo.maximize, doc='Objective function')

    return model

def multiply_transactions(ev_data_all, n_times,date_sample,del_t, f_windw, verbose_=False , restrict_stop_time=False, total_time_steps=36):

    if n_times >1:
        list_df_for_opt = []
        for i in range(n_times):
            date_sample_new = date_sample + dt.timedelta(days=i)
            connectivity_df, scalled_ev_data1= rescale_ev_data(ev_data_all, date_sample_new, del_t,restrict_stop_time=restrict_stop_time, total_time_steps=total_time_steps,verbose_=verbose_)
            df_ = process_for_opt(scalled_ev_data1, f_windw, del_t, verbose_)
            list_df_for_opt.append(df_)

        data_for_opt = pd.concat(list_df_for_opt, ignore_index=True)
    else:
        connectivity_df, scalled_ev_data1= rescale_ev_data(ev_data_all, date_sample, del_t,restrict_stop_time=restrict_stop_time, total_time_steps=total_time_steps,verbose_=verbose_)
        data_for_opt = process_for_opt(scalled_ev_data1, f_windw, del_t, verbose_)

    return data_for_opt

def single_run(data_for_opt, f_windw, del_t, verbose_=False):
    tic_direct = time.time()
    model_direct = build_model_direct(data_for_opt, f_windw, del_t)
    toc_direct = time.time()
    bt_direct = toc_direct-tic_direct
    if verbose_:
        print('Time taken to build model using direct method: ', bt_direct)

    tic_ul = time.time()
    A_poly, b_poly = generate_system_matrix(data_for_opt, f_windw, del_t)
    model_ul = build_model_ul(A_poly, b_poly, f_windw, del_t)
    toc_ul = time.time()
    bt_ul = toc_ul-tic_ul
    if verbose_:
        print('Time taken to build model using ul method: ', bt_ul)

    solver = pyo.SolverFactory('gurobi')

    tic_solve_direct = time.time()
    res_direct = solver.solve(model_direct, tee=False)
    toc_solve_direct = time.time()
    st_direct = toc_solve_direct-tic_solve_direct
    if verbose_:
        print('Time taken to solve model using direct method: ', st_direct)

    tic_solve_ul = time.time()
    res_ul = solver.solve(model_ul, tee=False)   
    toc_solve_ul = time.time()
    st_ul = toc_solve_ul-tic_solve_ul
    if verbose_:
        print('Time taken to solve model using ul method: ', st_ul)

    if verbose_:
        print('Objective value for direct method: ', model_direct.obj())
        print('Objective value for ul method: ', model_ul.obj())
        print(f'Speed up in build time (%): {(bt_direct-bt_ul)/(bt_direct)*100}')
        print(f'Speed up in solve time (%): {(st_direct-st_ul)/(st_direct)*100}')
        print(f'Total speed up (%): {(bt_direct*st_direct-bt_ul*st_ul)/(bt_direct*st_direct)*100}')





    res_dict = {'No of EVs': len(data_for_opt),
                'No of time steps': int(f_windw[1] - f_windw[0]),
                'Time step size (H)': del_t,
                'Direct optimization': {'Build time (s)': bt_direct,
                                        'Solve time (s)': st_direct,
                                        'Objective value': model_direct.obj(),
                                        'info': res_direct},
                'UL optimization': {'Build time (s)': bt_ul,
                                    'Solve time (s)': st_ul,
                                    'Objective value': model_ul.obj(),
                                    'info': res_ul},
                'Speed up in build time (%)': (bt_direct-bt_ul)/(bt_direct)*100,
                'Speed up in solve time (%)': (st_direct-st_ul)/(st_direct)*100,
                'Total speed up (%)': (bt_direct*st_direct-bt_ul*st_ul)/(bt_direct*st_direct)*100}
    
    return res_dict

def log_model_info(model, file_name='log.txt'):
    with open(file_name, 'w') as output_file:
            model.pprint(output_file)
# %%

ev_data_all = pd.read_pickle(config.ev_data_all)
date_sample = dt.datetime(2022, 8, 1)
verbose_ = 0
f_windw = [18,21] # in hours

n_days = range(1,30)

re_list = []

for n_times in tq.tqdm(n_days, position=0,leave=True,desc='Days of transactions'):
    data_for_opt = multiply_transactions(ev_data_all, n_times,date_sample,del_t=1, f_windw=f_windw, verbose_=verbose_ , restrict_stop_time=False, total_time_steps=36)
    for ts in tq.tqdm([1, 0.75, 0.5, 0.25], position=0,leave=True, desc='Time step size'):
        res_dict = single_run(data_for_opt, f_windw, del_t=ts, verbose_=verbose_)
        re_list.append(res_dict)

d = pd.DataFrame(re_list)

# %%
d.to_pickle('time_comparison.pkl')