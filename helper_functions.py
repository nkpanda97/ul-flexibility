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
import psutil
import os
import tracemalloc

process = psutil.Process(os.getpid())

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
def update_fonts_for_fig(col=2, **kwargs):
    if len(kwargs.keys())>0:
        print("Updating fonts with kwargs")
        SMALL_SIZE = kwargs['SMALL_SIZE']
        MEDIUM_SIZE = kwargs['MEDIUM_SIZE']
        BIGGER_SIZE = kwargs['BIGGER_SIZE']
    else:
        if col==2:
            SMALL_SIZE = 7
            MEDIUM_SIZE = 8
            BIGGER_SIZE = 9

           
        elif col==1:
            # Plotting Global Settings
            SMALL_SIZE = 6
            MEDIUM_SIZE = 7
            BIGGER_SIZE = 8

                
        else:
            print("Wrong column number")

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
def display_matrix(a):
    text = r'$\left[\begin{array}{*{'
    text += str(len(a[0]))
    text += r'}c}'
    text += '\n'
    for x in range(len(a)):
        for y in range(len(a[x])):
            text += str(a[x][y])
            text += r' & '
        text = text[:-2]
        text += r'\\'
        text += '\n'
    text += r'\end{array}\right]$'
    print(text)
def plot_feasibility_test(u,l,p_feasible, p_infeasible, fsize=(20,10),save_path=None):
    ''' This function plots the feasibility test for the given upper and lower limits of a EV for a provided feasible and infeasible power profile. C.F. Lemma 1 (Ordered UL repreentation) in the paper.
    param u: upper limit of the EV
    param l: lower limit of the EV
    param p_feasible: feasible power profile
    param p_infeasible: infeasible power profile
    param fsize: figure size
    param save_path: path to save the figure
    return: None
    '''
    t_steps = len(u)

    fig, ax = plt.subplots(1,2,figsize=fsize, sharey=True)
    plt.subplots_adjust(wspace=0.1)

    ax[0].plot(np.arange(t_steps),u, marker='o',markersize=3, label=r'$\vec{u}$', color='red',linewidth=0.5) # UPPER LIMIT
    ax[0].plot(np.arange(t_steps),l, marker='o',markersize=3, label=r'$\vec{l}$', color='red', linestyle='--',linewidth=0.5) # LOWER LIMIT
    p_up = [0] + list(np.cumsum(np.sort(p_feasible)[::-1]))  # sort decending Integral of descending
    p_low = [0] + list(np.cumsum(np.sort(p_feasible)) ) # sort ascending  INTEGRAL OF ASCENDING
    ax[0].plot(np.arange(t_steps),p_up, marker='o', markersize=3,label=r'$\sum_k$ descending $(\vec{p}_k)$', color='blue',linewidth=0.5)
    ax[0].plot(np.arange(t_steps),p_low, marker='o', markersize=3,label=r'$\sum_k$ ascending $(\vec{p}_k)$', color='blue', linestyle='--',linewidth=0.5)
    ax[0].set_xticks(np.arange(t_steps))
    ax[0].set_xlabel('time intervals ($k$)')
    ax[0].set_ylabel('Energy (kWh)')
    # ax[0].text(0.75, 0.8,r'$\vec{p}$='+str(p_feasible)+ ' kW', horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
    ax[0].grid(linewidth=0.5 )


    ax[1].plot(np.arange(t_steps),u, marker='o',markersize=3, label=r'$\vec{u}$', color='red',linewidth=0.5)
    ax[1].plot(np.arange(t_steps),l, marker='o',markersize=3, label=r'$\vec{l}$', color='red', linestyle='--',linewidth=0.5)
    p_up = [0] +list(np.cumsum(np.sort(p_infeasible)[::-1]))  # sort decending
    p_low =[0] + list(np.cumsum(np.sort(p_infeasible)))  # sort ascending
    ax[1].plot(np.arange(t_steps),p_up, marker='o',markersize=3, label=r'descending $\vec{p}$', color='blue',linewidth=0.5)
    ax[1].plot(np.arange(t_steps),p_low, marker='o',markersize=3, label=r'ascending $\vec{p}$', color='blue', linestyle='--',linewidth=0.5)
    ax[1].set_xticks(np.arange(t_steps))
    ax[1].set_xlabel('time intervals ($k$)')
    # ax[1].text(0.75, 0.08,r'$\vec{p}$='+str(p_infeasible)+ ' kW', horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
    ax[1].grid(linewidth=0.5 )


    # Annotate infeasible point
    # Find the index where p_up is more than u or p_low is less than l


    inf_index_low= np.where((np.array(p_low)<np.array(l)))
    inf_index_up = np.where((np.array(p_up)>np.array(u)))
    for i in inf_index_low[0]:
        ax[1].annotate('Infeasible', xy=(i, p_low[i]), xytext=(i, p_low[i]+2),
                arrowprops=dict(facecolor='black', shrink=0.05),
                )
    for i in inf_index_up[0]:
        ax[1].annotate('Infeasible', xy=(i, p_up[i]), xytext=(i+0.8, p_up[i]-2),
                arrowprops=dict( arrowstyle='-'),
                )
        ax[1].scatter(i, p_up[i], marker='o',facecolors='none', edgecolors='k', color='k', s=50, zorder=10)

    #make a single legend for the figure
    handles, labels = ax[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4)

   
    a = plt.axes([.165, .77, .07, .1])
    plt.step(np.arange(1,t_steps+1),list(p_feasible)+[p_feasible[-1]], label=r'Feasible $\vec{p}$', color='green',where='post', linewidth=0.5)
    # plt.text(0.245, 0.88,r'$\vec{p}$='+str(p_feasible)+ ' kW', horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
    plt.xticks([], fontsize=5, labels='')
    plt.yticks([0,5,10],fontsize=5)
    plt.ylim([-2, 12])
    plt.tick_params(axis='y',direction='out', pad=0)
    # y-grid only
    plt.grid(axis='y', linestyle='--', linewidth=0.5)

    b = plt.axes([.57, .77, .07, .1])
    plt.step(np.arange(1,t_steps+1),list(p_infeasible)+[p_infeasible[-1]] , label=r'Feasible $\vec{p}$', color='green',where='post', linewidth=0.5)
    plt.xticks([], fontsize=5, labels='')
    plt.yticks([0,10,20],fontsize=5)
    plt.ylim([-5, 25])
    plt.tick_params(axis='y',direction='out', pad=0)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return None

def rescale_ev_data(ev_data, date_sample, del_t, restrict_stop_time=False, total_time_steps=36, verbose_=1000):
    ev_data = ev_data[ev_data['START'].dt.date == date_sample.date()]

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

def build_model_direct(data_for_opt):
    # Pyomo model

    df_for_opt = data_for_opt[0]
    f_windw = data_for_opt[1]
    del_t = data_for_opt[2]

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

def build_model_ul(data_for_opt):

    A = data_for_opt[0]
    b = data_for_opt[1]
    f_windw = data_for_opt[2]
    del_t = data_for_opt[3]
    
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

# Define a function which builts and solves the function, returning time for bult, time for solve, and the solved model according to the type of built_function put as argument
def build_and_solve(data_for_opt, built_function, solver, verbose_=False, model_name=''):

    #Building
    tracemalloc.start()
    tic_build = time.time()
    model = built_function(data_for_opt)
    toc_build = time.time()
    solver_info = solver.solve(model, tee=False)
    toc_solve = time.time()

    t_built = toc_build-tic_build
    t_solve = toc_solve-toc_build
    
    _ , memory_peak = tracemalloc.get_traced_memory()
    if verbose_:
        print(f'Build time for {model_name}: {t_built} seconds')
        print(f'Solve time for {model_name}: {t_solve} seconds')

    tracemalloc.stop()
    return t_built, t_solve, memory_peak, model, solver_info

def single_run(data_for_opt, f_windw, del_t, verbose_=False):
    solver = pyo.SolverFactory('gurobi')
    A_poly, b_poly = generate_system_matrix(data_for_opt, f_windw, del_t)
    data_for_opt_direct = [data_for_opt, f_windw, del_t]
    data_for_opt_ul = [A_poly, b_poly, f_windw, del_t, del_t]
    ##################### Direct method #####################
    t_built_direct, t_solve_direct, memory_direct, model_direct, res_direct = build_and_solve(data_for_opt_direct, build_model_direct, solver, verbose_=verbose_, model_name='direct method')
    
    ###################### UL method #######################
    t_built_ul, t_solve_ul, memory_ul, model_ul, res_ul = build_and_solve(data_for_opt_ul, build_model_ul, solver, verbose_=verbose_, model_name='ul method')


    res_dict = {'No of EVs': len(data_for_opt),
                'No of time steps': int((1/del_t)*(f_windw[1] - f_windw[0])),
                'Time step size (H)': del_t,
                'Direct optimization': {'Build time (s)': t_built_direct,
                                        'Solve time (s)': t_solve_direct,
                                        'Objective value': model_direct.obj(),
                                        'info': res_direct},
                'UL optimization': {'Build time (s)': t_built_ul,
                                    'Solve time (s)': t_solve_ul,
                                    'Objective value': model_ul.obj(),
                                    'info': res_ul},
                'Memory used by direct method (bytes)': memory_direct,
                'Memory used by ul method (bytes)':memory_ul}
    
    return res_dict

def log_model_info(model, file_name='log.txt'):
    with open(file_name, 'w') as output_file:
            model.pprint(output_file)