"""
 -*- coding: utf-8 -*-
This file contains all functions which are necessary for the project.
@Author : nkpanda
@FileName: __main__.py
@Git ：https://github.com/nkpanda97
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.io import savemat    
from pypoman.projection import project_polytope
from helper_functions import *
import tqdm.notebook as tq

#%% [markdown]
'''
###################### UL vectors for individual EVs ##################################
'''
print('######################## UL vectors for individual EVs ########################')
t_steps = 3
u1,l1 = calculate_u_l(p_max=20, p_min=0, e_min = 15, e_max=25, del_t=1, t_steps=3)
u2, l2 = calculate_u_l(p_max=10, p_min=5, e_min = 20, e_max=30, del_t=1, t_steps=3)
print(f'The U vector for EV#1 is: {u1}\n The L vector for EV#1 is: {l1}')
print(f'The U vector for EV#2 is: {u2}\n The L vector for EV#2 is: {l2}')

#%% [markdown]
'''
################################ Polytope representation #############################
'''
print('######################## Polytope representation ########################')
T = 3
# Vector A
A = generate_combinations_matrix(T)
# print side by side
print(f'The matrix A for T = {T} is:')
display_matrix(A) 

b = generate_polytope_b_matrix(T)
print(f'The matrix b for T = {T} is:')
display_matrix(b)

A_poly = np.vstack([A, -A])
b_poly1 = np.vstack([np.matmul(b,np.array(u1[1:])).reshape((len(b),1)),-np.matmul(b,np.array(l1[1:])).reshape((len(b),1))])
b_poly2 = np.vstack([np.matmul(b,np.array(u2[1:])).reshape((len(b),1)),-np.matmul(b,np.array(l2[1:])).reshape((len(b),1))])

print(f'The matrix A_poly for T = {T} is:')
display_matrix(A_poly)
print(f'The matrix b_poly of Polytope 1 for T = {T} is:')
display_matrix(b_poly1)
print('##################')
print(f'The matrix b_poly of Polytope 2 for T = {T} is:')
display_matrix(b_poly2)

# Saving A, B as  .mat file for plotting
savemat('polytope_matrix.mat', {'A':A_poly, 'B1':b_poly1, 'B2':b_poly2})
print(f'System matrix saved as polytope_matrix.mat')
#%% [markdown]
'''
####################################### Graphical feasibility check ##############################################
'''
print('######################## Graphical feasibility check ########################')
u, l = calculate_u_l(p_max=20, p_min=0, e_min = 15, e_max=25, del_t=1, t_steps=3)
p_feasible = [10,5,10]
p_infeasible = [2,22,1]

plot_feasibility_test(u,l,p_feasible,p_infeasible,save_path='figures/graphical_test.png')

#%% [markdown] # Time and Memory performance analysis 
#[markdown]  ## Constant time step size 
re_run = False
if re_run:
    ev_data_all = pd.read_pickle(config.ev_data_all)
    date_sample = dt.datetime(2020, 6, 1)
    verbose_ = 0
    f_windw = [18,19] # in hours
#     n_days = [1,50,250, 500, 750, 1000]
#     re_list = []
#     for n_times in tq.tqdm(n_days, position=0,leave=True,desc='Days of transactions'):
#         data_for_opt = multiply_transactions(ev_data_all, n_times,date_sample,del_t=1, f_windw=f_windw, verbose_=verbose_ , restrict_stop_time=False, total_time_steps=36)
#         for ts in tq.tqdm([1/4], position=0,leave=True, desc='Time step size'):
#             res_dict = single_run(data_for_opt, f_windw, del_t=ts, verbose_=verbose_)
#             re_list.append(res_dict)

#     time_analysis_df = pd.DataFrame(re_list)
#     time_analysis_df.to_pickle('time_comparison_single_ts.pkl')
#  ## Constant time step size 

    f_windw = [18,19] # in hours
    n_days = [1000]
    re_list = []
    for n_times in tq.tqdm(n_days, position=0,leave=True,desc='Days of transactions'):
        data_for_opt = multiply_transactions(ev_data_all, n_times,date_sample,del_t=1, f_windw=f_windw, verbose_=verbose_ , restrict_stop_time=False, total_time_steps=36)
        for ts in tq.tqdm([1,1/2,1/4, 1/8, 1/16, 1/32, 1/64,1/96], position=0,leave=True, desc='Time step size'):
            res_dict = single_run(data_for_opt, f_windw, del_t=ts, verbose_=verbose_)
            re_list.append(res_dict)

    time_analysis_df = pd.DataFrame(re_list)
    time_analysis_df.to_pickle('time_comparison_multi_ts.pkl')

# %%
''' ######################### Plotting time analysis ############################# 
'''
plt.rcParams.update({'font.size': 10})
############   Plotting for different time steps & number of EVs #################
# time_analysis_df1 = pd.read_pickle('time_comparison_2.pkl')
time_analysis_df2 = pd.read_pickle('time_comparison_single_ts.pkl')
cl_scheme = ['#f7f4f9','#e7e1ef','#d4b9da','#c994c7','#df65b0','#e7298a','#ce1256','#980043','#67001f'][::-1]

# drop rows with value more than 100
fig, ax = plt.subplots(1,3,figsize=(6.8,4), sharey=False)
# color = list of 10 colors

n_t_steps = np.array([  1,  20,    58,    96, 115,   172])
i = 0
# for n_t in n_t_steps:
#     df = time_analysis_df1[time_analysis_df1['No of time steps'] == n_t]
#     t_solve_direct = [df.iloc[i]['Direct optimization']['info']['Solver'][0]['Time'] for i in range(len(df))]
#     t_solve_ul = [df.iloc[i]['UL optimization']['info']['Solver'][0]['Time'] for i in range(len(df))]
#     ax[0].plot(df['No of EVs'],t_solve_direct,label=f'{n_t} time steps', linestyle=':', color=cl_scheme[i])
#     ax[0].plot(df['No of EVs'],t_solve_ul,label=f'{n_t} time steps', linestyle='-', color=cl_scheme[i])    
#     i += 1
ax[0].set_yscale('log')
custom_line = [plt.Line2D([0], [0], color=cl_scheme[i], lw=1.5) for i in range(len(n_t_steps))]
ax[0].legend(custom_line, n_t_steps, ncol=2, fontsize=7, loc='upper left', title='No of time steps')
ax[0].set_ylabel('Time (s)')
ax[0].set_xticks([0,5000,10000,15000], labels=['0','5','10','15'])

ax[0].grid(True, which='both', axis='both', linewidth=0.2)

##################


t_solve_direct2 = [time_analysis_df2.iloc[i]['Direct optimization']['info']['Solver'][0]['Time'] for i in range(len(time_analysis_df2))]
t_solve_ul2 = [time_analysis_df2.iloc[i]['UL optimization']['info']['Solver'][0]['Time'] for i in range(len(time_analysis_df2))]
ax[1].plot(time_analysis_df2['No of EVs'],t_solve_direct2,label=f'Direct optimization', linestyle='--', color=cl_scheme[0], marker='*',markersize=8)
ax[1].plot(time_analysis_df2['No of EVs'],t_solve_ul2,label=f'UL optimization', linestyle='-', color=cl_scheme[0],marker='*',markersize=8)
ax[1].set_yscale('log')
ax[1].set_xlabel('Number of EVs')
ax[1].set_xticks([0,25000,50000,75000, 100000, 125000], labels=['0','25','50','75', '100', '125'])
ax[1].grid(True, which='both', axis='both', linewidth=0.2)
plt.tight_layout()


#######
ax[2].plot(time_analysis_df2['No of EVs'],time_analysis_df2['Memory used by direct method (bytes)']/1024/1024/1024,color=cl_scheme[0], linestyle='--', marker='*',markersize=8)
ax[2].plot(time_analysis_df2['No of EVs'],time_analysis_df2['Memory used by ul method (bytes)'],color=cl_scheme[0], linestyle='-', marker='*',markersize=8)
# ax[2].set_yscale('log')

# %%

# Plot memory saved


plt.figure()
for n_t in [3]:
    df = time_analysis_df[time_analysis_df['No of time steps'] == n_t]
    plt.scatter(df['No of EVs'],df['Memory saved by ul method compared to direct (bytes)'],label=f'{n_t}')
plt.legend(ncol=5)