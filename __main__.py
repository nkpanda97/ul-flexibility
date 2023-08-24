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
#%% [markdown]
'''
######################## Time comparison analysis ##################################
'''
re_run = False
if re_run:
    ev_data_all = pd.read_pickle(config.ev_data_all)
    date_sample = dt.datetime(2022, 5, 1)
    verbose_ = 0
    f_windw = [18,19] # in hours
    n_days = range(1,50)
    re_list = []
    for n_times in tq.tqdm(n_days, position=0,leave=True,desc='Days of transactions'):
        data_for_opt = multiply_transactions(ev_data_all, n_times,date_sample,del_t=1, f_windw=f_windw, verbose_=verbose_ , restrict_stop_time=False, total_time_steps=36)
        for ts in tq.tqdm([1,1/2, 1/3,1/4,1/5,1/6,1/7,1/8,1/9,1/10], position=0,leave=True, desc='Time step size'):
            res_dict = single_run(data_for_opt, f_windw, del_t=ts, verbose_=verbose_)
            re_list.append(res_dict)

    time_analysis_df = pd.DataFrame(re_list)
    time_analysis_df.to_pickle('time_comparison_10_ts.pkl')
else:
    time_analysis_df = pd.read_pickle('time_comparison_10_ts.pkl')
# %%
''' ######################### Plotting time analysis ############################# 
'''
# drop rows with value more than 100


n_t_steps = time_analysis_df['No of time steps'].unique()
for n_t in n_t_steps:
    df = time_analysis_df[time_analysis_df['No of time steps'] == n_t]
    plt.plot(df['No of EVs'],df['Speed up in solve time (%)'], '-*',label=f'{n_t} time steps')

plt.xlabel('Number of EVs')
plt.ylabel('Speed up in solve time (%)')
plt.legend()


# %%

# Plot memory saved


plt.figure()
for n_t in [1,5,10]:
    df = time_analysis_df[time_analysis_df['No of time steps'] == n_t]
    plt.scatter(df['No of EVs'],df['Memory saved by ul method compared to direct (bytes)'],label=f'{n_t}')
plt.legend(ncol=5)