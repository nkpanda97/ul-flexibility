import numpy as np
import pandas as pd
import argparse


def calculate_u_l(p_min, p_max, e_min, e_max, del_t, t_steps):
    """ This function calculates the upper and lower limits for the EV
    param p_min: Minimum charging power of the EV
    param p_max: Maximum charging power of the EV
    param e_min: Minimum charging energy of the EV
    param e_max: Maximum charging energy of the EV
    param del_t: The time step for optimization
    param t_steps: The total time steps
    return u: The upper limit
    return l: The lower limit
    """
    u = [0]
    l = [0]
    for n in range(1, t_steps+1):
        u.append(min(n*del_t*p_max, e_max-(t_steps-n)*del_t*p_min))
        l.append(max(n*del_t*p_min, e_min-(t_steps-n)*del_t*p_max))
    return u, l

def generate_ev_transactions(n_sample, p_min, p_max, time_interval):
    """ This function generates random EV transactions based on the input parameters.
    param n_sample: Number of transactions to generate
    type n_sample: int
    param p_min: Minimum power consumption of the EV
    type p_min: list
    param p_max: Maximum power consumption of the EV
    type p_max: list
    param time_interval: Time interval of the transaction
    type time_interval: int
    return: A dataframe containing the generated transactions. The following columns are included:
            * p_min: Minimum power consumption of the EV
            * p_max: Maximum power consumption of the EV
            * e_min: Minimum energy consumption of the EV
            * e_max: Maximum energy consumption of the EV   
    rtype: pandas.DataFrame

    """
    def sample_x_y(a, b):
        x_prime = np.random.uniform(a, b)
        y_prime = np.random.uniform(a, b)
        
        x = min(x_prime, y_prime)
        y = max(x_prime, y_prime)
        
        return x, y
    sampled_p_min = np.random.uniform(p_min[0], p_min[1], n_sample)
    sampled_p_max = np.random.uniform(p_max[0], p_max[1], n_sample)
    transaction_p_min_pmax = [(sampled_p_min[i], sampled_p_max[i]) for i in range(n_sample)]
    e_min_theoritical = [sampled_p_min[i]*time_interval for i in range(n_sample)]
    e_max_theoritical = [sampled_p_max[i]*time_interval for i in range(n_sample)]
    sampled_e_min_e_max = [(sample_x_y(e_min_theoritical[i], e_max_theoritical[i])) for i in range(n_sample)]
    transactions_sampled = [{'p_min': transaction_p_min_pmax[i][0], 'p_max': transaction_p_min_pmax[i][1], 'e_min': sampled_e_min_e_max[i][0], 'e_max': sampled_e_min_e_max[i][1]} for i in range(n_sample)]
    transactions_sampled_df = pd.DataFrame(transactions_sampled)

    return transactions_sampled_df

def generate_flex_limits(n_nodes:int, evs_per_node:dict, p_min=[0,11], p_max=[11,22], del_t=1, time_interval=1):
    """ This function generates the flexibility limits for the EVs based on the input parameters.
    param n_nodes: Number of nodes in the network
    type n_nodes: int
    evs_per_node: Number of EVs to aggregate per node
    param evs_per_node: Dictionary containing the number of EVs to aggregate per node
    type evs_per_node: dict
    param p_min: Minimum power consumption of the EV
    type p_min: list
    param p_max: Maximum power consumption of the EV
    type p_max: list
    param del_t: Time step
    type del_t: int
    param time_interval: Time interval of the transaction
    type time_interval: int

    return: A dictionary containing the transactions per node, the aggregated upper limits per node and the aggregated lower limits per node,
            The following keys are included:
            * Transactions_per_node: A dictionary containing the transactions per node. Each dictionary key is the node name and the value is a dataframe containing the transactions
            * Agg_u_per_node: A dictionary containing the aggregated upper limits per node. Each dictionary key is the node name and the value is the aggregated upper limit
            * Agg_l_per_node: A dictionary containing the aggregated lower limits per node. Each dictionary key is the node name and the value is the aggregated lower limit
    rtype: dict
    """
    agg_u_per_node = {}
    agg_l_per_node = {}
    transactions_per_node = {}
    t_steps = int(time_interval/del_t)
    for node in range(n_nodes):
        # Generate random EV transactions
        transactions_sampled_df = generate_ev_transactions(evs_per_node[f'node_{node+1}'], p_min, p_max, time_interval)
        transactions_per_node[f'node_{node+1}'] = transactions_sampled_df
        u_matrix = []
        l_matrix = []
        for i in range(evs_per_node[f'node_{node+1}']):
            u, l = calculate_u_l(p_max=transactions_sampled_df.iloc[i]['p_max'],
                        p_min=transactions_sampled_df.iloc[i]['p_min'],
                        e_max=transactions_sampled_df.iloc[i]['e_max'],
                        e_min=transactions_sampled_df.iloc[i]['e_min'],
                        del_t=del_t,
                        t_steps=t_steps)
            u_matrix.append(u)
            l_matrix.append(l)

        u_matrix = np.array(u_matrix)
        l_matrix = np.array(l_matrix)

        u_agg = np.sum(u_matrix, axis=0)[-1]
        l_agg = np.sum(l_matrix, axis=0)[-1]

        agg_u_per_node[f'node_{node+1}'] = u_agg
        agg_l_per_node[f'node_{node+1}'] = l_agg

    return {'Transactions_per_node':transactions_per_node,
            'Agg_u_per_node':agg_u_per_node,
            'Agg_l_per_node':agg_l_per_node}

n_nodes = 2
evs_per_node = {'node_1': 4, 'node_2': 5}
p_min = [0,11]
p_max = [11,22]
del_t = 1 # Hours
time_interval = 1 # Hours always 1 hour

my_dict = generate_flex_limits(n_nodes, evs_per_node, p_min, p_max, del_t, time_interval)
my_dict

def main(args):
    """ main function to run the script
    """
    print("----- Generating Aggregate Flexibility Limits for EVs -----")

    my_dict = generate_flex_limits(n_nodes=args.n_nodes,
                                    evs_per_node=args.evs_per_node,
                                    p_min=args.p_min,
                                    p_max=args.p_max,
                                    del_t=args.del_t,
                                    time_interval=args.time_interval)
                                   
    print(my_dict)
    return my_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Aggregate Flexibility Limits for EVs')
    parser.add_argument('--n_nodes', type=int, help='Number of nodes in the network', default=2)
    parser.add_argument('--evs_per_node', type=dict, help='Number of EVs to aggregate per node', default={'node_1': 4, 'node_2': 5})
    parser.add_argument('--p_min', type=list, help='Minimum power consumption of the EV', default=[0,11])
    parser.add_argument('--p_max', type=list, help='Maximum power consumption of the EV', default=[11,22])
    parser.add_argument('--del_t', type=int, help='Time step', default=1)
    parser.add_argument('--time_interval', type=int, help='Time interval of the transaction', default=1)
    args = parser.parse_args()
    main(args)
    
