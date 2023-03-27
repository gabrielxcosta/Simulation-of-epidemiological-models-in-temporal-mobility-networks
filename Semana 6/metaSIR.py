import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import randint as rint
from scipy.integrate import odeint

'''
Gabriel Ferreira da Costa - 19.1.4047
Federal University of Ouro Preto // CNPq
Auxiliar package metaSIR.py for my scientific research

Last update: 09/03/2023
'''

def sorted_alphanumeric(data):
    '''
    Auxiliar function to sort the OD matrices in the folder
    '''
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

'''
Generating and manipulating the OD matrices
'''

def r_flow(node_population):
    return rint(1, node_population + 1)

def migration(G, node):
    if node == 'A':
        flow = np.array([.0, r_flow(G.nodes[node]['N']), r_flow(G.nodes[node]['N'])])
        while sum(flow) >= G.nodes[node]['N']:
            flow = np.array([.0, r_flow(G.nodes[node]['N']), r_flow(G.nodes[node]['N'])])
        return flow * 1.
    elif node == 'B':
        flow = np.array([r_flow(G.nodes[node]['N']), .0, r_flow(G.nodes[node]['N'])])
        while sum(flow) >= G.nodes[node]['N']:
            flow = np.array([r_flow(G.nodes[node]['N']), .0, r_flow(G.nodes[node]['N'])])
        return flow * 1.
    else:
        flow = np.array([r_flow(G.nodes[node]['N']), r_flow(G.nodes[node]['N']), .0])
        while sum(flow) >= G.nodes[node]['N']:
            flow = np.array([r_flow(G.nodes[node]['N']), r_flow(G.nodes[node]['N']), .0])
        return flow * 1.

def generate_OD_matrices(G, nodes_statuses):
    '''
    Generate the inflow and outflow data in csv
    '''
    for time in range(30):
        ########################INFLOW############################
        OD_matrix = pd.DataFrame(
            {
                'A' : migration(G, 'A'),
                'B' : migration(G, 'B'),
                'C' : migration(G, 'C')
            },
            index=[key for key in nodes_statuses.keys()]
        )
        OD_matrix = OD_matrix.T # We want the transpose of this matrix
        OD_matrix.to_csv(f'data/inflow/OD_matrix_in_{time + 1}.csv', index_label='districts', index=True)
        ########################OUTFLOW############################
        OD_matrix = pd.DataFrame(
            {
                'A' : migration(G, 'A'),
                'B' : migration(G, 'B'),
                'C' : migration(G, 'C')
            },
            index=[key for key in nodes_statuses.keys()]
        )
        OD_matrix = OD_matrix.T # We want the transpose of this matrix
        OD_matrix.to_csv(f'data/outflow/OD_matrix_out_{time + 1}.csv', index_label='districts', index=True)

def get_OD_matrices(data_flows):
    '''
    Return the zip of inflow and outflow data
    Tuple -> (in, out)
    '''
    inflow_data = []
    outflow_data = []
    for file_in, file_out in data_flows:
        ##################INFLOWS########################

        data_flow = pd.read_csv('data/inflow/' + file_in)
        data_flow.set_index('districts', inplace=True)
        data_flow = data_flow.to_numpy()
        inflow_data.append(data_flow)
        # g = nx.from_numpy_matrix(data_flows, create_using=nx.DiGraph)

        ##################OUTFLOWS#######################        
        data_flow = pd.read_csv('data/outflow/' + file_out)
        data_flow.set_index('districts', inplace=True)
        data_flow = data_flow.to_numpy()
        outflow_data.append(data_flow)
        # g = nx.from_numpy_matrix(data_flows, create_using=nx.DiGraph)
    return zip(inflow_data, outflow_data)

'''
The SIR metapopulation model in a temporal commuting network
'''

'''
    We will need to integrate the system of ordinary differential equations from the metapop SIR model. 
    The integrator we will use is odeint from the scipy.integrate package.

    ODEINT:
    Solve a system of ordinary differential equations using lsoda from the FORTRAN library odepack.

    Solves the initial value problem for stiff or non-stiff systems of first order ode-s:   

                    dy/dt = func(y, t, ...) [or func(t, y, ...)]

    where y can be a vector.

    WILL RECEIVE:

    func -> callable(y, t, …) or callable(t, y, …) -> metaSIR
    Computes the derivative of y at t. If the signature is callable(t, y, ...), then the argument 
    tfirst must be set True;
    
    y0array -> Initial condition on y (can be a vector) -> y0;
    
    tarray -> A sequence of time points for which to solve for y. The initial value point should 
    be the first element of this sequence. This sequence must be monotonically increasing or monotonically 
    decreasing; repeated values are allowed -> t;
    
    argstuple, optional -> Extra arguments to pass to function -> nodes_statuses (dict with attributes of the nodes), M (OD matrix);

    Our dY_dt will be a one-dimensional array encapsulating S_i, I_i, R_i with i going from 1 to 
    N_patches (Total number of metapops). 

    dY_dt = 
    [
        SA - 0,
        IA - 1,
        RA - 2,

        SB - 3,
        IB - 4,
        RB - 5,
    
        SC - 6,
        IC - 7,
        RC - 8
    ]
    _i -> Auxiliar index for dY_dt -> 3 in 3
'''

def equation_1(y, t, nodes_statuses, M):
    '''
    Equation 1 - view github project description
    '''

    N_patches = len(nodes_statuses) # Total number of metapops
    nodes = [key for key in nodes_statuses.keys()] # Auxiliar list of the nodes
    dY_dt = [] # dS_dt, dI_dt, dR_dt will be stored in this empty stack
    _i = 0

    for i in range(N_patches):
        S = y[_i] # Initial condition of S_i
        I = y[_i + 1] # Initial condition of I_i
        R = y[_i + 2] # Initial condition of R_i
        _i += 3
        beta = nodes_statuses[nodes[i]]['beta'] # beta of Node_i - infection rate
        gamma = nodes_statuses[nodes[i]]['gamma'] # gamma of Node_i - recovery rate
        
        if M[i][i] == 0.0:
            _aux = 1.0
            _evaluate = _aux * S * ((_aux * I) / (_aux * nodes_statuses[nodes[i]]['N']))
        else:
            _aux = M[i][i] / nodes_statuses[nodes[i]]['N']
            _evaluate = _aux * S * ((_aux * I) / (_aux * nodes_statuses[nodes[i]]['N']))

        dY_dt.append(-beta * _evaluate) # S
        dY_dt.append(beta * _evaluate - gamma * I) # I
        dY_dt.append(gamma * I) # R
    
    return dY_dt

def equation_2(y, t, nodes_statuses, M):
    '''
    Equation 2 - view github project description
    '''
    N_patches = len(nodes_statuses) # Total number of metapops
    nodes = [key for key in nodes_statuses.keys()] # Auxiliar list of the nodes
    dY_dt = [] # dS_dt, dI_dt, dR_dt will be stored in this empty stack
    _i = 0

    for i in range(N_patches):
        S = y[_i] # Initial condition of S_i
        I = y[_i + 1] # Initial condition of I_i
        R = y[_i + 2] # Initial condition of R_i
        _i += 3
        beta = nodes_statuses[nodes[i]]['beta'] # beta of Node_i - infection rate
        gamma = nodes_statuses[nodes[i]]['gamma'] # gamma of Node_i - recovery rate
        _sum_j = 0
        _sum_num = 0
        _sum_den = 0
        for k in range(N_patches):
            if k != i:
                _sum_num += (M[k][i] / nodes_statuses[nodes[k]]['N']) * I
                _sum_den += (M[k][i] / nodes_statuses[nodes[k]]['N']) * nodes_statuses[nodes[k]]['N']  
        if M[i][i] == 0.0:
            _aux = 1.0
            _sum_j = _aux * S * ((_aux * I) / (_aux * nodes_statuses[nodes[i]]['N']))
        else:
            _sum_j += (M[i][i] / nodes_statuses[nodes[i]]['N']) * S * (_sum_num / _sum_den)

        dY_dt.append(-beta * _sum_j) # S
        dY_dt.append(beta * _sum_j - gamma * I) # I
        dY_dt.append(gamma * I) # R

    return dY_dt

def equation_3(y, t, nodes_statuses, M):
    '''
    Equation 3 - view github project description
    '''
    N_patches = len(nodes_statuses) # Total number of metapops
    nodes = [key for key in nodes_statuses.keys()] # Auxiliar list of the nodes
    dY_dt = [] # dS_dt, dI_dt, dR_dt will be stored in this empty stack
    _i = 0

    for i in range(N_patches):
        S = y[_i] # Initial condition of S_i
        I = y[_i + 1] # Initial condition of I_i
        R = y[_i + 2] # Initial condition of R_i
        _i += 3
        beta = nodes_statuses[nodes[i]]['beta'] # beta of Node_i - infection rate
        gamma = nodes_statuses[nodes[i]]['gamma'] # gamma of Node_i - recovery rate
        _sum_j = 0
        for j in range(N_patches):
            _sum_num = 0
            _sum_den = 0
            if j != i:
                _sum_num += (M[i][j] / nodes_statuses[nodes[i]]['N']) * I
                _sum_den += (M[i][j] / nodes_statuses[nodes[i]]['N']) * nodes_statuses[nodes[i]]['N']
                _aux = (M[i][j] / nodes_statuses[nodes[i]]['N']) * S

            if not M[i][j] == 0.0:
                _sum_j += (_aux) * S * (_sum_num / _sum_den)
            else:
                _sum_j = 1.0 * S * ((1.0 * I) / (1.0 * nodes_statuses[nodes[i]]['N']))

        dY_dt.append(-beta * _sum_j) # S
        dY_dt.append(beta * _sum_j - gamma * I) # I
        dY_dt.append(gamma * I) # R

    return dY_dt

def equation_4(y, t, nodes_statuses, M):
    '''
    Equation 4 - view github project description
    '''    
    N_patches = len(nodes_statuses) # Total number of metapops
    nodes = [key for key in nodes_statuses.keys()] # Auxiliar list of the nodes
    dY_dt = [] # dS_dt, dI_dt, dR_dt will be stored in this empty stack
    _i = 0

    for i in range(N_patches):
        S = y[_i] # Initial condition of S_i
        I = y[_i + 1] # Initial condition of I_i
        R = y[_i + 2] # Initial condition of R_i
        _i += 3
        beta = nodes_statuses[nodes[i]]['beta'] # beta of Node_i - infection rate
        gamma = nodes_statuses[nodes[i]]['gamma'] # gamma of Node_i - recovery rate
        _sum_j = 0
        for j in range(N_patches):
            _sum_num = 0
            _sum_den = 0
            if j != i:
                for k in range(N_patches):
                    if k != i:
                        if not M[k][j] == 0.0:
                            _sum_num += (M[k][j] / nodes_statuses[nodes[k]]['N']) * I
                        else:
                            _sum_num += (1.0 / nodes_statuses[nodes[k]]['N']) * I

                        if not M[k][i] == 0.0:
                            _sum_den += (M[k][i] / nodes_statuses[nodes[k]]['N']) * nodes_statuses[nodes[k]]['N']
                        else:
                            _sum_den += (1.0 / nodes_statuses[nodes[k]]['N']) * nodes_statuses[nodes[k]]['N']

                if not M[i][j] == 0.0:
                    _sum_j += (M[i][j] / nodes_statuses[nodes[i]]['N']) * S * (_sum_num / _sum_den)
                else:
                    _sum_j += (1.0 / nodes_statuses[nodes[i]]['N']) * S * (_sum_num / _sum_den)

        dY_dt.append(-beta * _sum_j) # S
        dY_dt.append(beta * _sum_j - gamma * I) # I
        dY_dt.append(gamma * I) # R

    return dY_dt

def equation_4R(y, t, nodes_statuses, M):
    '''
    Equation 4R - view github project description
    '''    
    N_patches = len(nodes_statuses) # Total number of metapops
    nodes = [key for key in nodes_statuses.keys()] # Auxiliar list of the nodes
    dY_dt = [] # dS_dt, dI_dt, dR_dt will be stored in this empty stack
    _i = 0

    for i in range(N_patches):
        S = y[_i] # Initial condition of S_i
        I = y[_i + 1] # Initial condition of I_i
        R = y[_i + 2] # Initial condition of R_i
        _i += 3
        beta = nodes_statuses[nodes[i]]['beta'] # beta of Node_i - infection rate
        gamma = nodes_statuses[nodes[i]]['gamma'] # gamma of Node_i - recovery rate
        _sum_j = 0
        for j in range(N_patches):
            _sum_num = 0
            _sum_den = 0
            for k in range(N_patches):
                if not M[k][j] == 0.0:
                    _sum_num += (M[k][j] / nodes_statuses[nodes[k]]['N']) * I
                else:
                    _sum_num += (1.0 / nodes_statuses[nodes[k]]['N']) * I

                if not M[k][i] == 0.0:
                    _sum_den += (M[k][i] / nodes_statuses[nodes[k]]['N']) * nodes_statuses[nodes[k]]['N']
                else:
                    _sum_den += (1.0 / nodes_statuses[nodes[k]]['N']) * nodes_statuses[nodes[k]]['N']
            if not M[i][j] == 0.0:
                _sum_j += (M[i][j] / nodes_statuses[nodes[i]]['N']) * S * (_sum_num / _sum_den)
            else:
                _sum_j += (1.0 / nodes_statuses[nodes[i]]['N']) * S * (_sum_num / _sum_den)

        dY_dt.append(-beta * _sum_j) # S
        dY_dt.append(beta * _sum_j - gamma * I) # I
        dY_dt.append(gamma * I) # R

    return dY_dt

def chatGPT_model(y, t, nodes_statuses, M):
    '''
    Computes the equations for a metapopulation model with SIR dynamics.
    
    Args:
    y: array of floats, initial conditions of the system
    t: np.array, current time of the simulation
    nodes_statuses: dict, status of each node in the metapopulation
    M: 2D array of floats, the connectivity matrix between nodes
    
    Returns:
    dY_dt: array of floats, the derivative of each variable in the system
    '''
    N_patches = len(nodes_statuses) # Total number of metapops
    nodes = [key for key in nodes_statuses.keys()] # Auxiliary list of the nodes
    dY_dt = [] # dS_dt, dI_dt, dR_dt will be stored in this empty stack
    _i = 0
    
    for i in range(N_patches):
        S = y[_i] # Initial condition of S_i
        I = y[_i + 1] # Initial condition of I_i
        R = y[_i + 2] # Initial condition of R_i
        _i += 3
        beta = nodes_statuses[nodes[i]]['beta'] # beta of Node_i - infection rate
        gamma = nodes_statuses[nodes[i]]['gamma'] # gamma of Node_i - recovery rate
        N_i = nodes_statuses[nodes[i]]['N'] # Total population of Node_i
        
        sum_j = 0 # initialize the sum over j
        for j in range(N_patches):
            if j == i:
                continue
            M_ij = M[i][j]
            N_j = nodes_statuses[nodes[j]]['N']
            M_jj = M[j][j]
            
            sum_k_I_k = 0 # initialize the sum over k for I_k
            sum_k_N_k = 0 # initialize the sum over k for N_k
            
            for k in range(N_patches):
                if k == j or k == i:
                    continue
                M_kj = M[k][j]
                N_k = nodes_statuses[nodes[k]]['N']
                
                sum_k_I_k += M_kj * I / N_k
                sum_k_N_k += M_kj * N_k / N_k
                
            sum_j += M_ij * (M_jj / N_j + sum_k_I_k / sum_k_N_k) / N_i
        
        dY_dt.append(-beta * S * sum_j) # S
        dY_dt.append(beta * S * sum_j - gamma * I) # I
        dY_dt.append(gamma * I) # R
    
    return dY_dt
    
def all_equations_combined(y, t, nodes_statuses, M):
    eq_1 = equation_1(y, t, nodes_statuses, M)
    eq_2 = equation_2(y, t, nodes_statuses, M)
    eq_3 = equation_3(y, t, nodes_statuses, M)
    eq_4 = equation_4(y, t, nodes_statuses, M)

    # Remixing
    dY_dt = [eq_1[i] + eq_2[i] + eq_3[i] + eq_4[i] for i in range(0, 9)] # sum(dSi_dt), sum(dIi_dt), sum(dRi_dt) will be stored in this empty stack
    
    return dY_dt

def integrate_model(func, y0, t, nodes_statuses, M):
    '''
    Integrate the equation of SIR model using odeint
    '''
    return odeint(func, y0, t, args=(nodes_statuses, M))

def integrate_equations_separately(funcs, y0, t, nodes_statuses, M):
    '''
    Integrate the equations separately of metapop SIR model using odeint
    '''
    return \
        odeint(funcs[0], y0, t, args=(nodes_statuses, M)), \
        odeint(funcs[1], y0, t, args=(nodes_statuses, M)), \
        odeint(funcs[2], y0, t, args=(nodes_statuses, M)), \
        odeint(funcs[3], y0, t, args=(nodes_statuses, M))

def DIF(trajectory_1, trajectory_2, ord):
    '''
    Returns the ||DIF|| between two trajectory vectors

    ord -> order of the norm
    '''
    return np.linalg.norm(trajectory_1 - trajectory_2, ord)

'''
Save particular or combined dY_dt in a txt file(s)
'''

def save_integration_txt(cases, rets):
    '''
    Save the dY_dt array of all the equations in a txt file
    '''
    rets = [ret for ret in rets]
    _i = 1
    _aux = 0
    
    for case in cases:
        ret = rets[_aux]
        
        with open('results/' + case + '/' + case + '.txt', 'w') as file:
            for i in range(0, 9, 3):
                file.write(f'S_{_i}\n\n')
                file.write(str(ret[:, i]))
                file.write('\n\n\n')
                file.write(f'I_{_i}\n\n')
                file.write(str(ret[:, i + 1]))
                file.write('\n\n\n')
                file.write(f'R_{_i}\n\n')
                file.write(str(ret[:, i + 2]))
                file.write('\n\n\n')
                _i += 1

            _i = 1
            _aux += 1

def save_txt(case, ret):
    '''
    Save the dY_dt array of a particular equation in a txt file
    '''
    _i = 1
    _aux = 0
    
    with open('results/' + case + '/' + case + '.txt', 'w') as file:
        for i in range(0, 9, 3):
            file.write(f'S_{_i}\n\n')
            file.write(str(ret[:, i]))
            file.write('\n\n\n')
            file.write(f'I_{_i}\n\n')
            file.write(str(ret[:, i + 1]))
            file.write('\n\n\n')
            file.write(f'R_{_i}\n\n')
            file.write(str(ret[:, i + 2]))
            file.write('\n\n\n')
            _i += 1

        _i = 1
        _aux += 1

'''
Generating and saving the plots of the metapop SIR model
'''

def plot_individual_node_SIR(t, nodes_statuses, fig_node, dict_SIR, case):
    '''
    fig_node -> Node 'A', 'B' or 'C'
    dict_SIR -> ret[:, [S_i, I_i or R_i]] -> i : {0..N=8}
    '''    
    fig = plt.figure(figsize=(15, 6), facecolor='w')
    fig.dpi = 300

    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, dict_SIR['S'], 'b', alpha=.5, lw=3, label='Susceptible')
    ax.plot(t, dict_SIR['I'], 'r', alpha=.5, lw=3, label='Infected')
    ax.plot(t, dict_SIR['R'], 'g', alpha=.5, lw=3, label='Removed')
    ax.figure.suptitle(f'SIR Model - Node ' + fig_node + ' - ' + nodes_statuses[fig_node]['district_name'], fontsize=20)
    ax.set_title(
        r'$\beta$' + r'$\rightarrow$' + fr"{nodes_statuses[fig_node]['beta']}" + ' | ' + r'$\gamma$' + r'$\rightarrow$' + fr'{nodes_statuses[fig_node]["gamma"]}'
    )
    ax.set_xlabel(r'$t$' + ' [days]', fontsize=14)
    ax.set_ylabel('Fraction of ' + r'$N_{i}$', fontsize=14)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(.7)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    return fig.savefig('results/' + case + '/node_' + fig_node + '.png')

def plot_node_SIR_equations(ret, t, nodes_statuses, case):
    '''
    Plot the SIR model equations for a particular case
    '''
    nodes = [key for key in nodes_statuses.keys()]
    _i = 0
    for i in range(0, 9, 3):
        plot_individual_node_SIR(
            t,
            nodes_statuses,
            nodes[_i], 
            {
                'S' : ret[:, [i]],
                'I' : ret[:, [i + 1]],
                'R' : ret[:, [i + 2]]
            },
            case
        )
        _i += 1