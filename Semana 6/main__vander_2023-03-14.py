import numpy as np
import os
import metaSIR__vander as mSIR
import time
from data_in import G, nodes_statuses

start_time = time.time()

# Generating the OD matrices
# mSIR.generate_OD_matrices(G, nodes_statuses)

# Saving the OD matrices on stacks 
inflow = []
outflow = []

inflow_files = mSIR.sorted_alphanumeric(os.listdir('data/inflow/'))
outflow_files = mSIR.sorted_alphanumeric(os.listdir('data/outflow/'))
data_flows = zip(inflow_files, outflow_files)

for _in, _out in mSIR.get_OD_matrices(data_flows):
    inflow.append(_in)
    outflow.append(_out)

# Integrating the model for M = inflow[0]
# A grid of time points (in days)
t = np.linspace(0, 60, 1000)

_global = mSIR.integrate_model(
    mSIR.all_equations_combined, 
    
    np.array([
    nodes_statuses['A']['S'][0],
    nodes_statuses['A']['I'][0],
    nodes_statuses['A']['R'][0],
    
    nodes_statuses['B']['S'][0],
    nodes_statuses['B']['I'][0],
    nodes_statuses['B']['R'][0],

    nodes_statuses['C']['S'][0],
    nodes_statuses['C']['I'][0],
    nodes_statuses['C']['R'][0]
    ]),

    t,

    nodes_statuses,
    
    inflow[0]
)

cGPT = mSIR.integrate_model(
    mSIR.chatGPT_model, 
    
    np.array([
    nodes_statuses['A']['S'][0],
    nodes_statuses['A']['I'][0],
    nodes_statuses['A']['R'][0],
    
    nodes_statuses['B']['S'][0],
    nodes_statuses['B']['I'][0],
    nodes_statuses['B']['R'][0],

    nodes_statuses['C']['S'][0],
    nodes_statuses['C']['I'][0],
    nodes_statuses['C']['R'][0]
    ]),

    t,

    nodes_statuses,
    
    inflow[0]
)

sModel = mSIR.integrate_model(
    mSIR.sum_model, 
    
    np.array([
    nodes_statuses['A']['S'][0],
    nodes_statuses['A']['I'][0],
    nodes_statuses['A']['R'][0],
    
    nodes_statuses['B']['S'][0],
    nodes_statuses['B']['I'][0],
    nodes_statuses['B']['R'][0],

    nodes_statuses['C']['S'][0],
    nodes_statuses['C']['I'][0],
    nodes_statuses['C']['R'][0]
    ]),

    t,

    nodes_statuses,
    
    inflow[0]
)

simp = mSIR.integrate_model(
    mSIR.equation_simplified, 
    
    np.array([
    nodes_statuses['A']['S'][0],
    nodes_statuses['A']['I'][0],
    nodes_statuses['A']['R'][0],
    
    nodes_statuses['B']['S'][0],
    nodes_statuses['B']['I'][0],
    nodes_statuses['B']['R'][0],

    nodes_statuses['C']['S'][0],
    nodes_statuses['C']['I'][0],
    nodes_statuses['C']['R'][0]
    ]),

    t,

    nodes_statuses,
    
    inflow[0]
)

mSIR.save_txt('mixed', _global)
mSIR.save_txt('chatGPT', cGPT)
mSIR.save_txt('sumModel', sModel)
mSIR.save_txt('simplifiedTest', simp)

print('||DIF|| global and cGPT ->', mSIR.DIF(_global, cGPT, 2))
print('||DIF|| global and Sum Model ->', mSIR.DIF(_global, sModel, 2))
print('||DIF|| global and Simplified Model ->', mSIR.DIF(_global, simp, 2))

# Ploting and saving the SIR Model integration of all the three nodes
mSIR.plot_node_SIR_equations(_global, t, nodes_statuses, 'mixed')
mSIR.plot_node_SIR_equations(cGPT, t, nodes_statuses, 'chatGPT')
mSIR.plot_node_SIR_equations(sModel, t, nodes_statuses, 'sumModel')
mSIR.plot_node_SIR_equations(simp, t, nodes_statuses, 'simplifiedTest')

print("--- %s seconds ---" % (time.time() - start_time))