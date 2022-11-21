''''''
import seaborn as sns
import numpy as np
import networkx as nx
from random import randint
import matplotlib.pyplot as plt
from fa2l import force_atlas2_layout # https://github.com/bhargavchippada/forceatlas2
import ndlib.models.ModelConfig as mc # https://ndlib.readthedocs.io/en/latest/
import ndlib.models.epidemics as ep # https://ndlib.readthedocs.io/en/latest/
sns.set_theme()

'''
In NDlib are implemented the following Epidemic models:
    SI
    SIS
    SIR
    SEIR (DT)
    SEIR (CT)
    SEIS (DT)
    SEIS (CT)
    SWIR
    Threshold
    Generalised Threshold
    Kertesz Threshold
    Independent Cascades
    Profile
    Profile Threshold
    UTLDR
    Independent Cascades with Community Embeddedness and Permeability
    Independent Cascades with Community Permeability
    Independent Cascades with Community Embeddedness

'''

plt.rcParams.update({'figure.max_open_warning': 0})

g = nx.karate_club_graph()
N = g.number_of_nodes() - 1 # Começamos do nó zero

# Prepare the node positions for the output
positions = force_atlas2_layout(
    g,
    iterations=1000,
    pos_list=None,
    node_masses=None,
    outbound_attraction_distribution=False,
    lin_log_mode=False,
    prevent_overlapping=False,
    edge_weight_influence=1.0,
    jitter_tolerance=1.0,
    barnes_hut_optimize=True,
    barnes_hut_theta=0.5,
    scaling_ratio=2.0,
    strong_gravity_mode=False,
    multithread=False,
    gravity=1.0
)

# Selection of SIR model in 'ep' package 
model = ep.SIRModel(g)

# Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.7)
cfg.add_model_parameter('gamma', 0.4)
cfg.add_model_initial_configuration('Infected', [randint(0, N)])
model.set_initial_status(cfg)

# Running the simulation
number_of_iteration = 15
iterations = model.iteration_bunch(number_of_iteration)

color_map = []
for iteration in range(g.number_of_nodes()):
    color_map.append('green')

i = 1

S = []
I = []
R = []

# Para cada iteração
for iteration in iterations:
    S.append(iteration['node_count'][0])
    I.append(iteration['node_count'][1])
    R.append(iteration['node_count'][2])
    f = plt.figure()
    for index, status in iteration['status'].items():
        if status == 1:
            color_map[index] = 'red'
        if status == 2:
            color_map[index] = 'grey'
    
    nx.draw(g, positions, node_color=color_map, with_labels=True)
    f.savefig(f'Timelapse2/Time_Sir_{i}.png', dpi=300)
    i += 1

t = np.linspace(0, 15, 15)
plt.figure(figsize=(12, 6))
plt.plot(t, S, color='green')
plt.plot(t, I, color='red')
plt.plot(t, R, color='gray')
plt.ylabel('People in each compartment', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.title("SIR in time - Zachary's Karate Club Network", fontsize=18)
plt.savefig('SIR_In_Time.png', dpi=300)
