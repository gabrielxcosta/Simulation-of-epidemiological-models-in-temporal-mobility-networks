import networkx as nx
import numpy as np

'''
Our node is a digraph
N_i = 75000
Picking only 500 infected individuals in all the 3 nodes
''' 

G = nx.DiGraph()
G.add_edges_from([
    ('A', 'B'),
    ('B', 'A'),
    ('A', 'C'),
    ('C', 'A'),
    ('B', 'C'),
    ('C', 'B')
])

nx.set_node_attributes(G, {
    'A': {
        'label': 'Subpopulation A',
        'N': 75000
    },
    'B': {
        'label': 'Subpopulation B',
        'N': 75000
    },
    'C': {
        'label': 'Subpopulation C',
        'N': 75000
    }
})

nodes_statuses = {
    'A' : {
            'S' : np.array([G.nodes['A']['N'] - 500]),
            'I' : np.array([500]),
            'R' : np.array([0]),
            'beta' : .915,
            'gamma' : .225,
            'N' : G.nodes['A']['N'],
            'district_name' : G.nodes['A']['label']
        },
    'B' : {
            'S' : np.array([G.nodes['B']['N'] - 500]),
            'I' : np.array([500]),
            'R' : np.array([0]),
            'beta' : .915,
            'gamma' : .225,
            'N' : G.nodes['B']['N'],
            'district_name' : G.nodes['B']['label']    
        },
    'C' : {
            'S' : np.array([G.nodes['C']['N'] - 500]),
            'I' : np.array([500]),
            'R' : np.array([0]),
            'beta' : .915,
            'gamma' : .225,
            'N' : G.nodes['C']['N'],
            'district_name' : G.nodes['C']['label']            
        }
}