import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
import osmnx as ox
import string as s

def generate_synthetic_community_network(num_nodes, alpha, beta, T, p_mean, p_std, num_communities):
    """
    Generates a synthetic temporal mobility network with communities using the gravity model and a normal distribution 
    for the edge probabilities.

    :param num_nodes: Number of nodes in the network.
    :param alpha: A constant parameter in the gravity model.
    :param beta: A constant parameter in the gravity model.
    :param T: Number of time steps in the network.
    :param p_mean: Mean value of the normal distribution for edge probabilities.
    :param p_std: Standard deviation of the normal distribution for edge probabilities.
    :param num_communities: Number of communities in the network.
    :return: A 3D numpy array with shape (num_nodes, num_nodes, T), representing the mobility network.
    """
    # Initialize the network
    network = np.zeros((num_nodes, num_nodes, T))

    # Calculate the edge probabilities using the gravity model and a normal distribution
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i // int(num_nodes / num_communities) == j // int(num_nodes / num_communities):
                p_ij = alpha * (1 / np.sqrt(i + 1)) * (1 / np.sqrt(j + 1)) * np.exp(-beta * np.sqrt((i - j) ** 2))
            else:
                p_ij = alpha / 10 * (1 / np.sqrt(i + 1)) * (1 / np.sqrt(j + 1)) * np.exp(-beta * np.sqrt((i - j) ** 2))
            p_ij = np.clip(p_ij, 0, 1)  # Ensure the edge probability is between 0 and 1
            for t in range(T):
                network[i, j, t] = max(np.random.normal(p_ij, p_std), 0)
                network[j, i, t] = network[i, j, t]

    return network

def generate_network(n, avg_degree, spatial_pattern='random'):
    """
    Generate a synthetic network based on a network-based spatial model.

    Parameters:
    n (int): The number of nodes in the network.
    avg_degree (float): The average degree of the nodes in the network.
    spatial_pattern (str): The spatial pattern of the nodes in the network. Can be 'random', 'clustered', or 'dispersed'.

    Returns:
    G (networkx.Graph): The synthetic network.
    pos (dict): A dictionary of node positions.
    """
    # Generate spatial coordinates for the nodes
    if spatial_pattern == 'random':
        coords = np.random.rand(n, 2)
    elif spatial_pattern == 'clustered':
        coords = np.random.multivariate_normal([.5, 0.5], [[.01, 0], [0, .01]], size=n)
    elif spatial_pattern == 'dispersed':
        coords = np.array([[i / n, j / n] for i in range(n) for j in range(n)])
    
    # Calculate pairwise distances between nodes
    dists = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
    
    # Generate the network
    G = nx.Graph()
    labels = s.ascii_uppercase[:n] # Create alphabetic labels
    G.add_nodes_from(labels)
    pos = nx.spring_layout(G)
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < avg_degree / (n - 1) * np.exp(-dists[i, j]):
                G.add_edge(labels[i], labels[j])
    
    return G, pos

def plot_network_spatial_model():
    plt.figure(figsize=(12, 6), dpi=300)
    G, pos = generate_network(24, 8, 'clustered')

    plt.style.use('dark_background')
    plt.title('Spatial Model', fontsize=20, fontdict={'color' : 'white'})

    nx.draw(
        G, 
        pos=pos,
        with_labels=True, 
        width=2, 
        node_color='white',
        font_color='black', 
        alpha=1  # sets the opacity of the nodes to 1
    )

    labels = {n : n for n in G.nodes()}
    nx.draw_networkx_labels(
        G, 
        pos=pos,
        labels=labels, 
        font_size=12, 
        font_color='black'
    )

    # Get the edges of the graph
    edges = G.edges()

    # Set the opacity of the edges to 0.5
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='white', alpha=.5, style='solid')
    return plt.savefig('teste.png')

# Load shapefile of Brazilian states and Population
states = gpd.read_file('BR_UF_2021/BR_UF_2021.shp')
pops = pd.read_excel('POP2022_Brasil_e_UFs.xls')

pops.drop(pops.index[[0, 1, 2, 10, 20, 25, 29, 34, 35]], inplace=True)
pops.drop('Unnamed: 1', inplace=True, axis=1)

pops.rename(
    columns={
    'Prévia da população calculada com base nos resultados do Censo Demográfico 2022 até 25 de dezembro de 2022' : 'NM_UF',
    'Unnamed: 2' : 'POP_ESTIM'
    }, 
    inplace=True
)

pops = pops.reset_index(drop=True)

# Create a NetworkX graph from the states
G = nx.Graph()
pos = {}
for index, state in states.iterrows():
    pop = pops.loc[pops['NM_UF'] == state['NM_UF'], 'POP_ESTIM'].item()
    G.add_node(state['NM_UF'], pop=pop)
    pos[state['NM_UF']] = (state['geometry'].centroid.x, state['geometry'].centroid.y)

# Add edges to the graph based on adjacency between states
for i, state in states.iterrows():
    for j, neighbor in states.iterrows():
        if i != j and state['geometry'].intersects(neighbor['geometry']):
            neighbor_pop = pops.loc[pops['NM_UF'] == neighbor['NM_UF'], 'POP_ESTIM']
            if not neighbor_pop.empty:
                weight = neighbor_pop.values[0]
            else:
                weight = 0
            G.add_edge(state['NM_UF'], neighbor['NM_UF'], weight=weight)

# Set the parameters of the SIR model
beta = 0.3
gamma = 0.1
initial_infected = 0.001
timesteps = 100

# Initialize the state of the nodes
for node in G.nodes:
    if np.random.rand() < initial_infected:
        G.nodes[node]['state'] = 'I'
    else:
        G.nodes[node]['state'] = 'S'
    G.nodes[node]['infected'] = 0
    G.nodes[node]['recovered'] = 0

# Run the simulation
for timestep in range(timesteps):
    for node in G.nodes:
        if G.nodes[node]['state'] == 'I':
            G.nodes[node]['infected'] += 1
            if np.random.rand() < gamma:
                G.nodes[node]['state'] = 'R'

# Plot the results

colors = {'S': 'green', 'I': 'red', 'R': 'blue'}
node_colors = [colors[G.nodes[node]['state']] for node in G.nodes]
node_sizes = [np.log10(G.nodes[node].get('pop', 1)) * 10 for node in G.nodes]
edge_widths = [G[node][neighbor]['weight'] / G.nodes[node]['pop'] for node, neighbor in G.edges]
pos = nx.spring_layout(G, k=0.2, seed=42)
plt.figure(figsize=(10, 10), dpi=300)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.5)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
plt.axis('off')
plt.savefig('Brazil.png')