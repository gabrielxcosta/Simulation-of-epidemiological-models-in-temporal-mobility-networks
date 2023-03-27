import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from community import community_louvain
import seaborn as sns
'''
The gravity model is a mathematical framework commonly used to describe the flow of goods, people, or information 
between locations in physical and social systems. It postulates that the flow between two locations is proportional 
to the product of their masses and inversely proportional to their distance. The gravity model has been widely applied in 
various fields, such as transportation planning, economics, geography, and epidemiology, to understand the spatial distribution 
of phenomena and to inform policy-making. Despite its simplicity, the gravity model has shown to be effective in capturing the 
essential characteristics of real-world phenomena and providing insight into the underlying mechanisms governing their dynamics.

This implementation generates a synthetic mobility network with num_nodes nodes, num_steps time steps, 
and a time delta of time_delta between steps. The gravity model is used to determine edge probabilities, 
where gravity_alpha and gravity_beta are parameters that control the strength of the gravity model. 
Finally, edges are added to the network based on these probabilities and with weights drawn from a 
uniform distribution.
'''

def generate_synthetic_gravity_network(num_nodes, alpha, beta, T, p_mean, p_std):
    """
    Generates a synthetic temporal mobility network using the gravity model and a normal distribution for the edge
    probabilities.

    :param num_nodes: Number of nodes in the network.
    :param alpha: A constant parameter in the gravity model.
    :param beta: A constant parameter in the gravity model.
    :param T: Number of time steps in the network.
    :param p_mean: Mean value of the normal distribution for edge probabilities.
    :param p_std: Standard deviation of the normal distribution for edge probabilities.
    :return: A 3D numpy array with shape (num_nodes, num_nodes, T), representing the mobility network.
    """
    # Initialize the network
    network = np.zeros((num_nodes, num_nodes, T))

    # Calculate the edge probabilities using the gravity model and a normal distribution
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p_ij = alpha * (1 / np.sqrt(i + 1)) * (1 / np.sqrt(j + 1)) * np.exp(-beta * np.sqrt((i - j)**2))
            p_ij = np.clip(p_ij, 0, 1)  # Ensure the edge probability is between 0 and 1
            for t in range(T):
                network[i, j, t] = np.random.normal(p_ij, p_std)
                network[j, i, t] = network[i, j, t]

    return network

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
                p_ij = alpha/10 * (1 / np.sqrt(i + 1)) * (1 / np.sqrt(j + 1)) * np.exp(-beta * np.sqrt((i - j) ** 2))
            p_ij = np.clip(p_ij, 0, 1)  # Ensure the edge probability is between 0 and 1
            for t in range(T):
                network[i, j, t] = max(np.random.normal(p_ij, p_std), 0)
                network[j, i, t] = network[i, j, t]

    return network

def generate_synthetic_switching_network(num_nodes, T, p_on, p_off):
    """
    Generates a synthetic switching network with num_nodes nodes and T time steps.

    :param num_nodes: Number of nodes in the network.
    :param T: Number of time steps in the network.
    :param p_on: Probability of an edge being turned on at any given time step.
    :param p_off: Probability of an edge being turned off at any given time step.
    :return: A 3D numpy array with shape (num_nodes, num_nodes, T), representing the switching network.
    """
    network = np.zeros((num_nodes, num_nodes, T), dtype=int)

    for t in range(T):
        # Generate a random graph with the same number of nodes as the previous time step
        if t == 0:
            # For the first time step, generate a fully connected graph
            network[:, :, t] = np.triu(np.ones((num_nodes, num_nodes)), k=1)
        else:
            # For subsequent time steps, generate a random graph based on the previous time step
            network[:, :, t] = np.zeros((num_nodes, num_nodes))
            network[:, :, t][np.triu_indices(num_nodes, k=1)] = np.random.rand(num_nodes * (num_nodes - 1) // 2) < p_on
            network[:, :, t][np.diag_indices(num_nodes)] = 0  # set diagonal to zero to exclude self-loops
            network[:, :, t] *= network[:, :, t - 1]  # edges that were previously off cannot be turned on
            network[:, :, t] += np.random.rand(num_nodes, num_nodes) < p_off
            np.fill_diagonal(network[:, :, t], 0)
            network[:, :, t] *= (1 - network[:, :, t - 1])  # edges that were previously on cannot be turned off

    return network

def generate_metapopulation_network(num_nodes, alpha, beta, T, p_mean, p_std, num_communities, p_on, p_off):
    """
    Generates a synthetic metapopulation network with num_nodes nodes and T time steps.

    :param num_nodes: Number of nodes in the network.
    :param alpha: A constant parameter in the gravity model.
    :param beta: A constant parameter in the gravity model.
    :param T: Number of time steps in the network.
    :param p_mean: Mean value of the normal distribution for edge probabilities.
    :param p_std: Standard deviation of the normal distribution for edge probabilities.
    :param num_communities: Number of communities in the network.
    :param p_on: Probability of an edge being turned on at any given time step.
    :param p_off: Probability of an edge being turned off at any given time step.
    :return: A 3D numpy array with shape (num_nodes, num_nodes, T), representing the metapopulation network.
    """
    # Generate the mobility and switching networks
    mobility_network = generate_synthetic_community_network(num_nodes, alpha, beta, T, p_mean, p_std, num_communities)
    switching_network = generate_synthetic_switching_network(num_nodes, T, p_on, p_off)

    # Combine the networks to create the metapopulation network
    metapopulation_network = np.zeros((num_nodes, num_nodes, T))

    for t in range(T):
        metapopulation_network[:, :, t] = mobility_network[:, :, t] * switching_network[:, :, t]

    return metapopulation_network

def plot_network_with_communities(figname, network):
    """
    Plots the given network with different colors in the communities.
    
    :param network: A 3D numpy array with shape (num_nodes, num_nodes, T), representing the mobility network.
    """
    plt.style.use('dark_background')

    # Extract the graph at time step 0
    f = plt.figure(figsize=(12, 6), dpi=300)

    G = nx.Graph(network[:, :, 0])

    # Set the labels of the nodes
    labels = {i: str(i + 1) for i in range(G.number_of_nodes())}

    # Detect the communities using Louvain
    partition = community_louvain.best_partition(G)

    # Assign a color to each community
    colors = [partition[n] for n in G.nodes()]

    # Draw the graph with different colors in the communities
    pos = nx.spring_layout(G)
    nx.draw(
        G, 
        pos, 
        node_color=colors,
        font_color='black', 
        node_size=300,
        alpha=.7, 
        cmap=plt.cm.get_cmap('spring'), 
        with_labels=True, 
        labels=labels,
    )

    # Get the edges of the graph
    edges = G.edges()

    # Set the opacity of the edges to 0.5
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='white', alpha=.5, style='solid')

    # Return the plot save
    return f.savefig(figname)

def plot_switching_network(network):
    """
    Plots each time step of a switching network generated by generate_synthetic_switching_network.

    :param network: A 3D numpy array with shape (num_nodes, num_nodes, T), representing the switching network.
    """
    num_nodes, _, T = network.shape

    # Loop through each time step and plot the network
    for t in range(T):
        # Create a networkx graph from the adjacency matrix
        G = nx.from_numpy_matrix(network[:, :, t])

        # Plot the graph
        f = plt.figure(figsize=(12, 6), dpi=300)
        nx.draw(G, with_labels=True)
        plt.title(fr"Time step: ${t}$")

        f.savefig('net_' + str(t + 1) + '.png') 
        plt.close(f)

def save_network_to_csv(path, filename, network):
    """
    Saves a synthetic gravity network as a CSV file.

    :param filename: The name of the CSV file to save.
    :param network: A 3D numpy array with shape (num_nodes, num_nodes, T), representing the mobility network.
    """
    # Get the number of nodes and time steps
    num_nodes, _, T = network.shape

    # Initialize a DataFrame to store the network data
    df = pd.DataFrame(columns=['Time', 'Source', 'Target', 'Weight'])

    # Iterate over the time steps and add the edges to the DataFrame
    for t in range(T):
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = network[i, j, t]
                if weight > 0:
                    df = df.append({
                        'Time': t,
                        'Source': i,
                        'Target': j,
                        'Weight': weight
                    }, ignore_index=True)

    # Save the DataFrame as a CSV file
    df.to_csv('data/synthetic_networks/' + path + filename, index=False)

def draw_network(network):
    """
    Plots a metapopulation network.

    :param network: A 3D numpy array with shape (num_nodes, num_nodes, T), representing the metapopulation network.
    """
    T = network.shape[2]
    pos = nx.circular_layout(range(network.shape[0]))

    for t in range(T):
        plt.figure(figsize=(8,8))
        plt.title(f"Time Step {t+1}")
        G = nx.from_numpy_matrix(network[:, :, t])
        nx.draw(G, pos, with_labels=True, node_size=800, node_color='lightblue', edge_color='gray', width=2.0)
        plt.show()

def plot_degree_distribution(network):
    """
    Plots the degree distribution of the nodes in a network using a seaborn distplot.

    :param network: A 3D numpy array with shape (num_nodes, num_nodes, T), representing the mobility network.
    """
    plt.style.use('dark_background')
    sns.set_palette('Accent')
    degrees = np.sum(network, axis=1)
    degrees = degrees.reshape(-1)
    ax = sns.distplot(degrees, kde=True, bins=20)
    ax.grid(b=True, which='major', c='w', lw=1, ls='-', alpha=0.6)
    ax.figure.dpi = 300
    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Degree Distribution', fontsize=18)
    ax.figure.set_size_inches(12, 6)
    ax.figure.dpi = 300
    plt.show()

start_time = time.time()

# Generate the metapopulation network
num_nodes = 10
alpha = 1.0
beta = 1.0
T = 5
p_mean = 0.5
p_std = 0.1
num_communities = 2
p_on = 0.8
p_off = 0.2

metapopulation_network = generate_metapopulation_network(num_nodes, alpha, beta, T, p_mean, p_std, num_communities, p_on, p_off)

#draw_network(metapopulation_network)
plot_degree_distribution(metapopulation_network)
print("--- %s seconds ---" % (time.time() - start_time))