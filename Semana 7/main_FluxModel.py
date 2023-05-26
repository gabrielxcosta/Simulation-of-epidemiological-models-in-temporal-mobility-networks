import time
import numpy as np
import metaSIR as mSIR
import networkx as nx

start_time = time.time()

# Define the travel rates for a 3x3 model with constant movement rates
#travel_rates = mSIR.generate_travel_rates(3, 'FluxModel')
travel_rates = np.loadtxt('FluxModel/travel_rates.csv', delimiter=',')

# print(travel_rates)

print()

#G = nx.read_graphml("baidu_in_20200101.GraphML")

#travel_rates = nx.to_numpy_matrix(G)
#travel_rates = travel_rates[:3, :3]

row_sums = np.sum(travel_rates, axis=1)

# Print the row sums
#print(row_sums)

# Define a grid of time points to simulate the model
t = np.linspace(0, 120, 1000)

# print()

# Simulate the model
model = mSIR.FluxModel(travel_rates, t, 'FluxModel')

# print()

row_sums = np.sum(model.N_i, axis=1)

print(row_sums)

print()

print("--- %s seconds ---" % (time.time() - start_time))