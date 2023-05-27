import time
import numpy as np
import metaSIR as mSIR

start_time = time.time()

# Define the travel rates for a 3x3 model with constant movement rates
#travel_rates = mSIR.generate_travel_rates(3, 'SIRModel')

travel_rates = np.loadtxt('SIRModel/travel_rates.csv', delimiter=',')
#G = nx.read_graphml("baidu_in_20200101.GraphML")

#travel_rates = nx.to_numpy_matrix(G)
#travel_rates = travel_rates[:3, :3]

#print(travel_rates)

print()

row_sums = np.sum(travel_rates, axis=1)

# Print the row sums
print(row_sums)

print()

# Define a grid of time points to simulate the model
t = np.linspace(0, 60, 1000)

# Create a FluxModel object
model = mSIR.SIRModel(travel_rates, t, 'SIRModel', np.ones(len(travel_rates)) * .9, .2)

row_sums = np.sum(model.N_i, axis=1)

print(row_sums)

#print(model.Y)

print()

print("--- %s seconds ---" % (time.time() - start_time))