import time
import numpy as np
import metaSIR as mSIR 

start_time = time.time()

# Define the travel rates for a 3x3 model with constant movement rates
#in_, out_ = mSIR.generate_travel_rates(3, 'SIRModel')
in_, out_ = np.loadtxt('SIRModel/inflow_travel_rates.csv', delimiter=','), np.loadtxt('SIRModel/outflow_travel_rates.csv', delimiter=',')
#travel_rates = np.loadtxt('SIRModel/travel_rates.csv', delimiter=',')
#in_ = nx.read_graphml("baidu_in_20200101.GraphML")
#in_ = nx.to_numpy_matrix(in_)
#np.savetxt('inflow_travel_rates.csv', in_, delimiter=',')
#out_ = nx.read_graphml("baidu_out_20200101.GraphML")
#out_ = nx.to_numpy_matrix(out_)
#np.savetxt('outflow_travel_rates.csv', out_, delimiter=',')

print(in_)

print()

print(out_)

travel_rates = (in_, out_)

# Create a SIRModel object
model = mSIR.SIRModel(
    travel_rates, 
    np.linspace(0, 60, 1000), 
    'SIRModel', 
    np.ones(len(in_)) * .9, 
    .2
)

#row_sums = np.sum(model.N_i, axis=1)


#print(row_sums)

print()

print("--- %s seconds ---" % (time.time() - start_time))
