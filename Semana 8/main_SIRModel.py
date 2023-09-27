import time
import numpy as np
import metaSIR as mSIR 

start_time = time.time()

# Define the travel rates for a 3x3 model with constant movement rates
#in_, out_ = mSIR.generate_travel_rates(303, 'synthetic_data', 60)

travel_rates = (np.load('inflow.npy'), np.load('outflow.npy'))

dt = (60) / (6000 - 1)
T = np.arange(0, 60 + dt, step=dt)

scenarios = mSIR.generate_integration_scenarios(travel_rates, (T, dt), 'scenarios', (np.ones(len(travel_rates[0][0])) * .9, .2))

print()

print("--- %s seconds ---" % (time.time() - start_time))