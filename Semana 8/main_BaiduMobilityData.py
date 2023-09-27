import time
import numpy as np
import pickle as pk
import metaSIR as mSIR 

start_time = time.time()

dt = 60 / (6000 - 1)
T = np.arange(0, 60 + dt, step=dt)

inflow = np.load('normalized_inflow.npy')
outflow = np.load('normalized_outflow.npy')
travel_rates = (inflow, outflow)

scenarios = mSIR.generate_integration_scenarios(travel_rates, (T, dt), 'BaiduMobilityData', (np.ones(len(inflow[0])) * .76, .2))

print()
print("--- %s seconds ---" % (time.time() - start_time))