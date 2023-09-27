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

scenarios = mSIR.SIROrdinary((T, dt), 'SIROrdinary', .76, .2, inflow.shape[1])

print()
print("--- %s seconds ---" % (time.time() - start_time))