import time
import numpy as np
import metaSIR as mSIR

start_time = time.time()

inflow = np.load('inflow.npy')
outflow = np.load('outflow.npy')

inf, out = mSIR.normalize_travel_rates((inflow, outflow))

# See the normalization of the matrices of the day 1
print(np.sum(inf[0], axis=0))
print(np.sum(out[0], axis=1))

print("--- %s seconds ---" % (time.time() - start_time))