import time
import numpy as np
import metaSIR as mSIR 

start_time = time.time()

dt = 60 / (3000 - 1)
T = np.arange(0, 60 + dt, step=dt)

print(f'Total de pontos p/ integrar: {len(T)}')

inflow = np.load('inflow.npy')
outflow = np.load('outflow.npy')
travel_rates = (inflow, outflow)

#dt = (60 / 100) / (1000 - 1)
#T = np.arange(0, 60 + dt, step=dt)

print()

scenarios = mSIR.generate_integration_scenarios(travel_rates, (T, dt), 'BaiduMobilityData', (np.ones(len(inflow[0])) * .9, .2))

print("--- %s seconds ---" % (time.time() - start_time))