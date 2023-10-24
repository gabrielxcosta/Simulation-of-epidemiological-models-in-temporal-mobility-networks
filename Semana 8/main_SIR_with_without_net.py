import time
import numpy as np
import pickle as pk
import metaSIR as mSIR 

start_time = time.time()

dt = 60 / (6000 - 1)
T = np.arange(0, 60 + dt, step=dt)

beta, gamma = .76, .2

travel_rates = mSIR.generate_equal_travel_rates(303, 'equal_travel_rates', 60)
mSIR.generate_integration_scenarios('with_without_net', travel_rates, (T, dt), 'scenarios_with_without_net', (beta, gamma))

print()
print("--- %s seconds ---" % (time.time() - start_time))