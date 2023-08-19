import time
import numpy as np
import metaSIR as mSIR 

start_time = time.time()

# Define the travel rates for a 3x3 model with constant movement rates
in_, out_ = mSIR.generate_travel_rates(3, 'synthetic_data', 60)

print(in_)

print(f'Tamanho de inflow: {len(in_)}')

for i in range(len(in_)):
    print(np.sum(in_[i], axis=0)) # Inflow - Normalizado pelas colunas

print()

print(f'Tamanho de outflow: {len(out_)}') 

for i in range(len(out_)):
    print(np.sum(out_[i], axis=1).T) # Outflow - Normalizado pelas linhas

travel_rates = (in_, out_)

dt = (60) / (1000 - 1)
T = np.arange(0, 60 + dt, step=dt)

print(f'Total de pontos p/ integrar: {len(T)}')

print()

'''
# Create a SIRModel object
model = mSIR.SIRModel(
    travel_rates, 
    (T, dt), 
    'SIRModel', 
    np.ones(len(in_)) * .9, 
    .2
)
'''

scenarios = mSIR.generate_integration_scenarios(travel_rates, (T, dt), 'scenarios', (np.ones(len(in_)) * .9, .2))

#row_sums = np.sum(model.N_i, axis=1)

#print(row_sums)

print()

print("--- %s seconds ---" % (time.time() - start_time))