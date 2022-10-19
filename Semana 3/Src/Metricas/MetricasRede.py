import sys
import Vulnerabilidade as vn
import numpy as np
import igraph as ig

# To create directories
from pathlib import Path

# The network name comes from command line. 
net_name = sys.argv[1]
in_or_out = 'in' if 'in' in net_name else 'out'

relative_path = r'C:\Users\55119\Documents\Estudos\IC\Semana3\Resultados\\' + net_name + r'\metrics\\'

# create directory if it does not exist
Path(relative_path).mkdir(parents=True, exist_ok=True)

def export_data(g, data, stat):
    # Saving results to file
    file_out = open(relative_path +  stat + '.csv', 'w')
    for i in range(g.vcount()):
        file_out.write(str(int(g.vs[i]['GbCity'])) + ';' + str(int(g.vs[i]['GbProv'])) + ';' +  str(data[i]) + '\n')
    file_out.close()

def heterogeneity(g):
    degrees = g.degree(mode=in_or_out)
    acc = 0
    for k in degrees:
        acc = acc + k ** 2
    avg = acc / len(degrees)
    het = avg / (np.mean(degrees) ** 2)
    return het

# reading the network from file
g = ig.Graph.Read_GraphML(r"C:\Users\55119\Documents\Estudos\IC\Semana3\Dados\Redes\\" + net_name + '.GraphML')
#g.to_undirected()

print('  Density')
density = g.density()
file_out = open(relative_path + 'density.csv', 'w')
file_out.write(str(density))
file_out.close()

print('  Kappa')
kappa = heterogeneity(g)
file_out = open(relative_path + 'kappa.csv', 'w')
file_out.write(str(kappa))
file_out.close()

########## UNWEIGHTED ##########
print('  Degree')
degrees = g.degree(mode=in_or_out)
export_data(g, degrees, 'degree')

print('  Betweenness')
betweenness = g.betweenness(vertices=None, directed=True, cutoff=None)
export_data(g, betweenness, 'betweenness')

print('  Closeness')
closeness = g.closeness(vertices=None, mode=in_or_out, cutoff=None, weights=None, normalized=True)
export_data(g, closeness, 'closeness')

print('  Vulnerability')
vuln = vn.vulnerability(g, weights=None, mode=in_or_out)
export_data(g, vuln, 'vulnerability')

########## WEIGHTED ##########

# strength (flows are the weights)
print('  Strength')
strength = g.strength(weights='weight', mode=in_or_out)
export_data(g, strength, 'strength')

# Inverse of the flow 
# This is a valid notion of distance in mobility networks, 
# since the higher the flows between neighbors, the closer they are.
g.es['w_inv'] = 1.0 / np.array(g.es['weight'])

print('  Diameter')
diameter = g.diameter(directed=True, weights='w_inv')
file_out = open(relative_path + 'diameter.csv', 'w')
file_out.write(str(diameter))
file_out.close()

print('  Weighted Betweenness')
betweenness_w = g.betweenness(vertices=None, directed=True, cutoff=None, weights='w_inv') 
export_data(g, betweenness_w, 'betweenness_weight')

print('  Weighted Closeness')
closeness_w = g.closeness(vertices=None, mode=in_or_out, cutoff=None, weights='w_inv', normalized=True)
export_data(g, closeness_w, 'closeness_weight')

print('  Weighted Vulnerability')
vuln_w = vn.vulnerability(g, weights='w_inv', mode=in_or_out)
export_data(g, vuln_w, 'vulnerability_weight')