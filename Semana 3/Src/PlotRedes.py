import igraph as ig
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

# The network name comes from command line. 
net_name = sys.argv[1]

# reading the network from file
g = ig.Graph.Read_GraphML('../Dados/Redes/' + net_name + '.GraphML')

g.vs['label'] = g.vs['City_EN']
min_x = np.min(g.vs["xcoord"])
max_x = np.max(g.vs["xcoord"])
min_y = np.min(g.vs["ycoord"])
max_y = np.max(g.vs["ycoord"])

dim_x = max_x - min_x
dim_y = max_y - min_y
scale = 20.0
width = dim_x * scale
height = dim_y * scale
print(width, height)


metrics = ['degree', 'betweenness', 'strength', 'betweenness_weight', 'closeness_weight', 'vulnerability_weight']
metrics = ['betweenness_weight', 'closeness_weight']

for metric in metrics:
	# Metrics
	df = pd.read_csv('../Resultados/' + net_name + '/metrics/' + metric + '.csv', delimiter=';', header=None)
	df.columns = ['id', 'city_code', 'metric']
	print(df['metric'])

	g.vs[metric] = df['metric']

	g.vs["size"] = 12

	layout = []
	for i in range(g.vcount()):

		# coordinates
		layout.append((g.vs[i]["xcoord"],-g.vs[i]["ycoord"]))

		# Good for closeness
		if(g.vs[i]["label"] != "Wuhan" and g.vs[i]["label"] != "Beijing"): # and g.vs[i]["label"] != "Shiyan"):

			g.vs[i]["label"] = ""
			g.vs[i]["vertex_shape"] = "circle"
		else:
			g.vs[i]["vertex_shape"] = "triangle"
			g.vs[i]["size"] = 25	

	g.vs['metric_plt'] = ig.rescale(g.vs[metric], clamp=True)
	cmap1 = plt.get_cmap('RdYlGn_r') #LinearSegmentedColormap.from_list("vertex_cmap", ["green", "red"])
	g.vs["color"] = [cmap1(m) for m in g.vs['metric_plt']]

	g.es['weight_plt'] = ig.rescale(g.es['weight'], clamp=True)
	cmap2 = plt.get_cmap('RdYlGn_r') # LinearSegmentedColormap.from_list("egde_cmap", ["green", "red"])
	g.es["color"] = [cmap2(w) for w in g.es['weight_plt']]

	g.es['weight_plt'] = ig.rescale(g.es['weight'], (0.005,0.11))
	g.es["edge_width"] = [w**(1.5) * 150 for w in g.es['weight_plt']]

	visual_style = {
		"vertex_size": g.vs["size"],
		"vertex_shape": g.vs["vertex_shape"],
		"vertex_label_size": 20,
		"vertex_label_dist": 1,
		"vertex_label_color": "white",
		"edge_width": g.es['edge_width'],
		"layout": layout,
		"bbox": (width, height),
		"margin": 30,
		"edge_arrow_size": 0.2
	}

	ig.plot(g, 
            'network' + net_name + '_' + metric + '.png', **visual_style)