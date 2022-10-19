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
g = ig.Graph.Read_GraphML(r'C:\Users\55119\Documents\Estudos\IC\Semana 3\Dados\Redes\\' + net_name + '.GraphML')

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
	df = pd.read_csv('../results/' + net_name + '/metrics/' + metric + '.csv', delimiter=';', header=None)
	df.columns = ['id', 'city_code', 'metric']
	print(df['metric'])

	g.vs[metric] = df['metric']

	'''
	# normalization
	node_size = np.array(df['metric'])
	max_size = max(node_size)
	min_size = min(node_size)
	# Min and max plot sizes
	T_min = 8
	T_max = 25
	node_size = T_min + ( (node_size - min_size) / (max_size - min_size) ) * (T_max - T_min) # normalization
	g.vs["size"] = node_size.tolist()
	'''
	#n, bins, patches = plt.hist(g.vs[metric], 50, density=True, facecolor='g', alpha=0.75)

	g.vs["size"] = 12

	layout = []
	for i in range(g.vcount()):

		# coordinates
		layout.append((g.vs[i]["xcoord"],-g.vs[i]["ycoord"]))

		# keep only the names of the biggest cities and color them differently
		'''if(g.vs[i]["label"] != "Shanghai" and 
		g.vs[i]["label"] != "Beijing" and
 		g.vs[i]["label"] != "Guangzhou" and
		g.vs[i]["label"] != "Chengdu" and
		g.vs[i]["label"] != "Chongqing" and
		g.vs[i]["label"] != "Wuhan"):'''

		# Good for closeness
		if(g.vs[i]["label"] != "Wuhan" and g.vs[i]["label"] != "Beijing"): # and g.vs[i]["label"] != "Shiyan"):

			g.vs[i]["label"] = ""
			g.vs[i]["vertex_shape"] = "circle"
		else:
			g.vs[i]["vertex_shape"] = "triangle"
			g.vs[i]["size"] = 25	
		



		'''if(g.vs[i][metric] < bins[-40]):
			g.vs[i]['color'] = 'green'
		elif(g.vs[i][metric] >= bins[-40] and g.vs[i][metric] < bins[-37]):
			g.vs[i]['color'] = 'yellow'
		elif(g.vs[i][metric] >= bins[-37] and g.vs[i][metric] < bins[-35]):
			g.vs[i]['color'] = 'orange'
		else:
			g.vs[i]['color'] = 'red'''

	g.vs['metric_plt'] = ig.rescale(g.vs[metric], clamp=True)
	cmap1 = plt.get_cmap('RdYlGn_r') #LinearSegmentedColormap.from_list("vertex_cmap", ["green", "red"])
	g.vs["color"] = [cmap1(m) for m in g.vs['metric_plt']]

	#g.vs["size"] = ig.rescale(g.vs[metric], (6,25))

	#n, bins, patches = plt.hist(g.vs[metric], 50, density=True, facecolor='g', alpha=0.75)
	#plt.show()

	#print(bins)

	#vs_colors = [int(i * 255 / np.max(g.vs[metric])) for i in g.vs[metric]]
	#es_colors = [int(i * 255 / np.max(g.es['weight'])) for i in g.es['weight']]

	#g.vs['color'] = vs_colors
	#g.es['color'] = es_colors


	#n, bins, patches = plt.hist(g.es['weight'], 100, density=True, facecolor='g', alpha=0.75)
	#plt.show()

	# Edge color
	'''for i in range(g.ecount()):
		if(g.es[i]['weight'] < bins[-30]):
			g.es[i]['color'] = "rgba(0,150,0,0.15)"
			g.es[i]['edge_width'] = 0.3
		elif(g.es[i]['weight'] >= bins[-30]and g.es[i]['weight'] < bins[-25]):
			g.es[i]['color'] = "rgba(150,150,0,0.5)"
			g.es[i]['edge_width'] = 0.3
		elif(g.es[i]['weight'] >= bins[-25] and g.es[i]['weight'] < bins[-20]):
			g.es[i]['color'] = 'orange'
			g.es[i]['edge_width'] = 0.3
		else:
			g.es[i]['color'] = 'red'
			g.es[i]['edge_width'] = 0.3'''

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
	'''
	# Plot the graph with two colorbars
	fig, axs = plt.subplots(
		3, 1,
		figsize=(7, 6),
		gridspec_kw=dict(height_ratios=(15, 1, 1)
	)
	norm1 = ScalarMappable(norm=Normalize(min(g.vs[metric]), max(g.vs[metric])), cmap=cmap1)
	norm2 = ScalarMappable(norm=Normalize(min(g.es['weight']), max(g.es['weight'])), cmap=cmap2)
	plt.colorbar(norm1, cax=axs[1], orientation="horizontal", label='Vertex Betweenness')
	plt.colorbar(norm2, cax=axs[2], orientation="horizontal", label='Edge Betweenness')
	'''

	'''fig, axs = plt.subplots(
		3, 1,
		figsize=(7, 6),
		gridspec_kw=dict(height_ratios=(15, 1, 1)),
	)

	norm1 = ScalarMappable(norm=Normalize(min(g.vs[metric]), max(g.vs[metric])), cmap=cmap1)
	norm2 = ScalarMappable(norm=Normalize(min(g.es['weight']), max(g.es['weight'])), cmap=cmap2)
	plt.colorbar(norm1, orientation="horizontal", label='Vertex Betweenness')
	#plt.colorbar(norm2, cax=axs[2], orientation="horizontal", label='Edge Betweenness')'''

	ig.plot(g, 
            r'C:\Users\55119\Documents\Estudos\IC\Semana 3\Resultados\network' + net_name + '_' + metric + '.png', **visual_style)