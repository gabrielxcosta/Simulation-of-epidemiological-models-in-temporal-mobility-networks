import igraph as ig
import pandas as pd
import numpy as np
import os

files = os.listdir('Dados/DadosCrus/OD_matrix_in')
files.sort()

# Carregando os dados dos nós
data_nodes = pd.read_csv("Dados/DadosCrus/Index_City_CH_EN.csv", 
                        delimiter=','
                        )

coordinates = pd.read_csv("Dados/DadosCrus/coordinates.csv", 
                        delimiter=','
                        )

# There are files in two folders: OD_matrix_in and OD_matrix_out.
# in: W[i,j] is the percentage of people from i to j (wrt j). Each column sums to 100 percent.
# out: W[i,j] is the percentage of people from j to i (wrt j). Each column sums to 100 percent..
# Source: https://www.sciencedirect.com/science/article/pii/S2543925122000456?via%3Dihub

for file_name in files:
	print(file_name)

	########################## IN FLOWS ###################################
	# loading the daily population commuting flow between Chinese cities as links' weights
	data_flows = pd.read_csv('Dados/DadosCrus/OD_matrix_in/' + file_name, 
                            delimiter=','
                            )

	# Remove the cities in the flow matrix that are not in the Index_City_CH_EN.csv file
	rows_to_remove = []
	cols_to_remove = []

	for i in range(len(data_flows)):
		if(data_flows.iloc[i, 0] not in data_nodes['City_CH'].tolist()):
			rows_to_remove.append(i)
			cols_to_remove.append(i + 1)

	data_flows = data_flows.drop(data_flows.columns[cols_to_remove], axis=1)
	data_flows = data_flows.drop(data_flows.columns[0], axis=1)
	data_flows = data_flows.drop(data_flows.index[rows_to_remove])

	# Convert it to a weighted adjacency matrix and replace the occurrences of NaN with 0
	adj_matrix = data_flows.to_numpy()
	adj_matrix = np.nan_to_num(adj_matrix)

	# Create the graph
	g = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), attr="weight", mode=ig.ADJ_DIRECTED)

	g.vs['City_CH'] = data_flows.columns

	for i in range(g.vcount()):
		rows = data_nodes[data_nodes["City_CH"] == g.vs[i]["City_CH"]]
		g.vs[i]['GbCity'] = int(rows['GbCity'].to_numpy()[0])
		g.vs[i]['GbProv'] = int(rows['GbProv'].to_numpy()[0])
		g.vs[i]['City_EN'] = str(rows['City_EN'].to_numpy()[0])
		g.vs[i]['Prov_CH'] = str(rows['Prov_CH'].to_numpy()[0])
		g.vs[i]['Prov_EN'] = str(rows['Prov_EN'].to_numpy()[0])
		rows = coordinates[coordinates["GbCity"] == g.vs[i]["GbCity"]]
		g.vs[i]['xcoord'] = rows['xcoord']
		g.vs[i]['ycoord'] = rows['ycoord']

	# The in data must consider the strength mode in
	#print(g.vs['City_EN'])
	#s = g.strength(weights='weight', mode='in')

	#print()
	#for i in range(3):
	#	print(g.vs[i]['City_EN'], s[i])
	
	g.write_graphml('Dados/Redes/' + file_name[:-4] + '.GraphML')
	
	########################## OUT FLOWS ###################################
	# loading the daily population commuting flow between Chinese cities as links' weights
	data_flows = pd.read_csv('Dados/DadosCrus/OD_matrix_out/' + file_name[:6] + 'out' + file_name[8:], 
                            delimiter=','
                            )
	data_flows = data_flows.drop(data_flows.columns[cols_to_remove], axis=1)
	data_flows = data_flows.drop(data_flows.columns[0], axis=1)
	data_flows = data_flows.drop(data_flows.index[rows_to_remove])

	# convert it to a weighted adjacency matrix and replace the occurrences of NaN with 0
	adj_matrix = data_flows.to_numpy()
	adj_matrix = np.nan_to_num(adj_matrix)
	adj_matrix = np.transpose(adj_matrix)

	# Create the graph
	g = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), attr="weight", mode=ig.ADJ_DIRECTED)

	g.vs['City_CH'] = data_flows.columns

	for i in range(g.vcount()):
		rows = data_nodes[data_nodes["City_CH"] == g.vs[i]["City_CH"]]
		g.vs[i]['GbCity'] = int(rows['GbCity'].to_numpy()[0])
		g.vs[i]['GbProv'] = int(rows['GbProv'].to_numpy()[0])
		g.vs[i]['City_EN'] = str(rows['City_EN'].to_numpy()[0])
		g.vs[i]['Prov_CH'] = str(rows['Prov_CH'].to_numpy()[0])
		g.vs[i]['Prov_EN'] = str(rows['Prov_EN'].to_numpy()[0])

		rows = coordinates[coordinates["GbCity"] == g.vs[i]["GbCity"]]
		g.vs[i]['xcoord'] = rows['xcoord']
		g.vs[i]['ycoord'] = rows['ycoord']

	g.write_graphml('Dados/Redes/' + file_name[:6] + 'out' + file_name[8:-4]+'.GraphML')

	os.system('cls')