# -*- coding: utf-8 -*-
'''
We plot here the average values for all metrics through time.
To make the comparison between different metrics possible, we 
normalize them according to their values during the entire period (01/jan to 29/feb).
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.transforms as mtransforms
import igraph as ig
import os
from matplotlib import rc

# Latex font --------------------
rc('text', usetex=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

rc('font', **font)
params = {'legend.fontsize': 14}
plt.rcParams.update(params)
# -------------------------------
# Plotting 
fig, ax = plt.subplots(2, 1)
fig.set_size_inches(6, 5)

root_path = r'C:\Users\55119\Documents\Estudos\IC\Semana 3\Resultados'

# Finding the directories within the results folder
path_directories_ori = next(os.walk(root_path +'.'))[1]

# Sorting them according to their date
path_directories_ori.sort()

for axind in range(2):
	if(axind == 0):
		path_directories = path_directories_ori[:60]
	else:
		path_directories = path_directories_ori[60:]
	print(path_directories)

	# Removing the average case
	#path_directories.remove('averaged_chinese_network')
	metrics = ['betweenness_weight', 'closeness_weight', 'diameter'] #, 'density', 'kappa']
	#metrics = ['closeness_weight', 'betweenness_weight', 'vulnerability_weight']
	# rows: networks
	# cols: average metrics from each network
	time_series_metrics = np.zeros((len(path_directories), len(metrics)))

	row = 0
	for d in path_directories:
		col = 0
		for m in metrics:
			# Load the corresponding file with the metric
			file_name = root_path + d + r'\metrics\\' + m + '.csv' 
			metric = np.genfromtxt(file_name, delimiter=';')
			if(m == 'closeness_weight'):
				metric = np.nan_to_num(metric)
			if m == 'diameter' or m == 'density' or m == 'kappa':
				time_series_metrics[row,col] = metric
			else:
				# Storing the average number
				time_series_metrics[row,col] = np.mean(metric[:, 2])
			col += 1
		row += 1

	# Normalizing metrics, considering the entire period (01/jan to 29/fev)
	for c in range(len(metrics)):
		print(metrics[c] + ': ' + str(np.min(time_series_metrics[:, c])) + ' , ' + str(np.max(time_series_metrics[:,c])) )
		time_series_metrics[:,c] = ig.rescale(time_series_metrics[:, c])

	lbls = [r'$b_w$', r'$c_w$', r'$l_{max}$'] 
	colors = ['tab:orange', 'tab:purple', 'tab:brown', 'black', 'palevioletred','brown','darkgreen', 'gray',  'pink', 'red', 'mediumseagreen', 'purple', 'blue']
	markers = ['s', '^', 'o']
	#lbls = [r'$k$', r'$b$', r'$s$', r'$b_w$', r'$c_w$', r'$v_w$'] 
	#colors = ['gray',  'red', 'mediumseagreen', 'purple', 'blue', 'pink', 'magenta']
	metric_index = 0
	for m in metrics:	
		ax[axind].plot(np.linspace(1, len(path_directories), len(path_directories)),
                    time_series_metrics[:,metric_index], 
                    color=colors[metric_index], 
                    lw=2, 
                    label=lbls[metric_index], 
                    zorder=2, 
                    marker=markers[metric_index]
                    )
		metric_index += 1

	# Spring festival (Chunyun) begins
	ax[axind].vlines(10, -0.05, 1.05, colors='k', linestyle='dashed')
	ax[axind].text(6, 1.07, "Spring festival", fontsize=8)

	# Wuhan travel ban
	ax[axind].vlines(23, -0.05, 1.05, colors='k', linestyle='dashed')
	ax[axind].text(19, 1.14, "Wuhan", fontsize=8)
	ax[axind].text(17.25, 1.07, "travel ban", fontsize=8)

	# Lunar new year
	ax[axind].vlines(25, -0.05, 1.05, colors='k', linestyle='dashed')
	ax[axind].text(24.75, 1.14, "Lunar", fontsize=8)
	ax[axind].text(24.75, 1.07, "new year", fontsize=8)

	# End of Chunyun
	ax[axind].vlines(49, -0.05, 1.05, colors='k', linestyle='dashed')
	ax[axind].text(45, 1.14, "End of the", fontsize=8)
	ax[axind].text(44, 1.07, "Spring festival", fontsize=8)

	# Plot configuration
	#plt.plot()
	if(axind == 0):
		ax[axind].legend(ncol=1, fontsize=8, loc=(0.20, 0.5)) # loc='upper right')

	# converting the names into dates: dd-mm
	for i in range(len(path_directories)):
		aux_date = path_directories[i][14:] # only ddmm
		month = 'jan' if aux_date[:2]=='01' else 'feb'
		path_directories[i] = aux_date[2:] + '-' + month

	for i in range(len(path_directories)):
		if( i!=0 and i!=9 and i!=22 and i!=24 and i!=48 and i!=59):
			path_directories[i] = ''

	ax[axind].set_xlim([1, 60])
	ax[axind].set_ylim([-0.05, 1.2])
	ax[axind].set_xticks([])
	plt.xticks(ticks=np.linspace(1,60,60), labels=path_directories, fontsize=12, rotation=90)
	ax[axind].set_ylabel('Normalized metrics')

# label physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
ax[0].text(0.0, 
        0.97, 
        'a) Inflows', 
        transform=ax[0].transAxes + trans, 
        fontsize='medium', 
        va='bottom', 
        fontfamily='serif', 
        fontweight='bold'
        )

ax[1].text(0.0, 
        0.97, 
        'b) Outflows', 
        transform=ax[1].transAxes + trans, 
        fontsize='medium', 
        va='bottom', 
        fontfamily='serif', 
        fontweight='bold'
        )

plt.tight_layout()
fig.savefig(root_path + 'time_series_metrics.png', dpi=300)
fig.savefig(root_path + 'time_series_metrics.pdf', dpi=300)
fig.savefig(root_path + 'time_series_metrics.png', dpi=300)
fig.savefig(root_path + 'time_series_metrics.pdf', dpi=300)