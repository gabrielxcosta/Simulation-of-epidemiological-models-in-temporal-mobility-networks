import os

# Available networks
files = os.listdir('Dados/Redes')

# Sorting files
files.sort()

print(files)

# Go to the metrics directory
mydir = os.chdir('Src/Metricas')
mydir = os.getcwd()

for file in files:
	print('Computing metrics: ', file)
	os.system(r'python MetricasRede.py' + ' ' + file[:-8])