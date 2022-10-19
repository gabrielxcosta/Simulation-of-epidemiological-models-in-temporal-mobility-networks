import os

if __name__ == '__main__':
	# Available networks
	files = os.listdir(r'C:\Users\55119\Documents\Estudos\IC\Semana3\Dados\Redes\\')

	# Sorting files
	files.sort()

	print(files)

	# Go to the metrics directory
	mydir = os.chdir(r'C:\Users\55119\Documents\Estudos\IC\Semana3\Src\Metricas')
	mydir = os.getcwd()

	for file in files:
		print('Computing metrics: ', file)
		os.system(r'python MetricasRede.py' + ' ' + file[:-8])