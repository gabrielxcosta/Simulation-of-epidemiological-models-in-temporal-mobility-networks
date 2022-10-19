import time
import multiprocessing as mp

# Calculating the efficiency of a given network
def eff_global(g, weights, mode):
	N = g.vcount()
	eff = 0.0
	sp = g.shortest_paths(weights=weights, mode=mode)
	for l in sp:
		for ll in l:
			if ll != 0.0:
				# the eff of each edge is simply 1.0/(shortest path)
				eff += 1.0 / float(ll)

	E = eff / (float(N) * (float(N) - 1.0))
	return E

def vuln_for_each_node(g, weights, k, E, mode):
	g_copy = g.copy()
	# delete edges connected to node k
	del_list = []
	for target_vertex_id in range(g_copy.vcount()):
		try:
			del_list.append(g_copy.get_eid(k, target_vertex_id))
		except:
			pass
	g_copy.delete_edges(del_list)
	# compute the efficiency of the network after removing the edges that incide on node k
	Ek = eff_global(g_copy, weights, mode)
	# Vulnerability index
	Vk = (E - Ek) / E
	return Vk

def vulnerability(g, weights, mode):
	# Global efficiency of the original network
	E = eff_global(g, weights, mode)
	# For each node, remove its adjacency edges, compute the global efficiency of the remaining network,
	# and its associated Vulnerability index 
	'''
	# Serial version
	t = time.time()
	vuln = []
	for k in range(g.vcount()):
		Vk = vuln_for_each_node(g, weights, k, E)
		vuln.append(Vk)
	print('   required time (s): ', time.time() - t)
	'''
	# Parallel version
	t = time.time()
	# Number of subprocesses
	pool = mp.Pool(20)
	argss = [(g, weights, k, E, mode) for k in range(g.vcount())]
	vuln = pool.starmap(vuln_for_each_node, argss)
	print('   required time (s): ', time.time() - t)
	#fout = open('time_to_compute_vuln_in_seconds.txt', 'w')
	#fout.write(str(time.time() - t))
	#fout.close()
	return vuln