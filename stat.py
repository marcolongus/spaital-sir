#Cargamos en matrices los datos para construir el grafo recursivamente. 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import write_dot

######################################################################################################################
######################################################################################################################
#Data structure of topologia.txt and how to handle it:
#The first column of topologia.txt correspond to the infected particles. Non-childs agents are written first.  
#The nexts columns correspond to the childs of the first column particles.
#To build an epidemic graph we need to get access to every agent and their childs.
#The np.array particle_index[] get wich particle is in one of the n rows of the archive.
#With particle_inverse[particle] we get acces to the posistion of the "particle" in particle_index and we get their childs. 
#So in particle_index[particle_inverse[particle]] we get acces to the row of a "particle" and their childs.
#Input: size of the system N. 
######################################################################################################################
######################################################################################################################

######################################################################################################################
######################################################################################################################

#Deep First Search, tree building, breadth first search, hiarchy_walk

def particle_tree(n,truncate=False):

	particle = particle_inverse[n]	
	if (particle > N): return [-1]

	tree = np.loadtxt(data, skiprows=particle+shift[n_simul], max_rows=1, dtype=int)
	if (tree.size == 1): return []		
	if (truncate and (tree.size >= 2)):
		tree = tree[1:tree.size]

	return tree.tolist() 

def dfs(visited, node, graph,imp=False):

	graph[node] = particle_tree(node,True)

	if (imp): print(graph)
	if node not in visited:
		if (imp): print(node)
		visited.add(node)
		for neighbour in graph.get(node):
			prev_node = neighbour
			dfs(visited, neighbour, graph )

def hiarchy_walk(graph, nodes):
	branch_out_degree=0
	descendientes = set()
	for element in nodes:
		branch_out_degree+=graph.out_degree(element)
		for next_branch_node in list(graph.successors(element)):
			descendientes.add(next_branch_node)
	
	return [branch_out_degree,descendientes]	

######################################################################################################################
######################################################################################################################


#Data, Sourse node, simulation paramaters:

data       = "data/Data1/topologia.txt" #The data of the n-simulation where continuously written on this archive.
N          = 1000  
start_node = 0
simulation = np.loadtxt("data/Data1/evolution.txt",usecols=0, ndmin=1, dtype=int) #size of every epidemic in data. 

TotalSimu = 1

G_ditribution = []
Total_degree  = []

MaxDeg = 0 #Guardamos nodo con más conexiones. 
DeepestGen = 0

for n_simul in range(TotalSimu):
	#print()
	#print("Simulacion ",n_simul,":")

	#shift vector tells where to search for the n-epidemic in data. 
	shift = simulation.copy()

	for i in range(1,shift.size):
		shift[i] = simulation[0:i].sum()
	shift[0] = 0

	args = {"usecols" :0, 
			"max_rows":simulation[n_simul],
			"skiprows":shift[n_simul],
			"ndmin"   :1, 
			"dtype"   :int}

	particle_index   = np.loadtxt(data, **args)
	particle_inverse = np.full( N, 101 , dtype=int) 

	for i in range(N):
		if ( i < particle_index.size):
			particle_inverse[particle_index[i]] = i

	np.savetxt("data/topologia_inverso.txt", particle_inverse, fmt= '%1.f')

	###############################################################################################
	###############################################################################################

	#############################################################################
	#############################################################################

	graph   = {}
	visited = set()
	dfs(visited, start_node, graph)

	#################################################################################

	#Define nx.graph from dfs

	DG = nx.DiGraph(graph) #directed graph
	G  = nx.Graph(graph)   #undirected

	#Hiarchy_walk driver code and graph analysis:

	#Hiarchy walk:
	walk_set = set()
	walk_set.add(start_node)

	p=0

	#Np array for childs per generation [[childs,generation]]: 
	p_distribution = np.array([ hiarchy_walk(DG,walk_set)[0],p ])

	while (hiarchy_walk(DG,walk_set)[0]>0):
		p+=1
		walk_set = hiarchy_walk(DG,walk_set)[1]	
		#print(hiarchy_walk(DG,walk_set),p)
		p_distribution = np.vstack((p_distribution, np.array([ hiarchy_walk(DG,walk_set)[0],p ])))

	#Degree and degree distribution of the tree: 

	#print(p_distribution),print()

	x = p_distribution[...,1]
	y = p_distribution[...,0]

	Deep_x = x.max()+1
	if Deep_x>DeepestGen:
		DeepestGen = Deep_x

	G_ditribution.append(y.tolist())
	Degree_array = np.array(G.degree())[...,1] 
	Total_degree.append(Degree_array)

	Max = Degree_array.max()

	if Max>MaxDeg:
		MaxDeg = Max

	#print("Mean degree:", Degree_array.mean())
	#print("Max degree:" , Max )

	##########################################################
	#GRAFICOS
	##########################################################
	Grafico = True

	if Grafico:
		if (p_distribution.size>2):

			#GRAFICO DE GENERACIONES
			plt.title('Infected per generations')
			plt.xlabel("Generation")
			plt.ylabel("Generation childs")
			plt.xlim(0,Deep_x)
			plt.xticks([i*2 for i in range(Deep_x//2+1)])
			plt.ylim(0,y.max()+5)
			plt.plot(x,y)
			plt.savefig("figuras/epidemic%i.png" %n_simul)
			plt.clf()

			#GRAFICO DE DEGREE DISTRIBUCION
			plt.title("Degree distribution")
			
			bin_limit = Max
			bins = np.linspace(1, bin_limit, bin_limit)

			plt.xticks([i*2 for i in range(bin_limit//2+1)])			
			plt.hist(Degree_array,bins,label="Degree Dist.")
			plt.legend()
			plt.savefig("figuras/degree_histogram%i.png" %n_simul)
			plt.clf()

			#####################################################################################################################

			info=True
			if (info and n_simul>-1):
				#print("nodos:",DG.nodes)
				#print("vertices:",DG.edges),print()
				for element in DG.nodes:
					#print(element, DG.degree(element),np.array(list((DG.successors(element))),dtype=int), DG.out_degree(element))
					pass
				#print()
				#print(nx.info(DG))

				pos = nx.kamada_kawai_layout(G,scale=1000)
				#pos = nx.spring_layout(G,iterations=1000)

				args_g = {	"with_labels" : False, 
							"pos"         : pos,
							"node_size"   : 15,
							"node_color"  : "blue"
							#"node_color"  : range(simulation[n_simul]), 
							#"cmap"        : None
						 }

				
				nx.draw(DG, **args_g)
				nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color= "r", node_size=25)
				plt.savefig("figuras/grafo%i.png" %n_simul)
				#plt.show()
				plt.clf()
				#plt.show()

#####################################################################################################################
#####################################################################################################################

#ESCRITURA DE DATOS Y ORGANIZACIÓN DE ARCHIVOS


print()
G_l = []

for i, element in enumerate(G_ditribution):
	try:
		G_l.append(len(G_ditribution[i]))
	except:
		G_l.append(1)

def bubblesort(lista,G):
# Swap the elements to arrange in order
    for iter_num in range(len(lista)-1,0,-1):
        for idx in range(iter_num):
            if lista[idx]>lista[idx+1]:
                temp       = lista[idx]
                temp_swap  = G[idx]

                lista[idx]   = lista[idx+1]
                lista[idx+1] = temp
                G[idx]       = G[idx+1]
                G[idx+1]     = temp_swap


bubblesort(G_l,G_ditribution)
print("Profundidades:",G_l)

with open("estadistica.txt","w") as f:
	for i in range(TotalSimu-1,-1,-1):
		local = np.array(G_ditribution[i])
		np.savetxt(f, local.reshape(1,local.size), fmt="%3i")


#Plot de promedio de histogramas. 

TotalH = []
for element in Total_degree:
	for conections in element:
		TotalH.append(conections)

bin_limit = MaxDeg
bins = np.linspace(1, bin_limit, bin_limit)

plt.title("Exponential Distribution")
plt.xlabel("Degree")
plt.ylabel("Simulations")
plt.xticks([i*5 for i in range(MaxDeg//5+1)])
plt.hist(TotalH, bins, alpha=0.5, color='darkgreen')
plt.savefig("figuras/Histograma_promedio.png")
plt.show()
plt.clf()

print()
print("Deepest Generation:", DeepestGen)

def plot_loghist(x, bins):
  hist, bins = np.histogram(x, bins=bins)
  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
  plt.hist(x, bins=logbins)
  plt.xscale('log')


plot_loghist(TotalH, bins)
plt.show()