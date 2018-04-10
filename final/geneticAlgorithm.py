import numpy as np
import matplotlib.pyplot as plt 


class GenAlgo():
	def __init__(self, coords, demands, route, demandRoute, distances, depotDistances, vehicleCapacities, vehicleCosts, nGenes=20):
		self.demandRoute = demandRoute
		self.vehicleCapacities = vehicleCapacities
		self.vehicleCosts = vehicleCosts
		self.nGenes = nGenes
		self.route = route 
		self.distances = distances
		self.depotDistances = depotDistances
		self.initGenes()
		self.coords = coords
		self.f = plt.figure()
		self.ax = self.f.add_subplot(111)


	def getRouteLength(self,route):
		l = 0
		for i in range(len(route)-1):
			l += self.distances[route[i],route[i+1]]
		l += self.depotDistances[route[0]]
		l += self.depotDistances[route[-1]]
		return l 

	def initGenes(self):
		# randomly create different permutations of the vehicles
		self.genes = np.array([np.random.permutation(self.vehicleCapacities.shape[0]) for _ in range(self.nGenes)])

	def _calcCosts(self,gene):
		# cum sum of all deliveries
		c = np.cumsum(self.demandRoute)
		# for each vehicle in the gene
		costs = 0
		route_buffer = self.route.copy()
		assignment = []
		for vehicleNr in gene:
			vC = self.vehicleCapacities[vehicleNr]
			# find first customer not to be completely delivered by vehicle
			idx = np.where(c>vC)
			if idx[0].shape[0]==0:
				# add costs of current vehicle
				length_route = self.getRouteLength(route_buffer)
				new_costs = self.vehicleCosts[vehicleNr]*(length_route)
				costs += new_costs 
				assignment.append((route_buffer,self.vehicleCosts[vehicleNr],self.vehicleCapacities[vehicleNr],new_costs,vehicleNr))
				return costs, assignment
			idx = idx[0][0]
			# calculate path length for vehicle
			length_route = self.getRouteLength(route_buffer[:idx+1])
			# add costs of vehicle to current costs 
			new_costs = self.vehicleCosts[vehicleNr]*(length_route)
			costs += new_costs 
			assignment.append((route_buffer[:idx+1],self.vehicleCosts[vehicleNr],self.vehicleCapacities[vehicleNr],new_costs,vehicleNr))
			# delete start of c
			c = c[idx:]
			route_buffer = route_buffer[idx:]
			# and subtract capacity of current vehicle
			c -= vC

	def calcCosts(self,genes):
		costs = np.empty(genes.shape[0])
		assignments = [[]]*genes.shape[0]
		for gx,gene in enumerate(genes):
			costs[gx], assignments[gx] =self._calcCosts(gene)
		return costs, assignments


	def _mutate(self,gene,rate=.3):
		G = gene.copy()	
		# take rate amount of allels and put them in other places
		idx = np.random.permutation(G.shape[0])[:int(G.shape[0]*rate)]
		# new order
		nidx = np.random.permutation(idx.shape[0])
		G[idx] = G[idx[nidx]]
		return G

	def mutate(self,genes,nMutants,rate=.25):
		mutants = []
		for gene in self.genes:
			for mx in range(nMutants):
				mutants.append(self._mutate(gene,rate=rate))
		mutants = np.array(mutants)
		return mutants

	def _crossOver(self,geneA,geneB):
		geneA = geneA.copy()
		geneB = geneB.copy()
		# random index where genes are spliced
		idx = np.random.randint(geneA.shape[0])
		tmp = geneA.copy()
		geneA[:idx] = geneB[:idx]
		geneB[:idx] = tmp[:idx]
		return geneA, geneB

	def crossOver(self,genes,rate=.25):
		crossOvers = []
		# for every gene
		for gx,gene in enumerate(genes):
			# find some other genes to cross over
			idx = np.random.randint(0,genes.shape[0]-1,int(rate*genes.shape[1]))
			idx[idx>=gx] += 1
			for b in idx:
				X = self._crossOver(gene, genes[b,:])
				if X:
					cA, cB = X 
					crossOvers.extend([cA,cB])
		return np.array(crossOvers) 


	def reselect(self,genes,N):
		# get costs of gene:
		costs,_ = self.calcCosts(genes)
		# argsort and take N best
		idx = np.argsort(costs)[:N]
		return genes[idx,:]

	def printAssignment(self):
		self.ax.clear()
		colors = ['r*','b*','k*','m*','c*','g*',
				  'rs','bs','ks','ms','cs','gs']
		# find cheapest assignment
		costs,assignments = self.calcCosts(self.genes)
		idx = np.argmin(costs)
		self.ax.plot(self.coords[0,0],self.coords[0,1],'mo',markersize=10.)
		A = assignments[idx]
		for ax,a in enumerate(A):
			tmp = np.array(a[0])+1
			self.ax.plot(self.coords[tmp][:,0],self.coords[tmp][:,1],colors[ax%len(colors)])
		self.ax.plot(self.coords[0,0],self.coords[0,1],'mo',markersize=10.)
		self.ax.legend(['depot'])
		self.f.canvas.draw()
		self.f.show()
		

	def shiftRoute(self):
		# indicate some starting point:
		idx =  1
		route = np.hstack((self.route[idx:],self.route[:idx]))
		demandRoute = np.hstack((self.demandRoute[idx:],self.demandRoute[:idx]))
		# create some random genes
		genes = np.array([np.random.permutation(self.vehicleCapacities.shape[0]) for _ in range(100)])
		# calculate their costs
		tmpDR = self.demandRoute.copy()
		tmpR  = self.route.copy()
		self.demandRoute = demandRoute
		self.route = route
		costs, assignment = self.calcCosts(genes)
		# take the good genes
		genes = self.reselect(genes,10)
		# mutate
		mutants = self.mutate(genes,100,rate=.5)
		# reselect
		genes = self.reselect(np.vstack((genes,mutants)),10)
		costs, assignment = self.calcCosts(genes)
		# return best gene and corresponding route, demandRoute
		#self.route = tmpR
		#self.demandRoute = tmpDR
		return genes[costs.argmin()], costs.min(), route, demandRoute