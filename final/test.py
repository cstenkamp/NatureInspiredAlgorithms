import numpy as np 
import matplotlib.pyplot as plt 
from ant_colony import AntColony
from sklearn.decomposition import PCA
from geneticAlgorithm import GenAlgo
import time 


class DataParser():

	def __init__(self):
		pass

	def parse(self, fname='distance.txt'):
		with open(fname) as f:
			lines = f.readlines()
		lines = np.array([l.strip().split(' ') for l in lines],dtype=np.float)
		return np.squeeze(lines)



def getPositions(Dist):
	pca = PCA(n_components=2)
	coords = pca.fit_transform(Dist)
	return coords


def calcDistances(positions):
	d = []
	for p in positions:
		d.append(np.sqrt(np.sum((positions-p)**2,1)))
	return np.vstack(d)


def calcRouteLength(route,distances):
	l = 0
	R = [0]+[r+1 for r in route]+[0]
	for rx in range(len(R)-1):
		r1 = R[rx]
		r2 = R[rx+1]
		l += distances[r1,r2]
	l += distances[r2,0]
	return l

T = time.time()

'''
Data  Parsing
'''
dp = DataParser()

distances = dp.parse(fname='distance.txt')
depotDistances = distances[0,1:]
#distances = distances[1:,1:]
demands = dp.parse(fname='demand.txt')
vehicleCapacities = dp.parse(fname='capacity.txt')
vehicleCosts = dp.parse(fname='transportation_cost.txt')


"""
transform to 2d 
"""
coords = getPositions(distances)


'''
Ant Colony Optimization 
stolen from: 
https://raw.githubusercontent.com/Akavall/AntColonyOptimization/master/ant_colony.py

'''
n_ants = 100
n_best = 25
n_iterations = 300
decay = 0.1

distCopy = distances.copy()
np.fill_diagonal(distCopy,np.inf)

colony = AntColony(distCopy, n_ants, n_best, n_iterations, decay, alpha=1.0, beta=1.5,coords=coords)

shortes_path = colony.run()


# only city indices
route = [s[1]-1 for s in shortes_path[0][:-1]]

# demands sorted by traveling route
demandRoute = demands[route]

# genetic algorithm
gA = GenAlgo(coords, demands, route, demandRoute, distances[1:,1:], depotDistances, vehicleCapacities, vehicleCosts)

routes = []
demandRoutes = []
topCosts = []
assignments = []
topGenes = []

# actual learning loop
for k in range(200):
	if k>0:
		#mutate the route
		_=gA.shiftRoute()
	routes.append(gA.route)
	demandRoutes.append(gA.demandRoute)
	for _ in range(3):
		mutants = gA.mutate(gA.genes.copy(),20,rate=.25)
		#
		#xMutants = gA.crossOver(mutants.copy(),rate=.15)
		#
		mCosts,_ = gA.calcCosts(mutants)
		#xMCosts,_ = gA.calcCosts(xMutants)
		costs,_ = gA.calcCosts(gA.genes)
		#
		# take best 20 of current genes
		genes = gA.reselect(gA.genes,20)
		#
		# take best 10 of mutants
		mutants = gA.reselect(mutants,20)
		#
		# take best 10 of crossovers
		#xMutants = gA.reselect(xMutants,10)
		#
		# throw everything together and start again
		#gA.genes = np.vstack((genes,mutants,xMutants))
		gA.genes = np.vstack((genes,mutants))
		#
		# get best costs
		costs, assignment = gA.calcCosts(gA.genes)
		print(costs.min())
		gA.printAssignment()
	topCosts.append(costs.min())
	assignments.append(assignment[costs.argmin()])

topCosts = np.array(topCosts)
print('lowest costs: {0}'.format(topCosts.min()))
print('route: {0}'.format(1+routes[topCosts.argmin()]))
print('assignment: \n{0}'.format(assignments[topCosts.argmin()]))





print('Done: {0}'.format(time.time()-T))



