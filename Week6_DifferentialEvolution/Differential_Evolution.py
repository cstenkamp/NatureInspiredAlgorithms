import numpy as np 
import matplotlib.pyplot as plt 


class DifferentialEvolution():

	def __init__(self,
				maxPrices=[.45,.25,.2],
				maxDemands=[2e6,3e7,2e7],
				costPrice=.6,
				kWhPerPlant=[5e4,6e5,5e6],
				costPerPlant=[1e4,8e4,4e5],
				maxPlants=[100,50,3],
				n_agents=50,
				):

		self.maxPrices = np.array(maxPrices)
		self.maxDemands = np.array(maxDemands)
		self.costPrice = np.array(costPrice)
		self.kWhPerPlant = np.array(kWhPerPlant)
		self.costPerPlant = np.array(costPerPlant)
		self.maxPlants = np.array(maxPlants)
		self.initPopulation(n_agents)
		self.hist = self.optimize()
		profit = []
		for ax in range(n_agents):
			profit.append(self.calcProfit(self.population[ax,:]))
		self.best = self.population[np.argmax(profit),:]


	def initPopulation(self,n_agents):
		productions = np.random.uniform(0,1,[n_agents,3])*self.maxDemands
		perMarketSells = np.random.uniform(0,1./3,[n_agents,3])*np.sum(self.maxDemands)
		perMarketPrice = np.random.uniform(0,1,[n_agents,3])*self.maxPrices
		#
		self.population = np.hstack((productions,perMarketSells,perMarketPrice))

	def demand(self, price, maxPrice, maxDemand):
		if price > maxPrice:
			return 0
		if price <= 0:
			return maxDemand
		return maxDemand - price**2 * maxDemand / maxPrice**2

	def cost(self,x, kwhPerPlant, costPerPlant, maxPlants):
		if x == 0:
			return 0
		if x < 0:
			return np.inf 
		if x >= (kwhPerPlant * maxPlants):
			return np.inf
		plantsNeeded = np.ceil(x/kwhPerPlant)
		return plantsNeeded * costPerPlant

	def calcProfit(self,agent):
		revenue = 0
		for i in range(3):
			d_i = self.demand(agent[i+6],self.maxPrices[i],self.maxDemands[i])
			revenue += np.min([d_i,agent[i+3]]) * agent[i+6]
		costs = 0
		# produtions costs
		for i in range(3):
			costs += self.cost(agent[i],
							self.kWhPerPlant[i],
							self.costPerPlant[i],
							self.maxPlants[i])
		# purchasing costs
		costs += np.max([np.sum(agent[3:6])-np.sum(agent[:3]),0]) * self.costPrice
		#print('costs: {0}'.format(costs))
		profit = revenue - costs
		return profit


	def plotMarkets(self):
		prices = np.linspace(0,1,1000)
		# demands
		dM1 = np.array(list(map(lambda x: self.demand(x,self.maxPrices[0],self.maxDemands[0]),prices)))
		dM2 = np.array(list(map(lambda x: self.demand(x,self.maxPrices[1],self.maxDemands[1]),prices)))
		dM3 = np.array(list(map(lambda x: self.demand(x,self.maxPrices[2],self.maxDemands[2]),prices)))
		# revenues
		rM1 = dM1 * prices
		rM2 = dM2 * prices
		rM3 = dM3 * prices
		return rM1, rM2, rM3

	def optimize(self, max_iter=10000, F=1., CR=.25):
		n_agents = self.population.shape[0]
		profit_hist = []
		for iter in range(max_iter):
			profits = []
			# for each agent
			for ax in range(n_agents):
				# agent to be optimized/mutated
				x = self.population[ax,:]
				# four other agents
				tmp = np.random.permutation(n_agents)[:4]
				# take three that are not x
				a,b,c = tmp[tmp!=ax][:3]
				## potential new agent
				Y = x.copy()
				# take random indice
				R = np.random.randint(self.population.shape[1])
				# take other random indices
				i = np.random.rand(self.population.shape[1])<=CR
				# trial generation
				y = self.population[a,:] + np.random.uniform(F) * (self.population[b,:]-self.population[c,:])
				Y[R] = y[R]
				Y[i] = y[i]
				# reset below zero values
				jx = Y[:6]<=0
				Y[:6][jx] = x[:6][jx]
				# check old and new profits
				p0 = self.calcProfit(x)
				p1 = self.calcProfit(Y)
				if p1 > p0:
					self.population[ax,:] = Y
					profits.append(p1)
				else:
					profits.append(p0)
			m = np.max(profits)
			profit_hist.append(m)
			if iter % 100 == 0:
				print('iteration: {0:7d} | profit: {1:12.2f}'.format(iter,m))
		return profit_hist




D1 = DifferentialEvolution(maxPrices=[.45,.25,.2], maxDemands=[2e6,3e7,2e7],costPrice=.6,n_agents=50)
D2 = DifferentialEvolution(maxPrices=[.45,.25,.2], maxDemands=[2e6,3e7,2e7],costPrice=.1,n_agents=50)
D3 = DifferentialEvolution(maxPrices=[.5,.3,.1], maxDemands=[1e6,5e6,5e6],costPrice=.6,n_agents=50)


print('''
Best Solution: @{0:8.2f} Profit'''.format(D1.calcProfit(D1.best)))
print('''e1: {0:10.1f}  |  e2: {1:10.1f}  |  e3: {2:10.1f}  | total production: {9:13.2f}
s1: {3:10.1f}  |  s2: {4:10.1f}  |  s3: {5:10.1f}  | total sells:      {10:13.2f}
p1: {6:10.4f}  |  p2: {7:10.4f}  |  p3: {8:10.4f}  |'''.format(*D1.best,
																np.sum(D1.best[:3]),
																np.sum(D1.best[3:6])))
print('')
print('')

print('Best Solution: @{0:8.2f} Profit'.format(D2.calcProfit(D2.best)))
print('''e1: {0:10.1f}  |  e2: {1:10.1f}  |  e3: {2:10.1f}  | total production: {9:13.2f}
s1: {3:10.1f}  |  s2: {4:10.1f}  |  s3: {5:10.1f}  | total sells:      {10:13.2f}
p1: {6:10.4f}  |  p2: {7:10.4f}  |  p3: {8:10.4f}  |'''.format(*D2.best,
																np.sum(D2.best[:3]),
																np.sum(D2.best[3:6])))
print('')
print('')

print('Best Solution: @{0:8.2f} Profit'.format(D3.calcProfit(D3.best)))
print('''e1: {0:10.1f}  |  e2: {1:10.1f}  |  e3: {2:10.1f}  | total production: {9:13.2f}
s1: {3:10.1f}  |  s2: {4:10.1f}  |  s3: {5:10.1f}  | total sells:      {10:13.2f}
p1: {6:10.4f}  |  p2: {7:10.4f}  |  p3: {8:10.4f}  |'''.format(*D3.best,
																np.sum(D3.best[:3]),
																np.sum(D3.best[3:6])))
print('')
print('')



plt.plot(D1.hist)
plt.plot(D2.hist)
plt.plot(D3.hist)

plt.legend(['D1','D2','D3'])
plt.title('Profits Per Problem')
plt.xlabel('iteration')
plt.ylabel('profit')
plt.show()

"""
rm11,rm12,rm13=D1.plotMarkets()
rm21,rm22,rm23=D2.plotMarkets()
rm31,rm32,rm33=D3.plotMarkets()

plt.subplot(3,2,1)
plt.plot(rm11)
plt.plot(rm12)
plt.plot(rm13)

plt.subplot(3,2,3)
plt.plot(rm21)
plt.plot(rm22)
plt.plot(rm23)

plt.subplot(3,2,5)
plt.plot(rm31)
plt.plot(rm32)
plt.plot(rm33)

plt.subplot(3,2,2)
plt.plot(D1.hist)
plt.subplot(3,2,4)
plt.plot(D2.hist)
plt.subplot(3,2,6)
plt.plot(D3.hist)

plt.show()
"""
