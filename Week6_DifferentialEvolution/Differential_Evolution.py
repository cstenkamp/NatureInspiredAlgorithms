import numpy as np
import matplotlib.pyplot as plt
import csv

class DifferentialEvolution():

	def __init__(self,
				maxPrices=[.45,.25,.2],
				maxDemands=[2e6,3e7,2e7],
				costPrice=.6,
				kWhPerPlant=[5e4,6e5,4e6],
				costPerPlant=[1e4,8e4,4e5],
				maxPlants=[100,50,3],
				n_agents=50,
				scaling_factor = 1,
				crossover_rate = .25,
				max_iter=10000,
				doPrint = True,
				):

		self.maxPrices = np.array(maxPrices)
		self.maxDemands = np.array(maxDemands)
		self.costPrice = np.array(costPrice)
		self.kWhPerPlant = np.array(kWhPerPlant)
		self.costPerPlant = np.array(costPerPlant)
		self.maxPlants = np.array(maxPlants)
		self.doPrint = doPrint
		self.max_iter = max_iter
		self.initPopulation(n_agents)
		self.hist = self.optimize(scaling_factor=scaling_factor, crossover_rate=crossover_rate) #scaling_factor and crossover_rate can be passed, population_size is already in self.population
		profit = []
		for ax in range(n_agents):
			profit.append(self.calcProfit(self.population[ax,:]))
		self.best = self.population[np.argmax(profit),:]


	def initPopulation(self,n_agents):
		productions = np.random.uniform(0,1,[n_agents,3])*np.sum(self.maxDemands)
		perMarketSells = np.random.uniform(0,1,[n_agents,3])*self.maxDemands
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

	def optimize(self, scaling_factor = 1, crossover_rate = .25, random_scaling = False):
		n_agents = self.population.shape[0]
		profit_hist = [0]
		stop_crit = 0
		for iter in range(self.max_iter):
			profits = []
			# for each agent
			for ax in range(n_agents):
				# agent to be optimized/mutated (->target vector)
				x = self.population[ax,:]
				# four other agents
				tmp = np.random.permutation(n_agents)[:4]

				# take three that are not x (to ultimately generate the donor vector)
				a,b,c = tmp[tmp!=ax][:3]
				# a will be the base vector, and the difference will be generated from b and c
				donor_direction = self.population[b,:]-self.population[c,:]
				## potential new agent
				Y = x.copy()
				# take random indice
				R = np.random.randint(self.population.shape[1])
				# take other random indices
				i = np.random.rand(self.population.shape[1])<=crossover_rate #yields array of bools
				# trial generation
				if random_scaling:
					y = self.population[a,:] + np.random.uniform(scaling_factor) * donor_direction
				else:
					y = self.population[a,:] + scaling_factor * donor_direction
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
			if(m==profit_hist[-1]):
				stop_crit += 1
			else:
				stop_crit = 0
			profit_hist.append(m)
			if(stop_crit==3000):
				self.iters = iter
				print('Converged after iteration: {0:7d} | profit: {1:12.2f}'.format(iter,m))
				return profit_hist
			if iter % 100 == 0 and self.doPrint:
				print('iteration: {0:7d} | profit: {1:12.2f}'.format(iter,m))
		self.iters = self.max_iter
		return profit_hist




def print_solutions(DiffEvProb, title):
    print('''
  ##'''+title)
    print('''
    Best Solution: @{0:8.2f} Profit'''.format(DiffEvProb.calcProfit(DiffEvProb.best)))
    print('''
    e1: {0:10.1f}  |  e2: {1:10.1f}  |  e3: {2:10.1f}  | total production: {9:13.2f}
    s1: {3:10.1f}  |  s2: {4:10.1f}  |  s3: {5:10.1f}  | total sells:      {10:13.2f}
    p1: {6:10.4f}  |  p2: {7:10.4f}  |  p3: {8:10.4f}  |'''.format(*DiffEvProb.best,
    																np.sum(DiffEvProb.best[:3]),
    																np.sum(DiffEvProb.best[3:6])))
    print('')
    print('')


def main():
    D1 = DifferentialEvolution(maxPrices=[.45,.25,.2], maxDemands=[2e6,3e7,2e7],costPrice=.6,n_agents=100, max_iter=20000, scaling_factor = 0.9, crossover_rate = .5)
    # D2 = DifferentialEvolution(maxPrices=[.45,.25,.2], maxDemands=[2e6,3e7,2e7],costPrice=.1,n_agents=50)
    # D3 = DifferentialEvolution(maxPrices=[.5,.3,.1], maxDemands=[1e6,5e6,5e6],costPrice=.6,n_agents=50, max_iter=200000, scaling_factor = 0.8, crossover_rate = .4)
	#
    # print_solutions(D1, "Problem 1")
    # print_solutions(D2, "Problem 2")
    # print_solutions(D3, "Problem 3")
	#
    # plt.plot(D1.hist)
    # plt.plot(D2.hist)
    # plt.plot(D3.hist)
	#
    # plt.legend(['D1','D2','D3'])
    # plt.title('Profits Per Problem')
    # plt.xlabel('iteration')
    # plt.ylabel('profit')
    # plt.show()


def parameterSearch(trials_each=5, problemnum = 1, filename=False):
    if(filename):
        header = ["scaling_factor", "crossover_rate", "best", "mean", "md_iter", "best_iter"]
        data = [header]

    for scalFact in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.5, 2]:
        for crossRate in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]:

            print("#######################################################")
            print("Scaling Factor: ", scalFact);
            print("Crossover Rate: ", crossRate);

            if problemnum == 1:
                sols = [0]*trials_each
                iters = [0]*trials_each
                for i in range(trials_each):
                    D1 = DifferentialEvolution(maxPrices=[.45,.25,.2], maxDemands=[2e6,3e7,2e7],costPrice=.6, n_agents=50, scaling_factor = scalFact, crossover_rate = crossRate, doPrint=False, max_iter=20000)
                    sols[i] = D1.calcProfit(D1.best)
                    iters[i] = D1.iters
                print("Problem 1 | mean:",np.mean(sols), "| best:",np.max(sols), "| iters:",np.median(iters))
                if(filename):
                    data.append([scalFact, crossRate, np.max(sols), np.mean(sols), np.median(iters), iters[np.argmax(sols)]])

            elif problemnum == 2:
                sols = [0]*trials_each
                iters = [0]*trials_each
                for i in range(trials_each):
                    D2 = DifferentialEvolution(maxPrices=[.45,.25,.2], maxDemands=[2e6,3e7,2e7],costPrice=.1, n_agents=50, scaling_factor = scalFact, crossover_rate = crossRate, doPrint=False, max_iter=20000)
                    sols[i] = D2.calcProfit(D2.best)
                    iters[i] = D2.iters
                print("Problem 2 | mean:",np.mean(sols), "| best:",np.max(sols),"| iters:",np.median(iters))
                if(filename):
                    data.append([scalFact, crossRate, np.max(sols), np.mean(sols), np.median(iters), iters[np.argmax(sols)]])

            else:
                sols = [0]*trials_each
                iters = [0]*trials_each
                for i in range(trials_each):
                    D3 = DifferentialEvolution(maxPrices=[.5,.3,.1], maxDemands=[1e6,5e6,5e6],costPrice=.6, n_agents=50, scaling_factor = scalFact, crossover_rate = crossRate, doPrint=False, max_iter=20000)
                    sols[i] = D3.calcProfit(D3.best)
                    iters[i] = D3.iters
                print("Problem 3 | mean:",np.mean(sols), "| best:",np.max(sols),"| iters:",np.median(iters))
                if(filename):
                    data.append([scalFact, crossRate, np.max(sols), np.mean(sols), np.median(iters), iters[np.argmax(sols)]])

    if(filename):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerows(data)


if __name__ == "__main__":
    parameterSearch(problemnum = 1, filename="1.csv")
    #parameterSearch(problemnum = 2, filename="2.csv")
    #parameterSearch(problemnum = 3, filename="3.csv")
	#main()

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
