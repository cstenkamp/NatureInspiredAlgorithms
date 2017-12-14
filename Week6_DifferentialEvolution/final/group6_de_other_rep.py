import numpy as np

class Problem():
    def __init__(self,
        maxPrices=[.45,.25,.2],
        maxDemands=[2e6,3e7,2e7],
        costPrice=.6,
        kWhPerPlant=[5e4,6e5,4e6],
        costPerPlant=[1e4,8e4,4e5],
        maxPlants=[100,50,3]):
        self.np = len(maxPrices)
        self.costPrice = costPrice
        self.maxPrices = np.array(maxPrices)
        self.maxDemands = np.array(maxDemands)
        self.kWhPerPlant = np.array(kWhPerPlant)
        self.costPerPlant = np.array(costPerPlant)
        self.maxPlants = np.array(maxPlants)
        self.costPerKwh = self.costPerPlant/self.kWhPerPlant
        self.plantOrder = np.argsort(self.costPerKwh)

def comp_winnings(p, x):
    # compute demands from prices
    demands = np.zeros(p.np)
    for i in range(p.np):
        demands[i] = (p.maxDemands[i]-(x[i]*x[i])*p.maxDemands[i]/(p.maxPrices[i]*p.maxPrices[i]))
    # total demand
    d_all =np.sum(demands)
    # compute winnings
    wins = np.sum(demands*x)
    # compute the distribution on plant types for the best profit
    m_win_plants = get_max_win(p, d_all, wins)
    return m_win_plants + list(demands)

def get_max_win(p, d_all, wins):
    # go through plant types sorted after costPerKwh - increasing
    net_wins = []
    ptcs = np.zeros(p.np)
    rest = d_all
    for idx in p.plantOrder:
        if(p.costPerKwh[idx]>=p.costPrice):
            net_wins.append([wins-(np.sum(ptcs*p.costPerPlant)+rest*p.costPrice)] + list(ptcs))
            net_wins = np.array(net_wins)
            return list(net_wins[np.argmax(net_wins[:,-1]),:])
        # fit as many plants as possible
        ptcs[idx] = min(int(rest/p.kWhPerPlant[idx]),p.maxPlants[idx])
        rest = rest-ptcs[idx]*p.kWhPerPlant[idx]
        # if the maximum of this type of plants is not reached
        # compute the winnings for overproducing with this plant type
        if(ptcs[idx]<p.maxPlants[idx]):
            over_ptc = ptcs[:]
            over_ptc[idx]+=1
            net_wins.append([wins-np.sum(over_ptc*p.costPerPlant)] + list(over_ptc))

    net_wins.append([wins-(np.sum(ptcs*p.costPerPlant)+rest*p.costPrice)] + list(ptcs))
    net_wins = np.array(net_wins)
    return list(net_wins[np.argmax(net_wins[:,-1]),:])

def initPopulation(n_agents,maxPrices):
    pop = np.random.rand(n_agents,len(maxPrices))
    for i in range(len(maxPrices)):
        pop[:,i] *= maxPrices[i]
    return pop

def optimize(problem,
    n_agents=50,
    max_iter=10000,
    scaling_factor=1,
    crossover_rate=.25,
    verbose=0):

    population = initPopulation(n_agents, problem.maxPrices)
    profit_hist = []
    max_profit = 0
    best_x = None
    stop_crit = 0
    for iter in range(max_iter):
        profits = []
        # for each agent
        for ax in range(n_agents):
            # agent to be optimized/mutated (->target vector)
            x = population[ax,:]
            # four other agents
            tmp = np.random.permutation(n_agents)[:4]
            # take three that are not x (to ultimately generate the donor vector)
            a,b,c = tmp[tmp!=ax][:3]
            # a will be the base vector, and the difference will be generated from b and c
            donor_direction = population[b,:]-population[c,:]
            ## potential new agent
            Y = x.copy()
            # take random indice
            R = np.random.randint(population.shape[1])
            # take other random indices
            i = np.random.rand(population.shape[1])<=crossover_rate #yields array of bools
            # trial generation
            y = population[a,:] + scaling_factor * donor_direction

            Y[R] = y[R]
            Y[i] = y[i]
            # reset below zero values
            Y[Y<0] = x[Y<0]
            # reset to large prices
            Y[Y>problem.maxPrices] = x[Y>problem.maxPrices]
            # check old and new profits
            p0 = comp_winnings(problem,x)[0]
            p1 = comp_winnings(problem,Y)[0]
            if p1 > p0:
                population[ax,:] = Y
                profits.append(p1)
            else:
                profits.append(p0)
        m = np.max(profits)
        mx = population[np.argmax(profits),:]
        profit_hist.append(m)
        if(m>max_profit):
            max_profit=m
            best_x = mx
            stop_crit = 0
        else:
            stop_crit += 1
        if(stop_crit==300):
            iters = iter
            print('Converged after iteration: {0:7d} | profit: {1:12.2f}'.format(iter,m))
            res = comp_winnings(problem,best_x)
            print(best_x)
            print(res)
            return profit_hist
        if iter % 100 == 0 and verbose>=1:
            print('iteration: {0:7d} | profit: {1:12.2f}'.format(iter,m))
    iters = max_iter
    print('iteration: {0:7d} | profit: {1:12.2f}'.format(iters,m))
    res = comp_winnings(problem,best_x)
    print(best_x)
    print(res)
    return profit_hist

def get_best_example1():
    # Just go through a gird of possible prices
    # fine grained by previous rougher grid runs
    s1 = np.arange(0.307,0.309,0.0001)
    s2 = np.arange(0.19542,.19544,0.000001)
    s3 = np.arange(0.16810,.16812,0.000001)
    print(len(s1)*len(s2)*len(s3))

    idx=0
    prev_max = 0
    result = None
    # go thorugh the grid
    for p1 in s1:
        for p2 in s2:
            for p3 in s3:
                # compute demands from prices
                d1 = (2000000-(p1*p1)*2000000/(0.45*0.45))
                d2 = (30000000-(p2*p2)*30000000/(0.25*0.25))
                d3 = (20000000-(p3*p3)*20000000/(0.2*0.2))
                # total demand
                d_all = d1+d2+d3
                # compute winnings
                wins = d1*p1+d2*p2+d3*p3
                # compute the distribution on plant types for the best profit
                pt1,pt2,pt3,max_win = get_max_win(d_all, wins)
                # find best profit
                if(max_win>prev_max):
                    result = [max_win,pt1,pt2,pt3,d1,d2,d3,p1,p2,p3]
                    prev_max = max_win
                idx+=1

    print(result)

if __name__ == "__main__":
    # numerically compute best solution for example one
    # get_best_example1()
    p1 = Problem(maxPrices=[.45,.25,.2], maxDemands=[2e6,3e7,2e7],costPrice=.6)
    p2 = Problem(maxPrices=[.45,.25,.2], maxDemands=[2e6,3e7,2e7],costPrice=.1)
    p3 = Problem(maxPrices=[.5,.3,.1], maxDemands=[1e6,5e6,5e6],costPrice=.6)
    optimize(problem=p1, n_agents=50, max_iter=10000, scaling_factor = 0.8, crossover_rate = .25, verbose=1)


"""
Example 2
costPerKwh:
PlantType_1 = 0.2
PlantType_2 = 0.13333333
PlantType_3 = 0.1
Buying = 0.1
-> no need to produce -> costPerKwh fixed to 0.1

demand_i = (maxDemand_i-price_i²*maxDemand_i/maxPrice_i²)
netWinPerKwh_i = (price_i-costPerKwh)
win = sum_i(demand_i*netWinPerKwh_i)
-> derivative is seperable -> easy solution

res_2 = (2000000-0.295271²*2000000/0.45²)*(0.295271-0.1) + (30000000-0.18147²*30000000/0.25²)*(0.18147-0.1) + (20000000-0.153518²*20000000/0.2²)*(0.153518-0.1) = 1818406.11077
"""
