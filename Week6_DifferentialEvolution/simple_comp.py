import numpy as np

def get_max_win(d_all, wins):
    # go through plant types sorted after costPerKwh - increasing
    p_wins = np.zeros((4,4))
    # fit as many plants as possible
    pt1 = min(int(d_all/4000000),3)
    # substact the satisfied demand
    rest1 = d_all-pt1*4000000
    # if the maximum of this type of plants is not reached
    # compute the winnings for overproducing with this plant type
    if(pt1<3):
        p_wins[0,:] = [pt1+1,0,0,wins-400000*(pt1+1)]
    # continue likewise with the unsatisfyed demand and the next two plant types

    pt2 = min(int(rest1/600000),50)
    rest2 = rest1-pt2*600000
    if(pt2<50):
        p_wins[1,:] = [pt1,pt2+1,0,wins-(400000*pt1+80000*(pt2+1))]

    pt3 = min(int(rest2/50000),100)
    rest3 = rest2-pt3*50000
    if(pt3<100):
        p_wins[2,:] = [pt1,pt2,pt3+1,wins-(400000*pt1+80000*pt2+10000*(pt3+1))]

    # fill up with buying the remainders
    p_wins[3,:] = [pt1,pt2,pt3,wins-(400000*pt1+80000*pt2+10000*pt3+rest3*0.6)]
    # return the best distribution from the 3 overproducing and the last buying sceanrio
    return list(p_wins[np.argmax(p_wins[:,3]),:])

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

def comp_winnings(p1,p2,p3):
    # compute demands from prices
    d1 = (2000000-(p1*p1)*2000000/(0.45*0.45))
    d2 = (30000000-(p2*p2)*30000000/(0.25*0.25))
    d3 = (20000000-(p3*p3)*20000000/(0.2*0.2))
    # total demand
    d_all = d1+d2+d3
    # compute winnings
    wins = d1*p1+d2*p2+d3*p3
    # compute the distribution on plant types for the best profit
    return get_max_win(d_all, wins)

def initPopulation(n_agents,mp1,mp2,mp3):
    pop = np.random.rand(n_agents,3)
    pop[:,0] *= mp1
    pop[:,1] *= mp2
    pop[:,2] *= mp3
    return pop


def optimize_p1(n_agents=50, max_iter=10000, scaling_factor = 1, crossover_rate = .25, verbose=0):
    max_prices = np.array([0.45,0.25,0.2])
    population = initPopulation(n_agents, 0.45,0.25,0.2)
    profit_hist = [0]
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
            Y[Y>max_prices] = x[Y>max_prices]
            # check old and new profits
            p0 = comp_winnings(x[0], x[1], x[2])[3]
            p1 = comp_winnings(Y[0], Y[1], Y[2])[3]
            if p1 > p0:
                population[ax,:] = Y
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
            iters = iter
            print('Converged after iteration: {0:7d} | profit: {1:12.2f}'.format(iter,m))
            return profit_hist
        if iter % 100 == 0 and verbose>=1:
            print('iteration: {0:7d} | profit: {1:12.2f}'.format(iter,m))
    iters = max_iter
    return profit_hist

if __name__ == "__main__":
    # numerically compute best solution for example one
    # get_best_example1()
    optimize_p1(n_agents=50, max_iter=10000, scaling_factor = 0.8, crossover_rate = .25, verbose=1)


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
