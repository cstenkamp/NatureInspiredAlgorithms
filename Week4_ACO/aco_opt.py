import numpy as np
from multiprocessing import Process
from time import time
import csv

NUM_NOCHANGEITERS_FOR_CONVERGENCE = 20

def solution_construction(probMat, pheroMat, costs, result, q=0):
    S = np.ma.make_mask(np.ones(len(probMat)), shrink=False) # boolean mask indicating which cities can still be selected
    start = int(np.random.randint(0,len(probMat))) # save starting position
    i = start # current position
    only_pher = np.random.rand(len(probMat)-1) < q #its more efficient to calculate them all in the beginning than in the loop

    for c in range(len(probMat)-1):
        S[i] = False # remove i from set of possible next cities
        if(only_pher[c]): # check if only peromone is considered
            k = np.argmax(pheroMat[i][S]) # maximum pheromone value given available cities
        else:
            A = probMat[i][S] # list of probabilities for available cities
            A /= A.sum() # probabilities need to sum up to 1
            k = int(np.random.choice(len(A), 1, p=A)) # next city chosen from masked probability matrix
        idx, = np.nonzero(S) # indices of available cities
        j = idx[k] # index in unmasked probability matrix
        result[i,j] = costs[i,j]
        i = j # j becomes current position
    result[i,start] = costs[i,start]


def probability_matrix(pheroMat, heuriMat, alpha, beta):
    A = pheroMat
    B = heuriMat
    if(alpha>1):
        A = pheroMat ** alpha
    if(beta>1):
        B = heuriMat ** beta
    if(alpha==0):
        return normalize(B)
    elif(beta==0):
        return normalize(A)
    else:
        return normalize(A*B)

def normalize(M):
    return M / M.sum(axis=1, keepdims=True)

def softmax(X):
    return np.exp(X)/np.sum(np.exp(X))


def ACO(costMat, trials=1, iterations=100, alpha=1, beta=1, num_ants=10, p_ants=1, tau=10, evap_rate=0.1, intensification=1, q=0.2, verbose=0, apply_softmax=False):
    '''
    costMat:  TSP cost matrix
    num_ants: number of ants per iteration
    p_ants: number of ants allowed to increase pheromones per iteration
    trials: number of trials done for this param setup
    iterations: number of ant circuits
    tau: initial pheromone value > 0

    alpha: influence of pheromone
    beta: influence of heuristic
    evap_rate: evaporation rate
    intensification: amount of pheromone added during intensification
    q: probability to follow the strongest trail 0 <= q <= 1
    '''

    best_path = np.full_like(costMat, np.inf)
    best_solution = np.inf 
    trial_bests = []
    trial_times = []
    trial_iters = np.ones(trials)*iterations
    for i in range(1,trials+1):
        s_time = time()
        #### INITIALISATION ####
        pheroMat = np.full_like(costMat, tau) # pheromone matrix all equal, with pheroMat[i,j] > 0
        heuriMat = 1/costMat # heuristic as the inverse of each cost value (high=good)
        np.fill_diagonal(heuriMat,0) # diaogonal not needed
        probMat = probability_matrix(pheroMat, heuriMat, alpha, beta) # probability matrix
        soluMat = np.empty((num_ants,len(costMat),len(costMat)), dtype=np.float64) # solution matrix for every ant

        #### ITERATION ####
        no_change = 0
        l_pi = 0
        for j in range(iterations):
            pi_trial = np.inf
            soluMat.fill(0) # ants start anew
            #### MULTIPROCESSING
            proc = np.empty((num_ants),dtype=Process) # ant processes
            # let ants create solutions
            for k in range(num_ants):
                 p = Process(solution_construction(probMat, pheroMat, costMat, soluMat[k], q))
                 p.start()
                 proc[k] = p
            for p in proc:
                p.join()

            # sort solutions
            N = list(map(lambda x: x.sum(), soluMat))
            soluMat = np.array([y for y,_ in sorted(zip(soluMat,N), key=lambda x: x[1])])

            # update best solution if possible
            pi_c = soluMat[0].sum()
            if(pi_c < best_solution):
                best_path = soluMat[0].copy()
                best_solution = pi_c
            if(pi_c < pi_trial):
                pi_trial = pi_c
            # check if converged
            if(pi_c == l_pi):
                no_change+=1
            else:
                no_change=0
            if(no_change>=NUM_NOCHANGEITERS_FOR_CONVERGENCE):
                trial_iters[i-1] = j
                if(verbose>=1):
                    print("Solution converged at iter %d!"%(j,))
                break
            l_pi = pi_c

            #### EVAPORATION ####
            pheroMat *= (1-evap_rate)

            #### INTENSIFICATION ####
            if apply_softmax:
                soluMat = ((soluMat>0)+0) * intensification
                # invert distances, so lowest has biggest value
                N_inv = 1./(np.array(N))
                # normalize
                N_inv /= np.max(N_inv)
                # apply softmax to hundredth power (power, since we want to spread out the values)
                sm_n = softmax(N_inv**100)
                for ant_x,s in enumerate(sm_n):
                    k = soluMat[ant_x,:,:] * s * intensification
                    pheroMat += k
            else:
                soluMat[:p_ants][soluMat[:p_ants] > 0] = intensification # substitute costs with intensification rate for p_ants best solutions
                for a in soluMat[:p_ants]:
                    pheroMat += a  # add intensification to pheromone matrix
            # update probability matrix #
            probMat = probability_matrix(pheroMat, heuriMat, alpha, beta)

            if(verbose>=2): #log iteration info
                print("T{trial}; Iteration {iter}: {best}".format(trial=i, iter=j, best=pi_c))
        trial_bests.append(pi_trial) # save best trail result
        trial_times.append(time()-s_time) # save time
        if(verbose>=1): #log trial info
            print("Trial_{trial} - BEST SOLUTION: {best}".format(trial=i, best=pi_trial))
    if(verbose>=0): #log best info
        print("BEST SOLUTION: {best}".format(best=min(trial_bests)))
    return trial_bests, trial_times, trial_iters




def ACO_parameter_search(p_idx=0, filename=None):
    if(filename):
        header = ["alpha", "beta", "ants", "p_ants", "tau", "evap_rate", "inten", "q"]
        header += ["best_res", "mean_res", "std_res", "best_time", "mean_time", "best_iter", "mean_iter"]
        data = [header]

    problems = [np.genfromtxt("1.tsp"),
                np.genfromtxt("2.tsp"),
                np.genfromtxt("3.tsp")]
    
    problem = problems[p_idx]

    trials = 5
    iterations = 400
    alphas_must = [1, 1]
    betas_must = [1, 0]
    alphas_best = [1, 2, 3]
    betas_best = [4, 5, 8]
    # num_ants, p_ants, tau, evap_rate, intensification, q
    param_sets_large = [
        [100, 10 , 10, 0.2, 1, 0.1],
        [100, 10 , 10, 0.2, 2, 0.1],
        [100, 1 , 10, 0.2, 1, 0.1],
        [100, 1 , 10, 0.2, 2, 0.1],
        [50, 5 , 10, 0.2, 1, 0.1],
        [50, 1 , 10, 0.2, 2, 0.1]
    ]
    param_sets_small = []
    for i in range(6):
        n_set = [10, 1]
        n_set.append(np.random.randint(low=2, high=10))
        n_set.append(np.random.uniform(low=0.1, high=0.4))
        n_set.append(np.random.uniform(low=0.8, high=2.5))
        n_set.append(np.random.uniform(low=0.05, high=0.3))
        param_sets_small.append(n_set)

    num_sets = 48
    set_counter = 0
    #BEST alpha and beta, together with BOTH param-set-lists
    for i in range(len(alphas_best)):
        for p_set in param_sets_large+param_sets_small:
            s_time = time()
            set_counter+=1
            row = [alphas_best[i],betas_best[i]] + p_set
            res, r_times, r_iter = ACO(costMat=problem, trials=trials, iterations=iterations, alpha=alphas_best[i], beta=betas_best[i], num_ants=p_set[0], p_ants=p_set[1], tau=p_set[2], evap_rate=p_set[3], intensification=p_set[4], q=p_set[5], verbose=0)
            bi = np.argmin(res)
            row += [res[bi], np.mean(res), np.std(res), r_times[bi], np.mean(r_times), r_iter[bi], np.mean(r_iter)]

            data.append(row)
            print("%d/%d sets done"%(set_counter,num_sets))
            print("last set completed in %.2f seconds"%(time()-s_time))

    #REQUIRED alpha and beta, together with the small(random) param-set
    for i in range(len(alphas_must)):
        for p_set in param_sets_small:
            s_time = time()
            set_counter+=1
            row = [alphas_must[i],betas_must[i]] + p_set
            res, r_times, r_iter = ACO(costMat=problem, trials=trials, iterations=iterations, alpha=alphas_must[i], beta=betas_must[i], num_ants=p_set[0], p_ants=p_set[1], tau=p_set[2], evap_rate=p_set[3], intensification=p_set[4], q=p_set[5], verbose=0)
            bi = np.argmin(res)
            row += [res[bi], np.mean(res), np.std(res), r_times[bi], np.mean(r_times), r_iter[bi], np.mean(r_iter)]

            data.append(row)
            print("%d/%d sets done"%(set_counter,num_sets))
            print("last set completed in %.2f seconds"%(time()-s_time))

    if(filename):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerows(data)




def main():
    # TSP distance matrices
    problems = [np.genfromtxt("1.tsp"),
                np.genfromtxt("2.tsp"),
                np.genfromtxt("3.tsp")]
    
    for i in range(len(problems)):
        #best overall
        ACO(costMat=problems[i], trials=1, iterations=400, alpha=1, beta=4, num_ants=100, p_ants=10, tau=10, evap_rate=0.20, intensification=2, q=0.1, verbose=2)
    
        #best for alpha=1, beta=0
        ACO(costMat=problems[i], trials=1, iterations=400, alpha=1, beta=0, num_ants=100, p_ants=1, tau=2, evap_rate=0.177, intensification=1.5, q=0.25, verbose=2)
        
        #best for alpha=1, beta=1
        ACO(costMat=problems[i], trials=1, iterations=400, alpha=1, beta=1, num_ants=100, p_ants=10, tau=7, evap_rate=0.125, intensification=1.02, q=0.15, verbose=2)



if __name__ == "__main__":
    s = time()
    main()
#    for i in range(3):
#        ACO_parameter_search(p_idx=i, filename="aco_res_set"+str(i+1)+".csv")
    print(time()-s)



'''
RESULTS OF HYPERPARAMETER SEARCH

alpha	beta	ants	p_ants	tau	evap_rate	inten	q	best_res	mean_res
PROBLEM 1 (Nicos optimum: 3632)									
1	4	100	10	10	0.2	2	0.1	3642.0	3696.0
1	4	100	10	10	0.2	1	0.1	3670.0	3730.4
1	4	50	5	10	0.2	1	0.1	3724.0	3766.8
2	5	100	10	10	0.2	2	0.1	3740.0	3777.2
PROBLEM 2 (Nicos Optimum: 2878)									
1	4	100	10	10	0.2	2	0.1	2918.0	2943.4
1	4	100	10	10	0.2	1	0.1	2904.0	2947.2
1	4	50	5	10	0.2	1	0.1	2902.0	2960.2
3	8	100	1	10	0.2	1	0.1	2948.0	2966.6
PROBLEM 3 (Nicos Optimum: 2617)									
1	4	100	10	10	0.2	1	0.1	2662.0	2685.6
1	4	100	10	10	0.2	2	0.1	2668.0	2701.4
1	4	50	5	10	0.2	1	0.1	2663.0	2712.0
2	5	100	10	10	0.2	1	0.1	2668.0	2731.6

'''
