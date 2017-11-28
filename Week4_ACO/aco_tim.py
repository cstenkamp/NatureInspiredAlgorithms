import numpy as np
from multiprocessing import Process

def solution_construction(P, T, costs, result, q=0):
    S = np.ma.make_mask(np.ones(len(P)), shrink=False) # boolean mask indicating which cities can still be selected
    start = int(np.random.randint(0,len(P))) # save starting position
    i = start # current position
    while(S.sum() > 1):
        S[i] = False # remove i from set of possible next cities
        if(np.random.choice([0,1], 1, p=[1-q, q])): # if max pheromone is chosen directly according to prob q
            k = np.argmax(T[i][S]) # maximum pheromone value given available cities
        else:
            A = P[i][S] # list of probabilities for available cities
            A /= A.sum() # probabilities need to sum up to 1
            k = int(np.random.choice(len(A), 1, p=A)) # next city chosen from masked probability matrix
        idx, = np.nonzero(S) # indices of available cities
        j = idx[k] # index in unmasked probability matrix
        result[i,j] = costs[i,j]
        i = j # j becomes current position
    result[i,start] = costs[i,start]

def probability_matrix(T, H, alpha, beta):
    A = T ** alpha
    B = H ** beta
    return normalize(A*B)

def normalize(M):
    return M / M.sum(axis=1, keepdims=True)

def timeit(method):
    import time
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

@timeit
def main():
    #### HYPERPARAMETERS ####
    # TSP distance matrices
    problems = [np.genfromtxt("1.tsp"),
                np.genfromtxt("2.tsp"),
                np.genfromtxt("3.tsp")]

    m = 10 # number of ants per iteration
    m_bar = 1 # number of ants allowed to increase pheromones per iteration
    trials = 1
    iterations = 10
    tau = 10 # initial pheromone value > 0

    alpha = 1 # influence of pheromone
    beta = 1 # influence of heuristic
    gamma = 10 # gaussian heat kernel parameter for transformation distance matrix into heuristics matrix
    evap_rate = 0.1 # evaporation rate
    intensification = 1 # amount of pheromone added during intensification
    q = 0.2 # probability to follow the strongest trail 0 <= q <= 1
    ###########################

    A = problems[0]

    best_path = np.full_like(A, np.inf)
    pi = np.inf # best solution
    for i in range(1,trials+1):
        #### INITIALISATION ####
        T = np.full_like(A, tau) # pheromone matrix with T[i,j] > 0
        H = (np.exp(- A ** 2 / (2. * gamma ** 2))) # heuristics matrix created by transforming distance matrix with gaussian heat kernel
        np.fill_diagonal(H,0)
        P = probability_matrix(T, H, alpha, beta) # probability matrix
        # M contains a solution matrix for every ant k in the current iteration with
        # M[k]_ij = d_ij if the edge was chosen and 0 otherwise.
        # The length of a path is then the sum of all entries in the matrix M[k]
        M = np.empty((m,len(A),len(A)), dtype=np.float64)
        #### ITERATION ####
        for j in range(iterations):
            M.fill(0)
            proc = np.empty((m),dtype=Process) # ant processes
            # let ants create solutions
            for k in range(m):
                 p = Process(solution_construction(P, T, A, M[k], q))
                 p.start()
                 proc[k] = p
            for p in proc:
                p.join()

            # sort solutions
            N = list(map(lambda x: x.sum(), M))
            M = np.array([y for y,_ in sorted(zip(M,N), key=lambda x: x[1])])

            # update best solution if possible
            pi_c = M[0].sum()
            if(pi_c < pi):
                best_path = M[0].copy()
                pi = pi_c

            #### EVAPORATION ####
            T *= (1-evap_rate)

            #### INTENSIFICATION ####
            M[:m_bar][M[:m_bar] > 0] = intensification # substitute costs with intensification rate for m_bar best solutions
            for a in M[:m_bar]:
                T += a  # add intensification to pheromone matrix

            # update probability matrix #
            P = probability_matrix(T, H, alpha, beta)

            print("T{trial}; Iteration {iter}: {best}".format(trial=i, iter=j, best=pi_c))
    print("BEST SOLUTION: {best}".format(best=pi))

if __name__ == "__main__":
    main()
