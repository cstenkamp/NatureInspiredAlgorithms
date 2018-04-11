import numpy as np
import matplotlib.pyplot as plt
from time import time
from ant_colony import AntColony
from itertools import permutations

class algo():
    def __init__(self, capacities, transportation_costs, demands, distances, n_chromosomes):

        self.n_trucks = capacities.shape[0]
        self.capacities = capacities
        self.transportation_costs = transportation_costs
        self.demands = demands
        self.distances = distances
        self.fitness_counter = sum(sum(self.distances))*max(transportation_costs)
        self.distances[range(len(self.distances)),range(len(self.distances))] = np.inf
        self.n_chromosomes = n_chromosomes
        self.chromosomes = np.zeros((self.n_trucks,self.demands.shape[0],self.n_chromosomes))

    def brute_force_solution(self, d_idc):
        best_distance = np.inf
        best_path = []
        for p in permutations(d_idc):
            p = [0] + [idx+1 for idx in p]
            p_distance = sum([self.distances[p[i],p[(i+1)%len(p)]] for i in range(len(p))])
            path = [(p[i],p[(i+1)%len(p)]) for i in range(len(p))]
            if(p_distance<best_distance):
                best_distance = p_distance
                best_path = path
        return best_distance, best_path

    def run(self, n_gens=100, v_init=1, v_select=1, v_crossover=2, v_mutate=2, v_replace=2, m_rate=0.01, s_rate=0.1, verbose=0):
        self.initChr(v=v_init)
        # track best result
        self.calcFitnesses()
        self.best_cost = np.min(self.costs)
        self.best_per_generation = [self.best_cost]
        # go through generations
        for generation in range(n_gens):
            t0 = time()
            self.generate_offspring(o_size=self.n_chromosomes,
                                 p_size=int(self.n_chromosomes*s_rate),
                                 m_rate=m_rate,
                                 sv=v_select,
                                 cv=v_crossover,
                                 mv=v_mutate)
            self.replace(rv=v_replace)
            best_in_gen = np.min(self.costs)
            self.best_per_generation.append(best_in_gen)
            if(verbose==1):
                print("s/g: {}".format(time()-t0))
                print("best: {}".format(best_in_gen))
            if(best_in_gen<self.best_cost):
                self.best_cost=best_in_gen
        print("best_cost %f"%(self.best_cost,))

    def submat(self, idc):
        sorted_dests = [0] + [idx+1 for idx in idc]
        sorted_dests.sort()
        rows = np.dstack([sorted_dests]*len(sorted_dests))[0]
        columns = [sorted_dests]*len(sorted_dests)
        return self.distances[rows,columns]

    def calcFitnesses(self):
        '''
        calculate fitness of every chromosome
        '''
        costs = []
        routes = []

        for c in range(self.n_chromosomes):
            cost = 0
            truck_routes = []
            for t in range(self.n_trucks):
                if(len(self.truck_loads[c][t])>0):
                    if(len(self.truck_loads[c][t])<=9):
                        truck_distance, path = self.brute_force_solution(self.truck_loads[c][t])
                    else:
                        truck_distance_mat = self.submat(self.truck_loads[c][t])
                        ac = AntColony(distances=truck_distance_mat, n_ants=20, n_best=10, n_iterations=100, decay=0.95, alpha=1, beta=4)
                        path, truck_distance = ac.run()
                    cost += (truck_distance * self.transportation_costs[t])
                    truck_routes.append(path)
                else:
                    truck_routes.append(())
            costs.append(cost)
            routes.append(truck_routes)
        self.costs = np.array(costs)
        self.fitnesses = (self.fitness_counter - self.costs)+1
        self.routes = routes

    def calcFitnessesOffspring(self):
        '''
        calculate fitness of every chromosome
        '''
        costs = []
        routes = []

        for c in range(np.shape(self.offspring)[2]):
            cost = 0
            truck_routes = []
            for t in range(self.n_trucks):
                if(len(self.offspring_loads[c][t])>0):
                    truck_distance_mat = self.submat(self.offspring_loads[c][t])
                    if(len(self.offspring_loads[c][t])>3):
                        ac = AntColony(distances=truck_distance_mat, n_ants=20, n_best=10, n_iterations=10, decay=0.95, alpha=1, beta=4)
                        path, truck_distance = ac.run()
                    else:
                        truck_distance = sum([truck_distance_mat[i,(i+1)%len(truck_distance_mat)] for i in range(len(truck_distance_mat))])
                        path = [(i,(i+1)%len(truck_distance_mat)) for i in range(len(truck_distance_mat))]
                    cost += (truck_distance * self.transportation_costs[t])
                    truck_routes.append(path)
                else:
                    truck_routes.append(())
            costs.append(cost)
            routes.append(truck_routes)
        self.offspring_costs = np.array(costs)
        self.offspring_fitnesses = (self.fitness_counter - self.costs)+1
        self.routes = routes

    def initChr(self,v=1):
        if v==1:
            '''
            randomly assign demands to machines
            '''
            for n in range(self.n_chromosomes):
                rx = np.random.randint(self.n_trucks,size=self.demands.shape[0])
                for x,r in enumerate(rx):
                    self.chromosomes[r,x,n] = 1
        elif(v==2):
            '''
            Each machine gets same amount of demands
            '''
            n_demands = self.demands.shape[0]
            j_p_m = int(n_demands/self.n_trucks)
            dist = j_p_m*list(np.arange(self.n_trucks))
            missing = self.demands.shape[0]-len(dist)
            if(missing>0):
                dist += list(np.random.randint(0, self.n_trucks, missing))
            dist = np.array(dist)
            for cx in range(self.n_chromosomes):
                np.random.shuffle(dist)
                self.chromosomes[dist,np.arange(n_demands),cx] = 1

        else:
            '''
            random machines are filled up with jobs
            '''
            truck_idc = np.arange(self.n_trucks)
            demand_idc = np.arange(len(self.demands))
            for cx in range(self.n_chromosomes):
                np.random.shuffle(truck_idc)
                np.random.shuffle(demand_idc)
                t_counter = 0
                t_id = truck_idc[t_counter]
                truck_capacity = self.capacities[t_id]
                i=0
                while(i < len(self.demands)):
                    d_id = demand_idc[i]
                    truck_capacity -= self.demands[d_id]
                    if(truck_capacity < 0):
                        t_counter +=1
                        t_id = truck_idc[t_counter]
                        truck_capacity = self.capacities[t_id]
                    else:
                        self.chromosomes[t_id,d_id,cx] = 1
                        i+=1


        self.truck_loads = [[list(np.where(self.chromosomes[i,:,c] == 1)[0]) for i in range(self.n_trucks)] for c in range(self.n_chromosomes)]
        self.redistribute_overloads()

    def redistribute_overloads(self):
        # redistribute every chromosome
        for c in range(self.n_chromosomes):
            # get the current overall demands per truck
            demand_per_truck = [np.sum(self.demands[idc]) for idc in self.truck_loads[c]]
            # fix trucks that have too high load
            for t in np.where(np.array(demand_per_truck)>self.capacities)[0]:
                # as long as load too high
                while(demand_per_truck[t]>self.capacities[t]):
                    # pop random element from truck
                    reset_idx = self.truck_loads[c][t].pop(np.random.randint(len(self.truck_loads[c][t])))
                    # get a random new truck that has space for it
                    new_truck_idx = np.random.choice(np.where(np.array(demand_per_truck)-self.demands[reset_idx]<self.capacities)[0])
                    # update demand per truck
                    demand_per_truck[new_truck_idx] += self.demands[reset_idx]
                    demand_per_truck[t] -= self.demands[reset_idx]
                    # update demand_idc
                    self.truck_loads[c][new_truck_idx].append(reset_idx)
                    # update chromosomes
                    self.chromosomes[t,reset_idx,c] = 0
                    self.chromosomes[new_truck_idx,reset_idx,c] = 1

    def generate_offspring(self, o_size=None, p_size=None, m_rate=0.01, sv=1, cv=1, mv=1):
        if(o_size is None):
            o_size = self.n_chromosomes
        if(p_size is None):
            p_size = int(self.n_chromosomes/2)
        offspring = []

        # select chromosomes for breeding
        parents = self.select(n=p_size,v=sv)
        for i in range(int(o_size/2)):
            p1i, p2i = np.random.choice(parents, 2, True)
            c1,c2 = self.crossover(p1i, p2i, v=cv)
            c1 = self.mutate(c1, p=m_rate, v=mv)
            c2 = self.mutate(c2, p=m_rate, v=mv)
            offspring.append(c1)
            offspring.append(c2)
        self.offspring  = np.dstack(offspring)
        self.offspring_loads = [[list(np.where(self.offspring[i,:,c] == 1)[0]) for i in range(self.n_trucks)] for c in range(self.n_chromosomes)]
        self.redistribute_offspring_overloads()

    def select(self, n=None, v=1):
        if n is None:
            n = int(self.n_chromosomes/2)
        '''
        Returns two chromosomes.
        Selection probability is based on fitness.
        '''
        rouletteSpaces = np.cumsum(self.fitnesses)/np.sum(self.fitnesses)
        # roulette
        if(v==1):
            r = np.random.uniform(size=n)
            c1 = [np.where(rouletteSpaces>=r1)[0][0] for r1 in r]
            return c1
        # advanced roulette
        elif(v==2):
            distance = 1/n
            move = np.random.rand()*distance
            r = np.cumsum(n*[distance]) + move
            r_shift = [r[-1]-1] + list(r[0:-1])
            c1 = [np.where(rouletteSpaces>=r1)[0][0] for r1 in r_shift]
            return c1
        # tournament
        elif(v==3):
            winners = []
            n_contenders = int(len(self.fitnesses)/n)
            c_pool = list(enumerate(self.fitnesses))
            perm = np.random.permutation(len(self.fitnesses))
            c_pool = [c_pool[j] for j in perm]
            for i in range(0,len(c_pool),n_contenders):
                contenders = [c_pool[k] for k in range(i,i+n_contenders)]
                sorted_contenders = sorted(contenders, key=lambda x: x[1], reverse=True)
                winners.append(sorted_contenders[0][0])
            return winners
        else:
            return None

    def crossover(self,c1,c2,v=1):
        # single point crossover
        if(v==1):
            ix = np.random.randint(self.demands.shape[0])
            # temp save chromosomes
            c1_tmp = self.chromosomes[:,:,c1]*1
            c2_tmp = self.chromosomes[:,:,c2]*1
            # splice
            c1_tmp[:,ix:] = self.chromosomes[:,ix:,c2]
            c2_tmp[:,ix:] = self.chromosomes[:,ix:,c1]
            return c1_tmp, c2_tmp
        # uniform crossover
        elif(v==2):
            job_crosses = np.random.rand(self.demands.shape[0])>0.5
            c1_tmp = self.chromosomes[:,:,c1]*1
            c2_tmp = self.chromosomes[:,:,c2]*1
            # cross
            c1_tmp[:,job_crosses] = self.chromosomes[:,job_crosses,c2]
            c2_tmp[:,job_crosses] = self.chromosomes[:,job_crosses,c1]
            return c1_tmp, c2_tmp
        else:
            return None, None

    def mutate(self,C,p=.01,v=1):
        '''
        Return mutated form of input chromosome
        '''
        n_demands = self.demands.shape[0]
        mutation = C.copy()
        # random resetting: mutated gen gets a random new integer
        if v==1:
            mutate_gens = np.random.rand(n_demands)>(1-p)
            muts = np.sum(mutate_gens)
            if(muts>0):
                mutation[:,mutate_gens] = 0
                mutation[np.random.choice(self.n_trucks, muts),mutate_gens] = 1
        # random resetting: add or substract small value
        elif v==2:
            mutate_gens = np.random.rand(n_demands)>(1-p)
            muts = np.sum(mutate_gens)
            if(muts>0):
                m_ints = np.where(mutation==1)[0]
                mutation[:,mutate_gens] = 0
                m_ints = np.add(m_ints[mutate_gens],np.random.randint(-3, 4, muts))
                m_ints = np.mod(m_ints, self.n_trucks)
                mutation[m_ints,mutate_gens] = 1
        return mutation

    def replace(self, rv=1):
        # replace all
        if(rv==1):
            self.calcFitnessesOffspring()
            self.chromosomes = self.offspring
            self.truck_loads = self.offspring_loads
            self.costs = self.offspring_costs
            self.fitnesses = self.offspring_fitnesses
        # replace with the best of all (size stays the same)
        elif(rv==2):
            self.calcFitnessesOffspring()
            self.chromosomes = np.dstack([self.chromosomes, self.offspring])
            self.truck_loads += self.offspring_loads
            self.fitnesses = np.append(self.fitnesses, self.offspring_fitnesses)
            self.costs = np.append(self.costs, self.offspring_costs)
            sorted_chrom = sorted(zip(self.fitnesses,range(len(self.fitnesses))), key=lambda x: x[0], reverse=True)
            take_idc = [sorted_chrom[i][1] for i in range(self.n_chromosomes)]
            print(take_idc)
            self.fitnesses = self.fitnesses[take_idc]
            self.costs = self.costs[take_idc]
            self.chromosomes = self.chromosomes[:,:,take_idc]
            self.truck_loads = [self.truck_loads[i] for i in take_idc]

    def redistribute_offspring_overloads(self):
        # redistribute every chromosome
        for c in range(self.n_chromosomes):
            # get the current overall demands per truck
            demand_per_truck = [np.sum(self.demands[idc]) for idc in self.offspring_loads[c]]
            # fix trucks that have too high load
            for t in np.where(np.array(demand_per_truck)>self.capacities)[0]:
                # as long as load too high
                while(demand_per_truck[t]>self.capacities[t]):
                    # pop random element from truck
                    reset_idx = self.offspring_loads[c][t].pop(np.random.randint(len(self.offspring_loads[c][t])))
                    # get a random new truck that has space for it
                    new_truck_idx = np.random.choice(np.where(np.array(demand_per_truck)-self.demands[reset_idx]<self.capacities)[0])
                    # update demand per truck
                    demand_per_truck[new_truck_idx] += self.demands[reset_idx]
                    demand_per_truck[t] -= self.demands[reset_idx]
                    # update demand_idc
                    self.offspring_loads[c][new_truck_idx].append(reset_idx)
                    # update chromosomes
                    self.offspring[t,reset_idx,c] = 0
                    self.offspring[new_truck_idx,reset_idx,c] = 1

if __name__ == "__main__":
    demands = np.loadtxt("demand.txt")
    distances = np.loadtxt("distance.txt")
    transportation_costs = np.loadtxt("transportation_cost.txt")
    capacities = np.loadtxt("capacity.txt")

    n_trucks = len(capacities)
    # transportation_costs = transportation_costs[-13:]
    # capacities = capacities[-13:]

    A = algo(capacities, transportation_costs, demands, distances, 10)
    A.run(n_gens=100, v_init=3, v_select=1, v_crossover=2, v_mutate=1, v_replace=2, m_rate=0.1, s_rate=0.5, verbose=1)
