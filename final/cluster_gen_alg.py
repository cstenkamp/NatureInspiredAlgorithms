import numpy as np
import matplotlib.pyplot as plt
from ant_colony import AntColony
from sklearn.decomposition import PCA
from time import time
from itertools import permutations
from time import time

def getPositions(Dist):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(Dist)
    return coords

def dist_mat(dep_x,dep_y, mu_x,mu_y, points_x,points_y): # points_x,points_y is the point
    # print("mu {} {}".format(mu_x,mu_y))
    # print("dep {} {}".format(dep_x,dep_y))
    px = mu_x-dep_x
    py = mu_y-dep_y
    something = px*px + py*py
    u =  ((points_x - dep_x) * px + (points_y - dep_y) * py) / float(something)
    u[u>1] = 1
    u[u<0] = 0

    x = dep_x + u * px
    y = dep_y + u * py
    dx = x - points_x
    dy = y - points_y
    dists = np.sqrt(dx*dx + dy*dy)
    return dists

class KMeans():
    def __init__(self, X, depot, capacities, demands, max_iter=100, verbose=0):
        self.X = X
        self.depot = depot
        self.capacities = capacities
        self.demands = demands
        self.max_iter = max_iter
        self.verbose = verbose

    def cluster_points(self):
        # generate demand cluster distance matrix
        for i,m in enumerate(self.mu):
            if (np.array_equal(m, self.depot)):
                self.dist_matrix[:,i] = np.linalg.norm(np.subtract(self.X,m),axis=1)
            else:
                self.dist_matrix[:,i] = dist_mat(self.depot[0],self.depot[1], m[0],m[1], self.X[:,0], self.X[:,1])
        # generate demand cluster mapping
        self.clusters = np.argmin(self.dist_matrix,axis=1)

    def map_trucks_to_clusters(self):
        # assign demands to clusters
        self.cluster_idc = [list(np.where(self.clusters == i)[0]) for i in range(self.K)]
        # get the total demand
        self.cluster_total_demand = [sum(self.demands[c]) for c in self.cluster_idc]
        # order cluster indices at their demand size
        self.cluster_order = np.argsort([-x for x in self.cluster_total_demand])
        # assign biggest cluster to biggest truck
        self.cluster_truck_dict = dict(zip(self.cluster_order,self.truck_order))

    def fit_clusters_to_trucks(self):
        # go throug each truck-cluster
        for c_idx, t_idx in zip(self.cluster_order,self.truck_order):
            # reassign demands while cluster too big for truck
            while(self.cluster_total_demand[c_idx] > self.capacities[self.cluster_truck_dict[c_idx]]):
                # get the demand furthest offcenter
                r_idx = np.argmax(self.dist_matrix[self.cluster_idc[c_idx],c_idx])
                d_idx = self.cluster_idc[c_idx][r_idx]
                found_alternative = False
                # find best alternative
                sorted_alternatives = np.argsort(self.dist_matrix[d_idx,:])[1:]
                for alt_c_idx in sorted_alternatives:
                    if((self.cluster_total_demand[alt_c_idx] + self.demands[d_idx]) <= self.capacities[self.cluster_truck_dict[alt_c_idx]]):
                        # move to best alternative truck
                        self.cluster_total_demand[alt_c_idx] += self.demands[d_idx]
                        self.cluster_idc[alt_c_idx].append(d_idx)
                        self.clusters[d_idx] = alt_c_idx
                        # delete fromm original truck
                        self.cluster_total_demand[c_idx] -= self.demands[d_idx]
                        del self.cluster_idc[c_idx][r_idx]
                        found_alternative = True
                        break
                # get a larger truck
                if(not(found_alternative)):
                    # get other trucks' capacities
                    other_trucks = self.capacities[self.gen[self.K:]]
                    # get those big enough
                    bigger_trucks = np.where(other_trucks >= (self.cluster_total_demand[c_idx]))[0]
                    if(len(bigger_trucks)>0):
                        bt_gen_idx = self.K + bigger_trucks[np.argmin(other_trucks[bigger_trucks])]
                        bt_idx = self.gen[bt_gen_idx]
                        self.gen[np.where(self.gen==t_idx)[0][0]] = bt_idx
                        self.gen[bt_gen_idx] = t_idx
                        self.cluster_truck_dict[c_idx] = bt_idx
                        if(self.verbose>=2):
                            print("Enlarged truck from {} to {}".format(self.capacities[t_idx], self.capacities[bt_idx]))
                    # get a larger truck for an alternative
                    else:
                        swapped_truck = False
                        for alt_c_idx in sorted_alternatives:
                            alt_t_idx = self.cluster_truck_dict[alt_c_idx]
                            bigger_trucks = np.where(other_trucks >= (self.cluster_total_demand[alt_c_idx]))[0]
                            if(len(bigger_trucks)>0):
                                bt_gen_idx = self.K + bigger_trucks[np.argmin(other_trucks[bigger_trucks])]
                                bt_idx = self.gen[bt_gen_idx]
                                self.gen[np.where(self.gen==alt_t_idx)[0][0]] = bt_idx
                                self.gen[bt_gen_idx] = alt_t_idx
                                if(self.verbose>=2):
                                    print("Enlarged other truck-{} from {} to {}".format(alt_c_idx, self.capacities[self.cluster_truck_dict[alt_c_idx]], self.capacities[bt_idx]))
                                self.cluster_truck_dict[alt_c_idx] = bt_idx
                                # move to best alternative truck
                                self.cluster_total_demand[alt_c_idx] += self.demands[d_idx]
                                self.cluster_idc[alt_c_idx].append(d_idx)
                                self.clusters[d_idx] = alt_c_idx
                                # delete fromm original truck
                                self.cluster_total_demand[c_idx] -= self.demands[d_idx]
                                del self.cluster_idc[c_idx][r_idx]
                                swapped_truck = True
                                break

                        # add a truck
                        if(not(swapped_truck)):
                            if(len(self.gen)>self.K):
                                self.K+=1
                                self.mu = np.concatenate([self.mu, self.X[d_idx,:].reshape((1,2))],axis=0)
                                # move to new truck
                                self.cluster_total_demand.append(self.demands[d_idx])
                                self.cluster_idc.append(np.array([d_idx]))
                                self.clusters[d_idx] = K
                                self.cluster_truck_dict[K] = self.capacities[gen[self.K]]
                                # delete fromm original truck
                                self.cluster_total_demand[c_idx] -= self.demands[d_idx]
                                del self.cluster_idc[c_idx][r_idx]
                                if(self.verbose>=2):
                                    print("Added truck with id: {} and capacity {}".format(gen[K],self.capacities[self.gen[self.K]]))
                                break
                            else:
                                print("Buy another Truck!")
                                break

        self.clean_empty_clustes()

    def clean_empty_clustes(self):
        # delete empty clusters
        rm_idx = []
        for c_idx, t_idx in zip(self.cluster_order,self.truck_order):
            # reassign demands while cluster too big for truck
            if(self.cluster_total_demand[c_idx] == 0):
                rm_idx.append(c_idx)
                np.delete(self.truck_order, np.where(self.truck_order==t_idx))
                # print(self.cluster_idc)
                # print(self.cluster_total_demand)
        self.K-=len(rm_idx)
        self.mu = np.delete(self.mu, rm_idx, axis=0)
        self.dist_matrix = np.delete(self.dist_matrix, rm_idx, axis=1)
        if(self.verbose>=2):
            if(len(rm_idx)>0):
                print("removed {}".format(str(rm_idx)))
        # print("jippi")
        # print(self.cluster_idc)
        # input()

    def reevaluate_centers(self):
        self.mu = [np.mean(np.append(self.X[np.where(self.clusters==i)[0],:],np.array([self.depot]),axis=0),axis=0) for i in range(self.K)]

    def has_converged(self):
        return set([tuple(a) for a in self.mu]) == set([tuple(a) for a in self.oldmu])

    def initialize(self):
        self.dist_matrix = np.zeros((len(self.X),self.K))
        self.oldmu = self.X[np.random.randint(len(self.X),size=self.K),:]
        self.mu = self.X[np.random.randint(len(self.X),size=self.K),:]

    def k_means_alg(self, truck_fitting=False):
        i=0
        while not self.has_converged():
            i+=1
            self.oldmu = self.mu
            # Assign all points in X to clusters
            self.cluster_points()
            if(truck_fitting):
                self.map_trucks_to_clusters()
                self.fit_clusters_to_trucks()

            # Reevaluate centers
            self.reevaluate_centers()
            if(i>self.max_iter):
                if(self.verbose>=3):
                    print("Not converged after {} iterations".format(i))
                break
                # print("{}th iteration".format(i))
                # for j in range(len(self.cluster_idc)):
                #     print(self.cluster_truck_dict[j], self.capacities[self.cluster_truck_dict[j]], sum(self.demands[self.cluster_idc[j]]), self.cluster_idc[j])
                # for k in range(len(self.mu)):
                #     print(self.mu[k], self.oldmu[k])
        if(i<=self.max_iter and self.verbose>=3):
            print("Converged after {} iterations".format(i))

    def run(self, K, gen):
        self.K = K
        self.gen = gen
        self.truck_order = self.gen[:self.K][np.argsort(-self.capacities[self.gen[:self.K]])]
        # Initialize to K random centers
        self.initialize()
        self.k_means_alg(truck_fitting=False)
        self.map_trucks_to_clusters()
        # print("Pre")
        # for j in self.cluster_order:
        #     print(self.cluster_truck_dict[j], self.capacities[self.cluster_truck_dict[j]], sum(self.demands[self.cluster_idc[j]]), self.cluster_idc[j])
        # print()
        # input()
        self.ancientmu = self.X[np.random.randint(len(self.X),size=self.K),:]
        self.oldmu = self.X[np.random.randint(len(self.X),size=self.K),:]
        self.k_means_alg(truck_fitting=True)
        # print("Post")
        # for j in self.cluster_order:
        #     print(self.cluster_truck_dict[j], self.capacities[self.cluster_truck_dict[j]], sum(self.demands[self.cluster_idc[j]]), self.cluster_idc[j])
        # print()
        # input()
        return self.cluster_idc, self.gen, self.cluster_truck_dict

class GeneticAlgorithm():
    def __init__(self, capacities, transportation_costs, demands, distances, coords, n_chromosomes, kmeans_iter=20, verbose=0):

        self.n_trucks = capacities.shape[0]
        self.capacities = capacities
        sort_cap_idx = np.argsort(self.capacities)
        self.capacities = self.capacities[sort_cap_idx]
        self.transportation_costs = transportation_costs[sort_cap_idx]
        self.demands = demands
        self.total_demand = sum(demands)
        self.distances = distances
        self.fitness_counter = sum(sum(self.distances))*max(transportation_costs)
        self.n_chromosomes = n_chromosomes
        self.coords = coords
        self.start_figure()
        self.verbose = verbose
        self.km = KMeans(X=self.coords[1:,:], depot=self.coords[0,:], capacities=self.capacities, demands=self.demands, max_iter=kmeans_iter, verbose=verbose)

    def start_figure(self):
        plt.ion()
        self.f = plt.figure(figsize=(24, 13.5))
        self.ax = self.f.add_subplot(111)
        self.ax.scatter(self.coords[1:,0],self.coords[1:,1],alpha=0.8, c="red", edgecolors='none', s=30, label="not_assgined")
        self.ax.scatter(self.coords[0,0],self.coords[0,1], alpha=0.8, c="black", edgecolors='none', s=60, label="depot")
        self.ax.legend()
        plt.title('Truck Routes')
        plt.legend(loc=4)
        self.f.canvas.draw()

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
        return best_path, best_distance

    def run(self, n_gens=100, v_init=1, v_select=1, v_crossover=2, v_mutate=2, v_replace=2, m_rate=0.01, s_rate=0.1):
        self.initChr(v=v_init)
        # track best result
        self.calcFitnesses()
        self.best_cost = np.min(self.costs)
        self.best_per_generation = [self.best_cost]
        # go through generations
        for generation in range(n_gens):
            t0 = time()
            print("Generation: {}".format(generation+1))
            self.generate_offspring(o_size=self.n_chromosomes,
                                 p_size=int(self.n_chromosomes*s_rate),
                                 m_rate=m_rate,
                                 sv=v_select,
                                 cv=v_crossover,
                                 mv=v_mutate)
            self.replace(rv=v_replace)
            best_in_gen = np.min(self.costs)
            self.best_per_generation.append(best_in_gen)
            # print(best_in_gen)
            if(self.verbose>=1):
                print("s/g: {}".format(time()-t0))
                print("best: {}".format(best_in_gen))
            if(best_in_gen<self.best_cost):
                self.best_cost=best_in_gen
                self.printAssignment()
        print("best_cost %f"%(self.best_cost,))

    def submat(self, idc):
        sorted_dests = [0] + [idx+1 for idx in idc]
        sorted_dests.sort()
        rows = np.dstack([sorted_dests]*len(sorted_dests))[0]
        columns = [sorted_dests]*len(sorted_dests)
        submat = self.distances[rows,columns]
        submat[range(len(submat)),range(len(submat))] = np.inf
        return submat

    def calcFitnesses(self):

        costs = []
        routes = []
        for chromosome in self.chromosomes:
            # cum sum of all deliveries
            n_clusters = np.where(np.cumsum([self.capacities[vehicleNr] for vehicleNr in chromosome]) > self.total_demand)[0][0] + 1

            if(self.verbose>=2):
                print("Starting KM")
            cluster_idc, chromosome, cluster_truck_dict = self.km.run(K=n_clusters, gen=chromosome)

            assignment = []
            cost = 0
            for i,c in enumerate(cluster_idc):
                if(len(c)<=9):
                    path, truck_distance = self.brute_force_solution(c)
                else:
                    if(self.verbose>=2):
                        print("Starting ACO")
                    truck_distance_mat = self.submat(c)
                    ac = AntColony(distances=truck_distance_mat, n_ants=20, n_best=10, n_iterations=30, decay=0.95, alpha=1, beta=4)
                    path, truck_distance = ac.run()
                new_cost = int(truck_distance)*self.transportation_costs[int(cluster_truck_dict[i])]
                cost += new_cost
                assignment.append((c,self.transportation_costs[cluster_truck_dict[i]],self.capacities[cluster_truck_dict[i]],new_cost))

            if(self.verbose>=1):
                print("Cost:{} Trucks:{}".format(cost, [(self.capacities[t], sum(self.demands[cluster_idc[k]])) for k,t in cluster_truck_dict.items()]))
            costs.append(cost)
            routes.append(assignment)
        self.costs = np.array(costs)
        self.fitnesses = (self.fitness_counter - self.costs)+1
        self.routes = routes

    def calcFitnessesOffspring(self):

        costs = []
        routes = []
        for chromosome in self.offspring:
            # cum sum of all deliveries
            n_clusters = np.where(np.cumsum([self.capacities[vehicleNr] for vehicleNr in chromosome]) > self.total_demand)[0][0] + 1

            if(self.verbose>=2):
                print("Starting KM")
            cluster_idc, chromosome, cluster_truck_dict = self.km.run(K=n_clusters, gen=chromosome)

            assignment = []
            cost = 0
            for i,c in enumerate(cluster_idc):
                if(len(c)<=9):
                    path, truck_distance = self.brute_force_solution(c)
                else:
                    if(self.verbose>=2):
                        print("Starting ACO")
                    truck_distance_mat = self.submat(c)
                    ac = AntColony(distances=truck_distance_mat, n_ants=20, n_best=10, n_iterations=30, decay=0.95, alpha=1, beta=4)
                    path, truck_distance = ac.run()
                new_cost = int(truck_distance)*self.transportation_costs[int(cluster_truck_dict[i])]
                cost += new_cost
                assignment.append((c,self.transportation_costs[cluster_truck_dict[i]],self.capacities[cluster_truck_dict[i]],new_cost))
            if(self.verbose>=1):
                print("Cost:{} Trucks:{}".format(cost, [(self.capacities[t], sum(self.demands[cluster_idc[k]])) for k,t in cluster_truck_dict.items()]))
            costs.append(cost)
            routes.append(assignment)
        self.offspring_costs = np.array(costs)
        self.offspring_fitnesses = (self.fitness_counter - self.offspring_costs)+1
        self.offspring_routes = routes

    def initChr(self,v=1):
        ''' RANDOM TRUCK ORDER '''
        self.chromosomes = np.array([np.random.permutation(self.capacities.shape[0]) for _ in range(self.n_chromosomes)]).astype(int)

    def generate_offspring(self, o_size=None, p_size=None, m_rate=0.01, sv=1, cv=1, mv=1):
        if(o_size is None):
            o_size = self.n_chromosomes
        if(p_size is None):
            p_size = int(self.n_chromosomes/2)
        offspring = []

        # select chromosomes for breeding
        parents = self.select(n=p_size,v=sv)
        for i in range(int(o_size/2)):
            p1i, p2i = np.random.choice(parents, 2)
            # print(p1i,p2i)
            # print("Parents")
            # print(self.chromosomes[p1i],self.chromosomes[p2i])
            c1,c2 = self.crossover(p1i, p2i, v=cv)
            # print("Crossed")
            # print(c1,c2)
            c1 = self.mutate(c1, p=m_rate, v=mv)
            c2 = self.mutate(c2, p=m_rate, v=mv)
            # print("Mutated")
            # print(c1,c2)
            offspring.append(c1)
            offspring.append(c2)
            # input()
        self.offspring  = np.stack(offspring)

    def select(self, n=None, v=1):
        if n is None:
            n = int(self.n_chromosomes/2)
        '''
        Returns two chromosomes.
        Selection probability is based on fitness.
        '''
        rouletteSpaces = np.cumsum(self.fitnesses)/np.sum(self.fitnesses)
        if(v==1):
            ''' ROULETTE '''
            r = np.random.uniform(size=n)
            c1 = [np.where(rouletteSpaces>=r1)[0][0] for r1 in r]
            return c1
        elif(v==2):
            ''' FITTED DISTANCE ROULETTE '''
            distance = 1/n
            move = np.random.rand()*distance
            r = np.cumsum(n*[distance]) + move
            r_shift = [r[-1]-1] + list(r[0:-1])
            c1 = [np.where(rouletteSpaces>=r1)[0][0] for r1 in r_shift]
            return c1
        elif(v==3):
            ''' TOURNAMENT '''
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
        assert len(self.chromosomes[c1])==self.n_trucks and len(self.chromosomes[c2])==self.n_trucks, "Pre Crossover"
        assert len(np.unique(self.chromosomes[c1]))==self.n_trucks and len(np.unique(self.chromosomes[c2]))==self.n_trucks, "Pre Crossover Unique"

        if(v==1):
            ''' ZIP FRONT-FRONT'''
            # temp save chromosomes
            c1_tmp = []
            c2_tmp = []
            # splice
            for i in range(self.n_trucks):
                if(not(self.chromosomes[c1][i] in c1_tmp)):
                    c1_tmp.append(self.chromosomes[c1][i])
                if(not(self.chromosomes[c2][i] in c1_tmp)):
                    c1_tmp.append(self.chromosomes[c2][i])

                if(not(self.chromosomes[c2][i] in c2_tmp)):
                    c2_tmp.append(self.chromosomes[c2][i])
                if(not(self.chromosomes[c1][i] in c2_tmp)):
                    c2_tmp.append(self.chromosomes[c1][i])
        else:
            ''' ZIP FRONT-BACK'''
            # temp save chromosomes
            c1_tmp = []
            c2_tmp = []
            # splice
            for i in range(len(self.chromosomes[c1])):
                if(not(self.chromosomes[c1][i] in c1_tmp)):
                    c1_tmp.append(self.chromosomes[c1][i])
                if(not(self.chromosomes[c2][-(i+1)] in c1_tmp)):
                    c1_tmp.append(self.chromosomes[c2][-(i+1)])

                if(not(self.chromosomes[c2][i] in c2_tmp)):
                    c2_tmp.append(self.chromosomes[c2][i])
                if(not(self.chromosomes[c1][-(i+1)] in c2_tmp)):
                    c2_tmp.append(self.chromosomes[c1][-(i+1)])

        assert len(c1_tmp)==self.n_trucks and len(c2_tmp)==self.n_trucks, "Post Crossover"
        assert len(np.unique(c1_tmp))==self.n_trucks and len(np.unique(c1_tmp))==self.n_trucks, "Post Crossover Unique"
        return c1_tmp, c2_tmp

    def mutate(self,C,p=.01,v=1):
        ''' Return mutated form of input chromosome '''
        mutation = C.copy()
        assert len(mutation)==self.n_trucks, "Pre Mutation"
        assert len(np.unique(mutation))==self.n_trucks, "Pre Mutation Unique"
        if(v==1):
            ''' NEIGHBOUR SWAP '''
            mutate_idc = np.where(np.random.rand(self.n_trucks)>(1-p))[0]
            for m_idx in mutate_idc:
                tmp = mutation[m_idx]
                mutation[m_idx] = mutation[(m_idx+1)%self.n_trucks]
                mutation[(m_idx+1)%self.n_trucks] = tmp
        elif(v==2):
            ''' RANDOM SWAP'''
            mutate_idc = np.where(np.random.rand(self.n_trucks)>(1-p))[0]
            swap_idc = np.random.randint(self.n_trucks,size=len(mutate_idc))
            for m_idx, s_idx in zip(mutate_idc,swap_idc):
                tmp = mutation[m_idx]
                mutation[m_idx] = mutation[s_idx]
                mutation[s_idx] = tmp

        assert len(mutation)==self.n_trucks, "Post Mutation"
        assert len(np.unique(mutation))==self.n_trucks, "Post Mutation Unique"
        return mutation

    def replace(self, rv=1):
        # replace all
        if(rv==1):
            self.calcFitnessesOffspring()
            self.chromosomes = self.offspring
            self.costs = self.offspring_costs
            self.fitnesses = self.offspring_fitnesses
        # replace with the best of all (size stays the same)
        elif(rv==2):
            for chrom in self.offspring:
                assert len(chrom)==self.n_trucks, "Pre Calc"
                assert len(np.unique(chrom))==self.n_trucks, "Pre Calc Unique"
            self.calcFitnessesOffspring()
            for chrom in self.offspring:
                assert len(chrom)==self.n_trucks, "Post Calc"
                assert len(np.unique(chrom))==self.n_trucks, "Post Calc Unique"
            self.chromosomes = np.append(self.chromosomes, self.offspring, axis=0)
            self.fitnesses = np.append(self.fitnesses, self.offspring_fitnesses, axis=0)
            self.costs = np.append(self.costs, self.offspring_costs, axis=0)
            self.routes = self.routes + self.offspring_routes
            take_idc = np.argsort(self.costs)[:self.n_chromosomes]
            self.fitnesses = self.fitnesses[take_idc]
            self.costs = self.costs[take_idc]
            self.chromosomes = self.chromosomes[take_idc]
            self.routes = [self.routes[t_idx] for t_idx in take_idc]

    def printAssignment(self):
        self.ax.clear()
        colors = ["#e6194b","#0082c8","#ffe119","#f58231","#3cb44b","#911eb4","#46f0f0","#f032e6","#d2f53c","#fabebe","#008080","#e6beff","#aa6e28","#fffac8","#800000","#aaffc3","#808000","#ffd8b1","#000080","#808080","#000000"]
        # find cheapest assignment
        idx = np.argmin(self.costs)
        routes = self.routes[idx]
        for a_idx,truck_assignment in enumerate(routes):
            (truck_route,_,cap,cost) = truck_assignment
            demand = sum(self.demands[truck_route])
            tmp = np.array(truck_route)+1
            self.ax.scatter(self.coords[tmp][:,0],self.coords[tmp][:,1],alpha=0.8, c=colors[a_idx%len(colors)], edgecolors='none', s=30, label="{}/{} - {}".format(demand,cap,cost))
        self.ax.scatter(self.coords[0,0],self.coords[0,1], alpha=0.8, c=colors[-1], edgecolors='none', s=60, label="depot")
        self.ax.legend()
        plt.title('Truck Routes')
        plt.legend(loc=4)
        self.f.canvas.draw()
        # self.f.show()

if __name__ == "__main__":
    demands = np.loadtxt("demand.txt")
    distances = np.loadtxt("distance.txt")
    transportation_costs = np.loadtxt("transportation_cost.txt")
    capacities = np.loadtxt("capacity.txt")

    coords = getPositions(distances)

    GA = GeneticAlgorithm(capacities, transportation_costs, demands, distances, coords, 10, verbose=1)
    GA.run(n_gens=100, v_init=1, v_select=1, v_crossover=2, v_mutate=2, v_replace=2, m_rate=0.1, s_rate=0.5)
