# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:17:43 2017

@author: csten_000
"""
import numpy as np
from copy import deepcopy
np.set_printoptions(threshold=10000)

DEFAULT_POPULATIONSIZE = 500
NUM_ITERATIONS = 500
DEFAULT_NUMCHILDREN = DEFAULT_POPULATIONSIZE
DEFAULT_MUTA_PROB = 0.005
PARENTS_MUST_DIFFER = True

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

##############################################################################

class Generator():
    def __init__(self, benchmark = 0, numMachines = 20, numJobs = 300):
        self.benchmark = benchmark
        self.numMachines = numMachines
        self.numJobs = numJobs
    def generateProblem(self):
        if self.benchmark == 0:
            self.jobtimes = np.random.randint(1, 1000, self.numJobs)
        elif self.benchmark == 1:
            self.numMachines = 20
            self.numJobs = 300
            self.jobtimes = np.append(np.random.randint(10,1000,200),(np.random.randint(100,300,100)))
        elif self.benchmark == 2:
            self.numMachines = 20
            self.numJobs = 300
            self.jobtimes = np.append(np.random.randint(10,1000,150),(np.random.randint(400,700,150)))
        elif self.benchmark == 3:
            self.numMachines = 50
            self.numJobs = 101
            self.jobtimes = np.array([50])
            for i in range(50,100):
                self.jobtimes = np.append(self.jobtimes, [i,i])
        elif self.benchmark == 4:
            self.numMachines = 3
            self.numJobs = 10
            self.jobtimes = np.array([1,2,3,3,3,3,3,8,9,10])
        return self.jobtimes, self.numMachines
        
        
        
    
class Initializer():
        def __init__(self):
            self.popsize = DEFAULT_POPULATIONSIZE
        def initializePopulation(self, numMachines, numJobs, popsize=False):
            if popsize:
                self.popsize = popsize    
            #Vector mit länge #jobs, jeder wert hat die domain {1..#machines}
            #Vector_j == k iff job j is assigned to machine k
            genomes = np.random.randint(0, high=numMachines, size=(self.popsize, numJobs))
            return genomes
    
    
    
class FitnessEvaluator():
    def __init__(self):
        pass
    
    def evaluate(self, genomes, jobtimes, numMachines):
        #Vector mit länge #jobs, jeder wert hat die domain {1..#machines}
        allCosts = []
        for currIndiv in genomes:
#            print("Current Individual:",currIndiv)
            costs = [0]*numMachines
            for currMachine in range(numMachines):
                currMachineJobs = np.where(currIndiv == currMachine)[0]
                costs[currMachine] = np.sum(jobtimes[currMachineJobs])
            allCosts.append(max(costs))
        return allCosts



class Crossover(): #1-point cross-over, selected by roulette-wheel 
    def __init__(self):
        self.num_children = DEFAULT_NUMCHILDREN
    
    def select_parents(self, population, values):
        chances = softmax([-i for i in values])
#        cum = [np.sum(chances[:i+1]) for i in range(len(chances))] #could make a real roulette-wheel like this
        parents = []       
        for i in range(self.num_children):
            parent1 = np.random.choice(np.arange(len(chances)),p=chances)
            p2values = deepcopy(values)
            if PARENTS_MUST_DIFFER:
                p2values[parent1] = float("inf")
            p2chances = softmax([-i for i in p2values])
            parent2 = np.random.choice(np.arange(len(chances)),p=p2chances)
            parents.append((parent1, parent2))
        return parents
        
    def crossover(self, population, parentlist):
        children = []
        for i in parentlist:
#            print(population[i[0]], population[i[1]])
            crossoverpunkt = np.random.randint(len(population[0]))
            if np.random.random() > 0.5: #die chance das parent1 das superhäufige ist ist viel höher, daher muss das auch mal hinten sein!
                child = np.append(population[i[0]][:crossoverpunkt], population[i[1]][crossoverpunkt:])
            else:
                child = np.append(population[i[1]][:crossoverpunkt], population[i[0]][crossoverpunkt:])
            children.append(child)
        children = np.array([list(i) for i in children])
        return children
            
        
    def select_and_crossover(self, population, values):
        parentlist = self.select_parents(population, values)
        children = self.crossover(population, parentlist)
        return children
    
    
    
            
class Mutator(): #random
    def __init__(self):
        self.mutateprob = DEFAULT_MUTA_PROB
    
    def mutate(self, population, numMachines):
        for i in range(len(population)):
            for j in range(len(population[i])):
                if np.random.random() < self.mutateprob:
                    population[i][j] = np.random.randint(numMachines)
        
        return population




    
class Selector: #delete-all
    def __init__(self):
        pass
    
    def nextGeneration(self, parents, children):
        return children
        



class GeneticAlgorithm():
    generator = Generator(benchmark=3)
    initializer = Initializer()
    evaluator = FitnessEvaluator()
    crossover = Crossover()
    mutator = Mutator()
    selector = Selector()
    
    
    jobtimes, numMachines = generator.generateProblem()
    
    print("SUM Jobtimes", np.sum(jobtimes))
    population = initializer.initializePopulation(numMachines, len(jobtimes))

    for i in range(NUM_ITERATIONS):
#        print("PARENTS", population)
        values = evaluator.evaluate(population, jobtimes, numMachines)
#        print("Values of the individuals", values)
        if i % 20 == 0:
            print("Generation", i)
            print("Best Individual:", np.min(values))
        children = crossover.select_and_crossover(population, values)
#        print("CHILDREN", children)
        children = mutator.mutate(children, numMachines)
#        print("MUTATED CHILDREN", children)
        population = selector.nextGeneration(population, children)
    

if __name__ == '__main__': 
    GeneticAlgorithm()
    
    
    
    
    
    
    
    
    
    
    
    
    