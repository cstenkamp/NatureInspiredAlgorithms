# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:19:04 2017

@author: csten_000
"""

import csv
from copy import deepcopy
import time
import numpy as np
np.random.seed()

use_NBAset = True
FirstChoice = False
SmallNeighbor = True
NBAsetpath = "./nba.csv"
show_everything = False
HOWOFTEN = 200


def read_file(file):
    lines = []
    with open(file, "r", newline="") as file:
        spamreader = csv.reader(file, delimiter=',')
        firstline = True
        for row in spamreader:
            if firstline:
                firstline = False
                continue
            lines.append(row)
        lines = np.array(lines)
        names = [i[0] for i in lines]
        weights = [int(i[1]) for i in lines]
        values = [int(i[2]) for i in lines]
        names_dict = dict(zip(list(range(len(names))),names))
        weights_dict = dict(zip(list(range(len(names))),weights))
        values_dict = dict(zip(list(range(len(names))),values))
        return names_dict, weights_dict, values_dict


if not use_NBAset:
    AMOUNT_ITEMS = 10
    WEIGHT_BOUNDS = [0, 50]
    VALUE_BOUNDS = [10, 500]
    MAX_SIZE = VALUE_BOUNDS[1]*4
    
    weight_list = np.random.randint(*WEIGHT_BOUNDS, AMOUNT_ITEMS)
    print("WEIGHTS", weight_list)
    value_list = np.random.randint(*VALUE_BOUNDS, AMOUNT_ITEMS)
    print("VALUES", value_list)
else:
    names, weights, values = read_file(NBAsetpath)
    MAX_SIZE = round(5*np.mean(np.array(list(weights.values()))))

print("Max size", MAX_SIZE)

#val_weight_order = np.argsort(np.array(list(values.values())) / np.array(list(weights.values())))

val_weight_dict = dict(zip(list(range(len(values.values()))),np.array(list(values.values())) / np.array(list(weights.values()))))
#vw_reverse = dict(zip(val_weight_dict.values(), val_weight_dict.keys()))

tmp = list(val_weight_dict.items())
tmp.sort(key=lambda x: x[1], reverse=True)
ratio_order = [i[0] for i in tmp]

#print([names[i] for i in ratio_order])

def contents(knapsack):
    return [i for i, x in enumerate(knapsack) if x]

def calculate_weight(indices):
    return np.sum([weights[i] for i in indices])

def calculate_value(indices):
    return np.sum([values[i] for i in indices])

def fill_random():
    knapsack = np.zeros(len(weight_list))
    all_indices = list(range(len(weight_list)))
    while True:
        rand = np.random.choice(all_indices)
        if calculate_weight(contents(knapsack) + [rand]) <= MAX_SIZE:
            knapsack[rand] = 1
            all_indices.remove(rand) 
            #print(calculate_weight(contents(knapsack)))
        else:
            break
    return knapsack

def print_names(indices):
    return [names[i] for i in indices]

def print_ratios(indices):
    return [val_weight_dict[i] for i in indices]

def get_lowest_ratio(indices,which=0):
    ratios = [val_weight_dict[i] for i in indices]
    tmp = np.argsort(ratios)
    return indices[tmp[which]]


bestofall = []

for wat in range(4):
    
    if wat == 0:
        FirstChoice = False
        SmallNeighbor = True   
        print("Best Choice, Small neighborhood")
    elif wat == 1:
        FirstChoice = False
        SmallNeighbor = False   
        print("Best Choice, Big neighborhood")
    elif wat == 2:
        FirstChoice = True
        SmallNeighbor = True   
        print("First Choice, Small neighborhood")
    elif wat == 3:
        FirstChoice = True
        SmallNeighbor = False   
        print("First Choice, Big neighborhood")


    alliters = []
    allresults = []
    allgesTimes = []
    allIterTimes = []
    
    for _ in range(HOWOFTEN):
        knapsack = fill_random()
        gesT1 = time.time()
        tmp_IterTimes = []
        
        knap = np.zeros(1)
        iters = 0
        while not np.all(knapsack == knap):
            knap = deepcopy(knapsack)
            candidates = []
            iters += 1
            iterT1 = time.time()
            if show_everything:
                print("Players:",print_names(contents(knapsack)))
                print("Value:  ",calculate_value(contents(knapsack)))
                print("Weight: ",calculate_weight(contents(knapsack)))
                print("Ratios: ",print_ratios(contents(knapsack)))
                print()
            
            highestVal = calculate_value(contents(knapsack))
            nr = len(contents(knapsack)) # nr = 1 if SmallNeighbor else len(contents(knapsack))
            for j in range(nr):
                knapsack = deepcopy(knap)
                highestIte = original = get_lowest_ratio(contents(knapsack), j) #nimmt das j-kleinste (bei SmallNeighbor nur das eine)
                knapsack[original] = 0 #take item with lowest value-weight ratio out of the knapsack
                for i in range(len(names)):
                    if calculate_weight(contents(knapsack) + [ratio_order[i]]) <= MAX_SIZE and knapsack[ratio_order[i]] == 0:
                        if calculate_value(contents(knapsack) + [ratio_order[i]]) > highestVal:
                            highestVal = calculate_value(contents(knapsack) + [ratio_order[i]])
                            highestIte = ratio_order[i]
                            if FirstChoice:
                                knapsack[highestIte] = 1
                                break
                        knapsack[highestIte] = 1
                candidates.append((knapsack, calculate_value(contents(knapsack))))
                if SmallNeighbor:
                    break
            candidates.sort(key=lambda x: x[1], reverse=True)
            knapsack = candidates[0][0]
            iterT2 = time.time()
            tmp_IterTimes.append(iterT2-iterT1)
        
        gesT2 = time.time()
        allgesTimes.append((gesT2-gesT1)*1000)
        allIterTimes.append(np.mean(tmp_IterTimes)*1000)
        alliters.append(iters)
        allresults.append(calculate_value(contents(knapsack)))
        
    print("  Mean Iterations:",np.mean(alliters))
    print("  Mean max-val:",np.mean(allresults))    
    print("  Mean Overall Time:",np.mean(allgesTimes))
    print("  Mean Time per Iter:",np.mean(allIterTimes))

    bestofall.append(np.max(allresults))

print("Best of all:",np.max(bestofall))



#print(calculate_weight(contents(knapsack)))

