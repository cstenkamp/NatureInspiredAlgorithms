# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:47:00 2017

@author: csten_000
"""
import numpy as np
import matplotlib.pyplot as plt

dist = np.linspace(0,100,1000)
H=(np.exp(- dist ** 2 / (2. * 10 ** 2)))
#H = 1/dist

plt.figure()
axes = plt.gca()
axes.set_ylim([0,1])
first, = plt.plot(dist,H, label="**1")
legends=[first]


for alpha in range(2,7):
    I = H**alpha
    leg, = plt.plot(dist,I,label="**"+str(alpha))
    legends.append(leg)
    
plt.title('Gaussian Heat Kernel')
plt.legend(handles=legends)
plt.show()