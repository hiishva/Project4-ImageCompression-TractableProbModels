from __future__ import print_function
import numpy as np
import sys
import os
import time
from Util import *
from CLT_class import CLT
from sklearn.utils import resample
## MIXTURE OF TREES BAYESIAN NETWORKS USING RANDOM FOREST ##

class RndmFrst():
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks
    
    #Mixture of tree learning
    def learn(self, dataset, n_components=2, r=100):
        weights=np.zeros((n_components, dataset.shape[0]))
        self.n_components = n_components
        # Randomly initialize the chow-liu trees and the mixture probabilities
        self.mixture_probs = [1/n_components] * n_components
        self.clt_list = [CLT() for i in range(n_components)]

        for i in range(n_components):
            btstrpData = resample(dataset)
            self.clt_list[i].learn(btstrpData)
            self.clt_list[i].rfUpdate(btstrpData, weights[i],r)
    
    #Calculate the Log Likelihood for the dataset
    def computeLL(self,dataset):
        ll = 0.0

        for i in range(self.n_components):
            ll += self.clt_list[i].computeLL(dataset)
        
        return ll
if __name__=='__main__':
    datasetsNames = ['accidents', 'baudio', 'bnetflix', 'jester', 'kdd', 'msnbc', 'nltcs', 'plants', 'pumsb_star', 'tretail'] #Filenames
    k = [2, 5, 10, 20]