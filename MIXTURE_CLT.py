
from __future__ import print_function
import numpy as np
import sys
import time
from Util import *
from CLT_class import CLT
from scipy.special import logsumexp

class MIXTURE_CLT():
    
    def __init__(self):
        self.n_components = 2 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks
        

    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    def learn(self, dataset, n_components=2, max_iter=50, epsilon=1e-5):
        # For each component and each data point, we have a weight
        weights=np.zeros((dataset.shape[0], dataset.shape[1], n_components))
        OldLL = 0.0
        # Randomly initialize the chow-liu trees and the mixture probabilities
        for i in range(n_components):
            clt = CLT()
            self.clt_list.append(clt)
            weights[:, :, i] = np.random.rand(dataset.shape[0], dataset.shape[1])
            print(weights.shape[0])
            print(dataset.shape[0])

        for itr in range(max_iter):
            #E-step: Complete the dataset to yield a weighted dataset
            # We store the weights in an array weights[ncomponents,number of points]
            #E-Step:
            weights = self.EStep(dataset, weights)

            # M-step: Update the Chow-Liu Trees and the mixture probabilities
            #Your code for M-Step here
            self.MStep(dataset,weights)

            #Compute LL
            NewLL = self.computeLL(dataset)

            if itr > 0 and abs(NewLL - OldLL) < epsilon:
                break
            OldLL = NewLL
    
        
    def EStep(self, dataset,weights):
        for c in range(self.n_components):
            computeCompLL = self.clt_list[c].computeLL(dataset)
            weights[:, :, c] = np.exp(computeCompLL - logsumexp(computeCompLL))
        return weights
    
    def MStep(self,dataset, weights):
        self.mixture_probs = np.mean(weights,axis=1)
        for c in range(self.n_components):
            self.clt_list[c].update(dataset,weights[:,:,c])
    '''
    Compute the log-likelihood score of the dataset
    '''
    def computeLL(self, dataset):
        ll = 0.0
        # Write your code below to compute likelihood of data
        #   Hint:   Likelihood of a data point "x" is sum_{c} P(c) T(x|c)
        #           where P(c) is mixture_prob of cth component and T(x|c) is the probability w.r.t. chow-liu tree at c
        #           To compute T(x|c) you can use the function given in class CLT
        for c in range(self.n_components):
            compLL = self.clt_list[c].computeLL(dataset)
            print(compLL)
            ll += self.mixture_probs[c] * compLL
            print(ll)
        
        return ll/dataset.shape[0]
    

    
'''
    After you implement the functions learn and computeLL, you can learn a mixture of trees using
    To learn Chow-Liu trees, you can use
    mix_clt=MIXTURE_CLT()
    ncomponents=10 #number of components
    max_iter=50 #max number of iterations for EM
    epsilon=1e-1 #converge if the difference in the log-likelihods between two iterations is smaller 1e-1
    dataset=Util.load_dataset(path-of-the-file)
    mix_clt.learn(dataset,ncomponents,max_iter,epsilon)
    
    To compute average log likelihood of a dataset w.r.t. the mixture, you can use
    mix_clt.computeLL(dataset)/dataset.shape[0]
'''

    
    


    