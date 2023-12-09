from __future__ import print_function
import numpy as np
import sys
import os
import time
from Util import *
from CLT_class import CLT

class MIXTURE_CLT():
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks
        

    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    def learn(self, dataset, n_components=2, max_iter=50, epsilon=1e-5):
        # For each component and each data point, we have a weight
        weights=np.zeros((n_components, dataset.shape[0]))
        self.n_components = n_components
        # Randomly initialize the chow-liu trees and the mixture probabilities
        self.mixture_probs = np.random.random(n_components)
        self.mixture_probs /= self.mixture_probs.sum()

        #Init the Chow-liu trees
        ChowProbs = np.zeros((n_components, dataset.shape[0]))
        self.clt_list = [CLT() for i in range(n_components)]
        for clt in self.clt_list:
            clt.learn(dataset)
        
        # for i in range(n_components):
        #     clt = CLT()
        #     self.clt_list.append(clt)
        #     weights[:, :, i] = np.random.rand(dataset.shape[0], dataset.shape[1])
        #     print(weights.shape[0])
        #     print(dataset.shape[0])

        for itr in range(max_iter):
            for i in range(n_components):
                
                #E-step: Complete the dataset to yield a weighted dataset
                # We store the weights in an array weights[ncomponents,number of points]
                for j,sample in enumerate(dataset):
                    ChowProbs[i][j] = self.clt_list[i].getProb(sample)

                weights[i] = np.multiply(self.mixture_probs[i], ChowProbs[i]) / np.sum(np.multiply(self.mixture_probs[i], ChowProbs[i]))

            # M-step: Update the Chow-Liu Trees and the mixture probabilities
            #Your code for M-Step here
                self.clt_list[i].update(dataset,weights[i])
            
            #Compute the log-likelihood and check for convergence
            if itr == 0:
                curLL = self.computeLL(dataset) / dataset.shape[0] 
            else:
                newLL = self.computeLL(dataset) / dataset.shape[0] 
                if abs(newLL - curLL) < epsilon: #checking for convergence
                    return
                curLL = newLL #set the current Log Likelihood
    '''
    Compute the log-likelihood score of the dataset
    '''
    def computeLL(self, dataset):
        ll = 0.0
        # Write your code below to compute likelihood of data
        #   Hint:   Likelihood of a data point "x" is sum_{c} P(c) T(x|c)
        #           where P(c) is mixture_prob of cth component and T(x|c) is the probability w.r.t. chow-liu tree at c
        #           To compute T(x|c) you can use the function given in class CLT
        l = 0.0 #
        for sample in range(dataset.shape[0]):
            for i in range(self.n_components):
                l += np.multiply(self.mixture_probs[i], self.clt_list[i].getProb(dataset[sample]))
            ll += np.log(l)
        return ll

    
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
if __name__ == '__main__':
    
    datasetsNames = ['accidents', 'baudio', 'bnetflix', 'jester', 'kdd', 'msnbc', 'nltcs', 'plants', 'pumsb_star', 'tretail'] #Filenames
    k = [2, 5, 10, 20]
    for i,dsName in enumerate(datasetsNames):
        llVal =[] #Log Likelihood values
        for j in range(5):
            trainDataset= Util.load_dataset(os.getcwd() + '/dataset/' + dsName + '.ts.data')
            testDataset = Util.load_dataset(os.getcwd() + '/dataset/' + dsName + '.test.data')
            print('Dataset {} has been loaded and beginning processing'.format(dsName))
            

            #Beginning Training on Mixture with CLTs
            MT = MIXTURE_CLT()
            MT.learn(trainDataset, n_components=k[i], max_iter=1, epsilon=1e-1)

            #Beginning Testing on Test dataset
            logLike = MT.computeLL(testDataset) / testDataset.shape[0]
            llVal.append(logLike)
        print('Average Log Likelihood: {}'.format(np.mean(llVal),'.4f'))
        print('Standard Deviation of the Log Likelihood: {}'.format(np.std(llVal),'.4f'))


    # testDataset= Util.load_dataset(os.getcwd() + '/dataset/baudio.test.data')
    # MT = MIXTURE_CLT()
    # MT.learn(trainDataset, n_components=2,max_iter=1,epsilon=1e-1)

    # ll = MT.computeLL(testDataset) / testDataset.shape[0]
    # llVal.append(ll)
    # print('{} dataset -- ll val: {}'.format(testDataset, llVal))


