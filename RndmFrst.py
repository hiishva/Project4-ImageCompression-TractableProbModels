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
        weights=np.ones((n_components, dataset.shape[0]))
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
    kVal = [2, 5, 10, 20]
    rVal = [100, 150, 250, 500]
    bestLL = 0.0
    bestK = 2
    bestR = 100

    for dsName in datasetsNames:
        for k in kVal:
            print('K Val = {}'.format(k))
            for r in rVal:
                llVal=[]
                print('R Val = {}'.format(r))
                for j in range(5):
                    trainDS = Util.load_dataset(os.getcwd() + '/dataset/' + dsName + '.ts.data')
                    validDS = Util.load_dataset(os.getcwd() + '/dataset/' + dsName + '.valid.data')
                    testDS = Util.load_dataset(os.getcwd() + '/dataset/' + dsName + '.test.data')
                    print('dataset {} has been loaded'.format(dsName))

                    RF = RndmFrst()
                    RF.learn(trainDS, n_components=k,r=r)
                    LL = RF.computeLL(validDS) / validDS.shape[0]
                    llVal.append(LL)

                avgLL = np.mean(llVal)
                print('the average ll with k={} and r={} is {}'.format(k,r,avgLL))
                
                if bestLL == 0:
                    bestLL = avgLL
                if avgLL > bestLL:
                    bestK = k
                    bestR = r
                    bestLL = avgLL
        tstLL =[]
        for j in range(5):
            RF.learn(trainDS,n_components=bestK,r=bestR)
            TstLL = RF.computeLL(testDS)/ testDS.shape[0]
            tstLL.append(TstLL)
        avgTst = np.mean(tstLL)
        stdTst = np.std(tstLL)
        print('The hyperparameters for dataset {} are: k = {} and r = {}'.format(dsName,bestK, bestR))
        print('Average Log likelihood: {}'.format(avgTst))
        print('Standard Dev of Log likelihood: {}'.format(stdTst))
                    
            
