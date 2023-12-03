import numpy as np
import pandas as pd
import os
import CLT_class
import MIXTURE_CLT
import Util

# def ReadFromFolder(directoryPath):
#     print('in readfromfolder')
#     testDataset = pd.DataFrame()
#     trainDataset = pd.DataFrame()
#     for fileName in sorted(glob.glob(directoryPath+'/*.txt')):
#         if 'Test' in fileName:
#             testDataset = pd.read_csv(fileName,low_memory=False,header=None)
#             print('added to testdata')
#         elif 'Train' in fileName:
#             trainDataset = pd.read_csv(fileName,low_memory=False,header=None)
#             print('added to train data')
#     return testDataset, trainDataset

dataset=Util.load_dataset(os.getcwd() + '/dataset/baudio.ts.data')   
clt=CLT_class.CLT()
clt.learn(dataset)
print(dataset[0])