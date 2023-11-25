from PIL import Image
import numpy as np
import numpy.matlib
import sys
import os

def assignCluster(pixel,clusterCenter):
    print('in the assignCluster')
    distance  = np.linalg.norm(pixel[:, np.newaxis] - clusterCenter, axis=2)
    assgnmnts = np.argmin(distance,axis=1)
    return assgnmnts

def updateCluster(pixel,assignment, k):
    print('in the updateCluster')
    newCntrs = np.array([pixel[assignment == i].mean(axis=0) for i in range(k)])
    return newCntrs

def calcCompRatio(originalImg,compImg):
    originalSize = os.path.getsize('Koala.jpg')
    compSize = os.path.getsize('compressedKoala.jpg')
    compRatio = originalSize / compSize
    return compRatio