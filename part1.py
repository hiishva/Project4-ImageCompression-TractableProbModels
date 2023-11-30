from PIL import Image
import numpy as np
import numpy.matlib
import sys
import os
import random
from kmeansToolBox import updateCluster,calcCompRatio,assignCluster

def main():
    # if len(sys.argv) < 4:
    #     print("Usage: Kmeans <input-image> <k> <output-image>")
    #     return

    # input_image_path = sys.argv[1]
    input_image_path = os.path.join(os.getcwd(),'Penguins.jpg')
    #k = int(sys.argv[2])
    maxIters= 10000
    k = 10
    output_image_path = 'compressedPenguins.jpg'
    #output_image_path = os.getcwd()
    print('outputImage path: {}'.format(output_image_path))
    #output_image_path = sys.argv[3]

    try:
        original_image = Image.open(input_image_path)
        kmeans_image = kmeans_helper(original_image, k,maxIters)
        kmeans_image.save(output_image_path)
        compratio = calcCompRatio(original_image,kmeans_image)
        print(compratio)

    except IOError as e:
        print(e)

def kmeans_helper(original_image, k, maxIters):
    w, h = original_image.size
    print('w: {} and H: {}'.format(w,h))
    kmeans_image = Image.new('RGB', (w, h))

    # Read RGB values from the image
    Origpixels = np.array(original_image).reshape((w * h, 3))
    print("Shape of compressed_pixels before reshaping:", Origpixels.shape)

    # Call kmeans algorithm: update the RGB values
    newPix = kmeans(Origpixels, k, w, h,maxIters)
    
    # Write the new RGB values to the image
    pixels = newPix.reshape((h,w, 3))
    print("Shape of compressed_pixels after reshaping:", pixels.shape)
    kmeans_image = Image.fromarray(np.uint8(pixels))
    wc,hc = kmeans_image.size
    print('wc: {} and Hc: {}'.format(wc,hc))
    return kmeans_image

def kmeans(pixels, k, w, h,maxIters):
    # Your k-means code goes here
    # Update the array pixels by assigning each entry in the pixels array to its cluster center
    
    #clusterCntr = initializeCluster(pixels,k)
    clusterCntr = pixels[np.random.choice(pixels.shape[0], k, replace=False)]
    labels = np.argmin(np.linalg.norm(pixels[:, np.newaxis] - clusterCntr, axis=2), axis=1)
    for i in range (maxIters):
        print('round: {}'.format(i))
        print('clusterCntr: {}'.format(clusterCntr))
        print('labels: {}'.format(labels))
        #Assign Clusters
        print('assigning clusters')
        labels = np.argmin(np.linalg.norm(pixels[:, np.newaxis] - clusterCntr, axis=2), axis=1)

        # Update cluster centers
        print('updating clusters')
        newCntr = np.array([pixels[labels == i].mean(axis=0) for i in range(k)])
        print('newCntrs: {}'.format(newCntr))
        print('labels: {}'.format(labels))

        #Check Convergence
        print('checking for convergence: {}'.format(np.array_equal(newCntr,clusterCntr)))
        if np.array_equal(newCntr,clusterCntr):
            break
        
        clusterCntr = newCntr
    pixels[:] = clusterCntr[labels]
    return pixels


# def kmeans(pixels, k, w, h):
#     # Your k-means code goes here
#     # Update the array pixels by assigning each entry in the pixels array to its cluster center

    
#     clusterCntrs = random.sample(list(pixels),k)
#     for i in range(1000000):
#         #print('round {} in for loop'.format(i))
#         assgmnts = assignCluster(pixels,clusterCntrs) #Assign pixel to nearest cluster

#         newCntrs = updateCluster(pixels,assgmnts, k) #Change the centers
#         print('checking conditions: {}'.format(np.array_equal(newCntrs,clusterCntrs)))
#         if np.array_equal(newCntrs,clusterCntrs): #Check if its at convergence
#             break
#         clusterCntrs = newCntrs

#     finAssgmnts = assignCluster(pixels,clusterCntrs) #Assign pix to fin clusters
#     compPix = np.array([clusterCntrs[clusterId] for clusterId in finAssgmnts])
#     compPix = compPix.reshape((w,h,3))
#     compImg = Image.fromarray(np.uint8(compPix))

#     return compImg
if __name__ == "__main__":
    main()
