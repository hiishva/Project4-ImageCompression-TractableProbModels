from PIL import Image
import numpy as np
import sys
import os
import random


def main():
    if len(sys.argv) < 4:
        print("Usage: Kmeans <input-image> <k> <output-image>")
        return

    input_image_path_name = sys.argv[1]
    input_image_path = os.path.join(os.getcwd(),input_image_path_name)
    k = int(sys.argv[2])
    maxIters= 10000 #max number of iterations in case it doesn't reach convergence 
    output_image_path = sys.argv[3]

    try:
        original_image = Image.open(input_image_path)
        kmeans_image = kmeans_helper(original_image, k,maxIters)
        kmeans_image.save(output_image_path)
        compratio = CompressionRatio(input_image_path,output_image_path)
        print('The Compression ratio is: {:.4f}'.format(compratio))

    except IOError as e:
        print(e)

def kmeans_helper(original_image, k, maxIters):
    w, h = original_image.size
    kmeans_image = Image.new('RGB', (w, h))

    # Read RGB values from the image
    Origpixels = np.array(original_image).reshape((w * h, 3)) #Original Image


    # Call kmeans algorithm: update the RGB values
    newPix = kmeans(Origpixels, k, w, h,maxIters) #Compressed Pixels
    
    
    # Write the new RGB values to the image
    pixels = newPix.reshape((h,w, 3))
    kmeans_image = Image.fromarray(np.uint8(pixels)) #Compressed Image
    return kmeans_image


def kmeans(pixels, k, w, h,maxIters):
    # Your k-means code goes here
    # Update the array pixels by assigning each entry in the pixels array to its cluster center
    
    #Initialize the Cluster centers randomly
    clusterCntr = pixels[np.random.choice(pixels.shape[0], k, replace=False)]
    labels = np.argmin(np.linalg.norm(pixels[:, np.newaxis] - clusterCntr, axis=2), axis=1)

    #K Means Algorithm
    for i in range (maxIters):
        #print('round: {}'.format(i))
        
        #Assign Clusters
        print('Assigning clusters')
        labels = np.argmin(np.linalg.norm(pixels[:, np.newaxis] - clusterCntr, axis=2), axis=1)

        # Update cluster centers
        print('Updating clusters')
        newCntr = np.array([pixels[labels == i].mean(axis=0) for i in range(k)])

        #Check Convergence
        print('Checking for convergence: {}'.format(np.array_equal(newCntr,clusterCntr)))
        if np.array_equal(newCntr,clusterCntr):
            print('K means has converged')
            print('----------------')
            break
        
        clusterCntr = newCntr
        print('-------------')
    pixels[:] = clusterCntr[labels]
    return pixels

#Calucate the compression ratio in terms of bytes
def CompressionRatio(originalImg,kMeanImg):
    originalImgSize = os.path.getsize(originalImg) #gets the size in bytes
    compressedImgSize = os.path.getsize(kMeanImg) #gets the size in bytes
    compressionRatio = originalImgSize / compressedImgSize
    return compressionRatio

if __name__ == "__main__":
    main()
