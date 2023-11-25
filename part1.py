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
    input_image_path = os.path.join(os.getcwd(),'Koala.jpg')
    #k = int(sys.argv[2])
    k = 15
    output_image_path = 'compressedKoala.jpg'
    #output_image_path = os.getcwd()
    print('outputImage path: {}'.format(output_image_path))
    #output_image_path = sys.argv[3]

    try:
        original_image = Image.open(input_image_path)
        kmeans_image = kmeans_helper(original_image, k)
        kmeans_image.save(output_image_path)
        compratio = calcCompRatio(original_image,kmeans_image)
        print(compratio)

    except IOError as e:
        print(e)

def kmeans_helper(original_image, k):
    w, h = original_image.size
    kmeans_image = Image.new('RGB', (w, h))

    # Read RGB values from the image
    pixels = np.array(original_image).reshape((w * h, 3))

    # Call kmeans algorithm: update the RGB values
    kmeans(pixels, k, w, h)

    # Write the new RGB values to the image
    pixels = pixels.reshape((w, h, 3))
    kmeans_image = Image.fromarray(np.uint8(pixels))

    return kmeans_image

def kmeans(pixels, k, w, h):
    # Your k-means code goes here
    # Update the array pixels by assigning each entry in the pixels array to its cluster center

    
    clusterCntrs = random.sample(list(pixels),k)
    for i in range(1000000):
        #print('round {} in for loop'.format(i))
        assgmnts = assignCluster(pixels,clusterCntrs) #Assign pixel to nearest cluster

        newCntrs = updateCluster(pixels,assgmnts, k) #Change the centers
        print('checking conditions: {}'.format(np.array_equal(newCntrs,clusterCntrs)))
        if np.array_equal(newCntrs,clusterCntrs): #Check if its at convergence
            break
        clusterCntrs = newCntrs

    finAssgmnts = assignCluster(pixels,clusterCntrs) #Assign pix to fin clusters
    compPix = np.array([clusterCntrs[clusterId] for clusterId in finAssgmnts])
    compPix = compPix.reshape((w,h,3))
    compImg = Image.fromarray(np.uint8(compPix))

    return compImg

if __name__ == "__main__":
    main()
