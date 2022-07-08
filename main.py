import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import cv2
import math
np.seterr(over='ignore')

def main(): 
    imdir = '/Users/sathira/Downloads/Mean-and-Median-Filter-master/env/images/'
    outdir = '/Users/sathira/Downloads/Mean-and-Median-Filter-master/env/filtered_images/'
    ext = ['png', 'jpg', 'gif', 'jpeg']
    kernel_size = 3
    files = []
    [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
   
    images = [cv2.imread(file, 0) for file in files]

    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(8)

    for index, image in enumerate(images):
        mean_filter_image = mean_filter(image, 3)
        medain_filter_image = medain_filter(image, 3)
        midpoint_filter_image = midpoint_filter(image, 3)

        cv2.imwrite(outdir+"mean"+str(index)+'.png', mean_filter_image)
        cv2.imwrite(outdir+"median"+str(index)+'.png', medain_filter_image)
        cv2.imwrite(outdir+"midpoint"+str(index)+'.png',  midpoint_filter_image)

def mean_filter(image, kernel_size):
    print("Applying Mean Filter")

    validate_kernel(kernel_size)
    result_image = np.zeros(image.shape, np.uint8) 
    distance_from_piviot = kernel_size//2
    
    result = 0
    for col in range(distance_from_piviot, image.shape[0] - distance_from_piviot):
        for row in range(distance_from_piviot, image.shape[1] - distance_from_piviot):
            for col_kernel in range(-distance_from_piviot, kernel_size - distance_from_piviot):
                 for row_kernel in range(-distance_from_piviot, kernel_size - distance_from_piviot):
                    result = result + image[col_kernel + col, row_kernel + row]
                   
            result_image[col][row] = int(result/(kernel_size*kernel_size))
            result = 0
    return result_image

def medain_filter(image, kernel_size): 
    print("Applying Median Filter")

    validate_kernel(kernel_size)
    result_image = np.zeros(image.shape, np.uint8) 
    filter_array = [image[0][0]]*kernel_size*kernel_size

    distance_from_piviot = kernel_size//2
    for col in range(distance_from_piviot, image.shape[0] - distance_from_piviot):
        for row in range(distance_from_piviot, image.shape[1] - distance_from_piviot):
            filter_array_index = 0
            for col_kernel in range(-distance_from_piviot, kernel_size - distance_from_piviot):
                for row_kernel in range(-distance_from_piviot, kernel_size - distance_from_piviot):
                    filter_array[filter_array_index] = image[col_kernel + col, row_kernel + row]
                    filter_array_index += 1
            
            filter_array.sort()

            median_index = len(filter_array)//2
            result_image[col, row] = filter_array[median_index]
    return result_image

def midpoint_filter(image, kernel_size): 
    print("Applying Midpoint Filter")

    validate_kernel(kernel_size)
    result_image = np.zeros(image.shape, np.uint8) 
    filter_array = [image[0][0]]*kernel_size*kernel_size

    distance_from_piviot = kernel_size//2
    for col in range(distance_from_piviot, image.shape[0] - distance_from_piviot):
        for row in range(distance_from_piviot, image.shape[1] - distance_from_piviot):
            filter_array_index = 0
            for col_kernel in range(-distance_from_piviot, kernel_size - distance_from_piviot):
                for row_kernel in range(-distance_from_piviot, kernel_size - distance_from_piviot):
                    filter_array[filter_array_index] = image[col_kernel + col, row_kernel + row]
                    filter_array_index += 1
            
            filter_array.sort()
            result_image[col, row] = int((filter_array[len(filter_array) - 1] + filter_array[0])/2)
    return result_image


def validate_kernel(kernel_size): 
    if kernel_size <=1 :
        raise Exception("Minimum kernel is 3")
    elif kernel_size%2 !=1:
        raise Exception("Kernel should be odd")


if __name__ == "__main__" :
     main()