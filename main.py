import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import cv2
import math
import os
np.seterr(over='ignore')

def main(): 
    imdir = '/Users/sathira/Downloads/Mean-and-Median-Filter-master/env/images/'
    outdir = '/Users/sathira/Downloads/Mean-and-Median-Filter-master/env/filtered_images/'
    ext = ['png', 'jpg', 'gif', 'jpeg']
    kernel_size = 3

    images = {}
    files = []
    [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
   
    for file in files:
        file_name = os.path.basename(file)
        image = cv2.imread(file, 0)
        images[file_name] = image

    for fileName, image in images.items():
        print(fileName)
        mean_filter_image = mean_filter(image, 3, True)
        medain_filter_image = medain_filter(image, 3, True)
        midpoint_filter_image = midpoint_filter(image, 3, True)

        cv2.imwrite(outdir+"mean_"+fileName, mean_filter_image)
        cv2.imwrite(outdir+"median_"+fileName, medain_filter_image)
        cv2.imwrite(outdir+"midpoint_"+fileName,  midpoint_filter_image)

def mean_filter(image, kernel_size, wrap=False):
    print("Applying Mean Filter")
    distance_from_piviot = kernel_size//2

    result_image = np.zeros(image.shape, np.uint8) 

    if wrap :
        image = apply_wrapping(image, distance_from_piviot)
        
    validate_kernel(kernel_size)

    result = 0
    for col in range(distance_from_piviot, image.shape[0] - distance_from_piviot):
        for row in range(distance_from_piviot, image.shape[1] - distance_from_piviot):
            for col_kernel in range(-distance_from_piviot, kernel_size - distance_from_piviot):
                 for row_kernel in range(-distance_from_piviot, kernel_size - distance_from_piviot):
                    result = result + image[col_kernel + col, row_kernel + row]
            if wrap :
                result_image[col - distance_from_piviot][row - distance_from_piviot] = int(result/(kernel_size*kernel_size))
            else:
                result_image[col][row] = int(result/(kernel_size*kernel_size))
            result = 0
    return result_image

def medain_filter(image, kernel_size, wrap=False): 
    print("Applying Median Filter")

    distance_from_piviot = kernel_size//2

    result_image = np.zeros(image.shape, np.uint8) 

    if wrap :
        image = apply_wrapping(image, distance_from_piviot)
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
            if wrap: 
                  result_image[col - distance_from_piviot, row -distance_from_piviot] = filter_array[median_index]
            else:
                result_image[col, row] = filter_array[median_index]
    return result_image

def midpoint_filter(image, kernel_size, wrap=False): 
    print("Applying Midpoint Filter")

    distance_from_piviot = kernel_size//2

    result_image = np.zeros(image.shape, np.uint8) 

    if wrap :
        image = apply_wrapping(image, distance_from_piviot)
    validate_kernel(kernel_size)
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
            if wrap: 
                result_image[col - distance_from_piviot, row - distance_from_piviot] = int((filter_array[len(filter_array) - 1] + filter_array[0])/2)
            else:
                result_image[col, row] = int((filter_array[len(filter_array) - 1] + filter_array[0])/2)
    return result_image


def validate_kernel(kernel_size): 
    if kernel_size <=1 :
        raise Exception("Minimum kernel is 3")
    elif kernel_size%2 !=1:
        raise Exception("Kernel should be odd")

def apply_wrapping(image, wrap_size):

    width, height = image.shape
    result = image.copy()

    for col in range(0, wrap_size):

        # wrap at the beigining of column
        end_start_point = 1
        new_column = image.transpose()[height - end_start_point]
        result = np.insert(result, 0, new_column, axis = 1)
        end_start_point +=1

        # wrap at the end of the column
        start_start_point = 0
        new_column = image.transpose()[0 + start_start_point].reshape(width, 1)
        result = np.append(result, new_column, axis = 1)
        start_start_point+=1

    for row in range(0, wrap_size):

        #wrap at the top row
        end_bottom_start_point = 1
        new_row = result[width - end_bottom_start_point]
        result = np.insert(result, 0, new_row, axis = 0)
        end_bottom_start_point +=1

        #wrap at the bottom of the row
        start_top_start = 0
        new_row = result[start_top_start]
        result = np.append(result, [new_row], axis = 0)
        end_bottom_start_point +=1

    # handle the edges

    for row in range(0, wrap_size):
        for column in range(0, wrap_size):
            result[row][column] = image[row][column]
            result[row][height + column] = image[row][height - wrap_size + column]
            result[width+row][column] = image[width - wrap_size + row][column]
            result[width+row][height + column] =  image[width - wrap_size + row][height - wrap_size + column]
    return result

if __name__ == "__main__" :
     main()