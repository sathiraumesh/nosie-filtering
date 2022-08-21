
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.filters
import math
import uuid


def main():
    image = skimage.io.imread("objects.jpeg")
    gray_image = skimage.color.rgb2gray(image)

    cropped_image = np.array(gray_image.flat)
    min_val = np.max(cropped_image) / 256
    cropped_image[cropped_image < min_val] = min_val
    im = np.log(cropped_image)
    min_val = np.min(im)
    max_val = np.max(im)
    im = (im - min_val) / (max_val - min_val)

    pre_thresh = 0

    new_thresh = np.mean(gray_image)
    delta = 0.00001
    while abs(pre_thresh - new_thresh) > delta:
        pre_thresh = new_thresh
        mean1 = np.mean(im[im < pre_thresh])
        mean2 = np.mean(im[im >= pre_thresh])
        new_thresh = np.mean([mean1, mean2])
    t = math.exp(min_val + (max_val - min_val) * new_thresh)

    binary_mask = gray_image < t

    plt.imshow(binary_mask, cmap="gray")
    plt.savefig(str(uuid.uuid4())+".png")


if __name__ == "__main__" :
     main()