import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def measure(seg_file):
    """Measures the area and perimiter of a segmentation mask with the given path"""
    perimeter = 0
    # Pad the image, in case the mask extends to the edge of the image
    img = np.pad(plt.imread(seg_file), 3)
    # Kernel for finding edges in the mas
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # Convolve image and kernel, resulting in an array where the edge pixels all
    # have a negative value. Sum the number of pixels with negative value.
    perimeter = len(np.where(convolve(img, kernel) < 0)[0])
    # Area is the sum of the array elements with value 1
    area = np.sum(img)
    return (area, perimeter)
