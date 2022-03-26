import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def measure(seg_file):
    perimeter = 0
    img = np.pad(plt.imread(seg_file), 2)
    kernel = np.array([[-1, -1, 0], [-1, 8, -1], [-1, -1, -1]])
    perimeter = len(np.where(convolve(img, kernel) < 0)[0])
    area = np.sum(img)
    return (area, perimeter)

