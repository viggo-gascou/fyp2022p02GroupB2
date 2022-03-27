import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import math
from PIL import Image
from skimage.segmentation import slic


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


def segment_colors(img, mask):
    """Given an image file and corresponding segmentation mask, applies the mask
       to the image and segments the lesion into 10 zones by colour. Calculates
       the average color for each zone and returns an array of these values."""
    # Convert image to array and segment into 10 segments using simgma=5
    img_arr = np.array(img, dtype=float)
    segments = slic(img_arr, n_segments=10, sigma=5, mask=mask)
    # Find the mean RGB values of each segment and stack into one array
    avg_color = np.vstack([np.round(
        np.mean(img_arr[segments == s], axis=0)) for s in np.unique(segments)])
    # If there is an entry with mean color pure black, remove it
    avg_color = avg_color[np.where((avg_color != (0, 0, 0)).all(axis=1))]
    return avg_color


def avg_color_dist(color_arr):
    """
    Computes the Euclidean distance between all colors in the average color arr.
    Takes average color array as input and returns average distance as float.
    """
    distances = []
    for i, color in enumerate(color_arr):
        for j in range(i + 1, len(color_arr)):
            distances.append(math.dist(color_arr[i], color_arr[j]))
    return sum(distances) / len(distances)


def color_features(img, mask):
    avg_color = segment_colors(img, mask)
    avg_dist = avg_color_dist(avg_color)
    # Computes standard deviation of each channel of R, G and B
    color_sd = np.mean(np.std(avg_color, axis=0))
    return avg_dist, color_sd


def color_score(img):
    # Convert image to array with RGB values between [0, 1]
    img_arr = np.array(img) / 255
    # Remove all black pixels
    lesion = img_arr[np.where(~(
        (img_arr[:, :, 0] == 0)
        & (img_arr[:, :, 1] == 0)
        & (img_arr[:, :, 2] == 0))
    )]
    # Count non-black pixels in lesion
    pixels = len(lesion)
    color_count = 0
    # Masks for detecting the six colors, from (Majumder & Ullah, 2019)
    masks = {
        "white": (lesion[:, 0] >= 0.8) & (lesion[:, 1] >= 0.8) & (lesion[:, 2] >= 0.8),
        "red": (lesion[:, 0] >= 0.588) & (lesion[:, 1] >= 0.2) & (lesion[:, 2] >= 0.2),
        "lightbrown": (lesion[:, 0] >= 0.588)
        & (lesion[:, 1] > 0.2)
        & (lesion[:, 2] > 0)
        & (lesion[:, 0] <= 0.94)
        & (lesion[:, 1] <= 0.588)
        & (lesion[:, 2] < 0.392),
        "darkbrown": (lesion[:, 0] > 0.243)
        & (lesion[:, 1] >= 0)
        & (lesion[:, 2] > 0)
        & (lesion[:, 0] < 0.56)
        & (lesion[:, 1] < 0.392)
        & (lesion[:, 2] < 0.392),
        "blue-gray": (lesion[:, 0] >= 0)
        & (lesion[:, 1] >= 0.392)
        & (lesion[:, 2] >= 0.490)
        & (lesion[:, 0] <= 0.588)
        & (lesion[:, 1] <= 0.588)
        & (lesion[:, 2] <= 0.588),
        "black": (lesion[:, 0] <= 0.243)
        & (lesion[:, 1] <= 0.243)
        & (lesion[:, 2] <= 0.243),
    }
    # Check for each color if amount of pixels of that color is over 5%
    # Add 1 to the color score if true
    for mask in masks.values():
        if len(lesion[mask]) / pixels > 0.05:
            color_count += 1
    return color_count
