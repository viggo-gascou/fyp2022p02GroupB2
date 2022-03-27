import numpy as np
import math
from PIL import Image
from skimage.segmentation import slic


def segment_colors(img_file, seg_file):
    """Given an image file and corresponding segmentation mask, applies the mask
       to the image and segments the lesion into 10 zones by colour. Calculates
       the average color for each zone and returns an array of these values."""
    # Open the segmentation file as RGB and as bitmap
    img = Image.open(seg_file).convert("RGB")
    mask = Image.open(seg_file)
    # Paste the image onto the RGB mask using the bitmap mask as a mask
    # Resulting image is the masked lesion in RGB colours
    img.paste(Image.open(img_file).convert("RGB"), mask=mask)
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


def color_features(img_file, seg_file):
    avg_color = segment_colors(img_file, seg_file)
    avg_dist = avg_color_dist(avg_color)
    # Computes standard deviation of each channel of R, G and B
    color_sd = np.mean(np.std(avg_color, axis=0))
    return avg_dist, color_sd
