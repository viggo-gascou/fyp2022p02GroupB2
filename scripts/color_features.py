import numpy as np
import math
from PIL import Image
from skimage.segmentation import slic


def segment_colors(img_file, seg_file):
    img = Image.open(seg_file).convert("RGB")
    mask = Image.open(seg_file)
    img.paste(Image.open(img_file).convert("RGB"), mask=mask)
    img_arr = np.array(img, dtype=float)
    segments = slic(img_arr, n_segments=10, sigma=5, mask=mask)
    avg_color = np.vstack([np.round(
        np.mean(img_arr[segments == s], axis=0)) for s in np.unique(segments)])
    avg_color = avg_color[np.where((avg_color != (0, 0, 0)).all(axis=1))]
    return avg_color


def avg_color_dist(color_arr):
    """
    Computes the Euclidean distance between all colors in the average color arr.
    takes average color array as input.
    returns average distance as float.
    """
    distances = []
    for i, color in enumerate(color_arr):
        for j in range(i + 1, len(color_arr)):
            distances.append(math.dist(color_arr[i], color_arr[j]))
    return sum(distances) / len(distances)


def color_features(img_file, seg_file):
    avg_color = segment_colors(img_file, seg_file)
    avg_dist = avg_color_dist(avg_color)
    color_sd = np.mean(np.std(avg_color, axis=0))
    return avg_dist, color_sd

