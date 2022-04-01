import numpy as np
from scipy.ndimage import convolve
import math
from skimage import transform
from skimage.segmentation import slic
from skimage.filters import gaussian


def area_perimeter(seg):
    """Measures the area and perimiter of a segmentation mask with the given path"""
    perimeter = 0
    # Pad the image, in case the mask extends to the edge of the image
    img = np.pad(seg, 3)
    # Kernel for finding edges in the mas
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # Convolve image and kernel, resulting in an array where the edge pixels all
    # have a negative value. Sum the number of pixels with negative value.
    perimeter = len(np.where(convolve(img, kernel) < 0)[0])
    # Area is the sum of the array elements with value 1
    area = np.sum(img)
    return (area, perimeter)


def segment_colors(img, mask, n_segments, sigma):
    """Given an image file and corresponding segmentation mask, applies the mask
       to the image and segments the lesion into 10 zones by colour. Calculates
       the average color for each zone and returns an array of these values."""
    # Convert image to array and segment into 10 segments using simgma=5
    segments = slic(img, n_segments=n_segments, sigma=sigma, mask=mask)
    # Find the mean RGB values of each segment and stack into one array
    avg_color = np.vstack([np.mean(img[segments == s], axis=0) for s in np.unique(segments)])
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


def color_dist_sd(img, mask, n_segments, sigma):
    avg_color = segment_colors(img, mask, n_segments, sigma)
    avg_dist = avg_color_dist(avg_color)
    # Computes standard deviation of each channel of R, G and B
    color_sd = np.mean(np.std(avg_color, axis=0))
    return avg_dist, color_sd


def color_score(img):
    # Remove all black pixels
    lesion = img[np.where(~(
        (img[:, :, 0] == 0)
        & (img[:, :, 1] == 0)
        & (img[:, :, 2] == 0))
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


def edge_percentage(img):
    p = int(img.shape[1] * 1 / 100)

    top_r = img[0:p, :]
    bottom_r = img[-p:-1, :]
    left_c = img[p + 1:-p - 1, 0:p]
    right_c = img[p + 1:-p - 1, -p:-1]

    total_size = top_r.size + bottom_r.size + left_c.size + right_c.size
    total_white = int(np.sum(top_r) + np.sum(bottom_r) + np.sum(left_c) + np.sum(right_c))
    return (total_white / total_size)


def rotate(img):
    '''
    Takes segmentation mask array as input, determines the best rotation degree
    for the segmentation for it to be folded as "symetrically" as possible.
    Return the rotated array
    '''
    max_height = 0
    max_deg = 0

    # Add padding to image before rotating to ensure entire skin lesion stays in frame.
    shape = img.shape
    width_add, height_add = int(shape[0] * 0.25), int(shape[1] * 0.25)
    image = np.pad(img, (width_add, height_add), constant_values=(0, 0))

    for i in range(5, 181, 5):
        # Rotate image in intervals of 5 degrees and save max height.
        height_mask = transform.rotate(image, i)
        pixels_in_col = np.sum(height_mask, axis=0)
        max_pixels_in_col = np.max(pixels_in_col)
        if max_pixels_in_col > max_height:
            max_height = max_pixels_in_col
            max_deg = i

    # Rotate image
    img_rot = transform.rotate(image, max_deg)

    # Mask where skin lesion is, find corners to crop image close to lesion.
    white_mask = np.where(img_rot == 1)
    max_x, min_x = max(white_mask[0]), min(white_mask[0])
    max_y, min_y = max(white_mask[1]), min(white_mask[1])
    # Crop image
    img_rot = img_rot[min_x:max_x + 1, min_y:max_y + 1]
    return img_rot


def asymmetry(img):
    if edge_percentage(img) > 0.25:
        return (4, 4)
    img = rotate(img)
    img_gauss = img.copy()
    # Gaussian blur to smooth edges, sigma of 0.004 times smallest dimension
    # was found through experimental testing
    sigma = min(img_gauss.shape) * 0.004
    for i in range(3):
        img_gauss = gaussian(img_gauss, sigma=(sigma, sigma))
    scores = []
    for arr in [img, img_gauss]:
        arr[arr > 0.1] = 1
        arr[arr <= 0.1] = 0
        total_pixels = np.sum(arr)
        arr_addv = arr + np.flip(arr, 1)
        arr_addh = arr + np.flip(arr, 0)
        score_vert = np.sum(arr_addv == 1) / total_pixels
        score_hor = np.sum(arr_addh == 1) / total_pixels
        scores.append(score_vert + score_hor)
    return tuple(scores)
