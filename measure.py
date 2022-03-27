import numpy as np
import matplotlib.pyplot as plt
from feature_functions import asymmetry, area_perimeter, color_dist_sd, color_score
from math import pi
from PIL import Image


def measure(img_file, seg_file):
    # Open the segmentation file as RGB and as bitmap
    img = Image.open(seg_file).convert("RGB")
    mask = Image.open(seg_file)
    # Paste the image onto the RGB mask using the bitmap mask as a mask
    # Resulting image is the masked lesion in RGB colours
    img.paste(Image.open(img_file).convert("RGB"), mask=mask)
    # Convert image to array with RGB values between [0, 1]
    img = np.array(img) / 255
    # Read segmentation as 1D array
    seg = plt.imread(seg_file)
    asym, asym_gauss = asymmetry(seg)
    area, perim = area_perimeter(seg)
    col_dist, col_sd = color_dist_sd(img, mask)
    col_score = color_score(img)
    compactness = (4 * pi * area) / perim ** 2
    features = np.array([asym, asym_gauss, area, perim, compactness, col_dist, col_sd, col_score])
    return features
