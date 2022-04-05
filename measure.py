import numpy as np
import matplotlib.pyplot as plt
from feature_functions import asymmetry, area_perimeter, color_dist_sd, color_score, border_score
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
    col_dist_10_5, col_sd_10_5 = color_dist_sd(img, mask, 10, 5)
    col_dist_10_10, col_sd_10_10 = color_dist_sd(img, mask, 10, 10)
    col_dist_5_5, col_sd_5_5 = color_dist_sd(img, mask, 5, 5)
    col_dist_5_10, col_sd_5_10 = color_dist_sd(img, mask, 5, 10)
    col_score = color_score(img)
    compactness = (4 * pi * area) / perim ** 2
    border = border_score(seg)
    features = np.array(
        [
            asym,
            asym_gauss,
            area,
            perim,
            compactness,
            col_dist_10_5,
            col_sd_10_5,
            col_dist_10_10,
            col_sd_10_10,
            col_dist_5_5,
            col_sd_5_5,
            col_dist_5_10,
            col_sd_5_10,
            col_score,
            border
        ]
    )
    return features
