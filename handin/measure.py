import numpy as np
from feature_functions import asymmetry, area_perimeter, color_dist_sd, color_score, border_score
from math import pi
from PIL import Image


def measure(img, seg):
    # Open the segmentation file as RGB and as bitmap
    image = Image.fromarray(np.uint8(img)).convert("RGB")
    img = Image.fromarray(np.uint8(seg) * 255).convert("RGB")
    mask = Image.fromarray(np.uint8(seg) * 255)
    # Paste the image onto the RGB mask using the bitmap mask as a mask
    # Resulting image is the masked lesion in RGB colours
    img.paste(image, mask=mask)
    # Resize image and segmentation
    img.thumbnail((600, 600), resample=False)
    mask.thumbnail((600, 600), resample=False)
    img = np.array(img) / 255
    seg = np.array(mask) / 255
    asym, asym_gauss = asymmetry(seg)
    area, perim = area_perimeter(seg)
    col_dist_10_5, col_sd_10_5 = color_dist_sd(img, mask, 10, 5)
    col_dist_10_10, col_sd_10_10 = color_dist_sd(img, mask, 10, 10)
    col_dist_5_5, col_sd_5_5 = color_dist_sd(img, mask, 5, 5)
    col_dist_5_10, col_sd_5_10 = color_dist_sd(img, mask, 5, 10)
    col_score = color_score(img)
    print(perim)
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
