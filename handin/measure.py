import numpy as np
from feature_functions import asymmetry, area_perimeter, color_dist_sd, color_score, border_score, percent_json, hist_json
from math import pi
from PIL import Image


def measure(img, seg):
    # Convert the image and segmentation to PIL.Image format
    image = Image.fromarray(np.uint8(img)).convert("RGB")
    img = Image.fromarray(np.uint8(seg) * 255).convert("RGB")
    mask = Image.fromarray(np.uint8(seg) * 255)
    # Paste the image onto the RGB mask using the bitmap mask as a mask
    # Resulting image is the masked lesion in RGB colours
    img.paste(image, mask=mask)
    # Resize image and segmentation
    img.thumbnail((600, 600), resample=False)
    mask.thumbnail((600, 600), resample=False)
    # Convert image and segmentation back to numpy arrays
    img = np.array(img) / 255
    seg = np.array(mask) / 255
    # Measure asymmetry scores
    asym, asym_gauss = asymmetry(seg)
    # Measure area and perimiter
    area, perim = area_perimeter(seg)
    # Measure average color distance and standard deviation for different number of
    # segmenations and sigma scores
    col_dist_10_5, col_sd_10_5 = color_dist_sd(img, mask, 10, 5)
    col_dist_10_10, col_sd_10_10 = color_dist_sd(img, mask, 10, 10)
    col_dist_5_5, col_sd_5_5 = color_dist_sd(img, mask, 5, 5)
    col_dist_5_10, col_sd_5_10 = color_dist_sd(img, mask, 5, 10)
    # Measure color score for image
    col_score = color_score(img)
    # Calculate compactness
    compactness = (4 * pi * area) / perim ** 2
    # Measure border score (closeness to a circle)
    border = border_score(seg)
    # Return an array of all the features
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


def measure_selected(img, seg):
    # Convert the image and segmentation to PIL.Image format
    image = Image.fromarray(np.uint8(img)).convert("RGB")
    img = Image.fromarray(np.uint8(seg) * 255).convert("RGB")
    mask = Image.fromarray(np.uint8(seg) * 255)
    # Paste the image onto the RGB mask using the bitmap mask as a mask
    # Resulting image is the masked lesion in RGB colours
    img.paste(image, mask=mask)
    # Resize image and segmentation
    img.thumbnail((600, 600), resample=False)
    mask.thumbnail((600, 600), resample=False)
    # Convert image and segmentation back to numpy arrays
    img = np.array(img) / 255
    seg = np.array(mask) / 255
    # Measure asymmetry scores
    asym, _ = asymmetry(seg)
    # Measure area and perimiter
    area, perim = area_perimeter(seg)
    # Measure average color distance and standard deviation for different number of
    # segmenations and sigma scores
    col_dist_10_5, _ = color_dist_sd(img, mask, 10, 5)
    # Measure color score for image
    col_score = color_score(img)
    # Return an array of all the features
    features = np.array(
        [
            asym,
            area,
            perim,
            col_dist_10_5,
            col_score,
        ]
    )
    return features


def measure_json(spmask, df):
    spmask.thumbnail((100, 100), resample=False)
    spmask = np.array(spmask)
    hists = hist_json(spmask, df)
    percentages = percent_json(df)
    return percentages + hists
