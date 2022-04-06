import pandas as pd
import numpy as np
from measure import measure
from tqdm.contrib.concurrent import process_map


def wrapper(tup):
    return measure(*tup)


# Paths to the directories with the images and corresponding segmentations to measure
img_path = "resized_data/example_image_resized/"
seg_path = "resized_data/example_segmentation_resized/"

if __name__ == "__main__":
    # Read the example ground thruth csv to get image ids and melanoma data
    df = pd.read_csv("data/example_ground_truth.csv")[["image_id", "melanoma"]]

    # Add paths and extensions to image ids
    img_files = [img_path + img + ".jpg" for img in df["image_id"]]
    seg_files = [seg_path + img + "_segmentation.png" for img in df["image_id"]]

    print("Measuring features...")
    # Measure features using custom functions for each image and segmentation file
    # tqdm and list for the zip() facilitates a progress bar
    features = np.vstack(process_map(wrapper, list(zip(img_files, seg_files)), chunksize=5))
    feature_names = [
        "asymmetry",
        "asymmetry_gauss",
        "area",
        "perimeter",
        "compactness",
        "color_dist_10_5",
        "color_sd_10_5",
        "color_dist_10_10",
        "color_sd_10_10",
        "color_dist_5_5",
        "color_sd_5_5",
        "color_dist_5_10",
        "color_sd_5_10",
        "color_score",
        "border_score"
    ]
    # Adding the columns of features to the data frame
    for col, feature in zip(features.T, feature_names):
        df[feature] = col
    # Changing data type for the integer columns
    types = {k: int for k in ["area", "perimeter", "color_score"]}
    df.astype(types)
    df.to_csv("features/features_example.csv", index=False)
