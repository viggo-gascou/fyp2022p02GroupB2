import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from measure import measure
from tqdm.contrib.concurrent import process_map


def wrapper(img_name):
    # Wrapper function for processes, loads image and segmentation and returns measured features
    img = plt.imread(f"../data_to_resize/training_image/{img_name}.jpg")
    seg = plt.imread(f"../data_to_resize/training_segmentation/{img_name}_segmentation.png")
    return measure(img, seg)


if __name__ == "__main__":
    # Read the example ground thruth csv to get image ids and melanoma data
    df = pd.read_csv("../data/training_ground_truth.csv")[["image_id", "melanoma"]]

    print("Measuring features...")
    # Measure features using custom functions for each image and segmentation file
    # process_map facilitates a progress bar and multiprocessing
    features = np.vstack(process_map(wrapper, list(df["image_id"]), chunksize=5))
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
    df.to_csv("../features/features_training.csv", index=False)
