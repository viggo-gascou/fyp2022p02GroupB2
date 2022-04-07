import pandas as pd
import numpy as np
from PIL import Image
from groupB2_functions import measure_json
from tqdm.contrib.concurrent import process_map


def wrapper(img_name):
    # Wrapper function for processes, loads image and segmentation and returns measured features
    spmask = Image.open(f"../data/training_superpixels/{img_name}_superpixels.png")
    json_df = pd.read_json(f"../data/training_json/{img_name}_features.json")
    return measure_json(spmask, json_df)


if __name__ == "__main__":
    # Read the example ground thruth csv to get image ids and melanoma data
    df = pd.read_csv("../data/training_ground_truth.csv")[["image_id", "melanoma"]]
    feature_names = ["pigment_network_percent", "negative_network_percent", "milia_like_percent", "streaks_percent"]
    feature_names.extend([feat[:-7] + "hist" for feat in feature_names])
    print("Measuring features...")
    # Measure features using custom functions for each superpixel mask and json metadata file
    # process_map facilitates a progress bar and multiprocessing
    features = np.vstack(process_map(wrapper, list(df["image_id"]), chunksize=5))
    for feat, col in zip(feature_names, features.T):
        df[feat] = col
    df.to_csv("../features/json_training.csv", index=False)
