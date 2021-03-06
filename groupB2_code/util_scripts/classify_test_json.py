import pandas as pd
import matplotlib.pyplot as plt
from groupB2_classify import classify_json
from tqdm.contrib.concurrent import process_map
from PIL import Image


def wrapper(img_name):
    """Wrapper function for multiprocessing
       Measures features for given image and returns the classification"""
    img = plt.imread(f"../data_to_resize/test_image/{img_name}.jpg")
    seg = plt.imread(f"../data_to_resize/test_segmentation/{img_name}_segmentation.png")
    spmask = Image.open(f"../data/test_superpixels/{img_name}_superpixels.png")
    json_df = pd.read_json(f"../data/test_json/{img_name}_features.json")
    return classify_json(img, seg, spmask, json_df)


if __name__ == "__main__":
    # Read the image ids for the test images
    df = pd.read_csv("../data/test_ground_truth.csv")[["image_id"]]
    # CLassify all test images (process_map maps wrapper function to image ids and gives a progress bar and multiprocessing)
    labels, probabilities = list(zip(*process_map(wrapper, list(df["image_id"]))))
    # Store the results in the data frame and save to csv
    df["label"] = labels
    df["probabilities"] = probabilities
    df.to_csv("classifications/classification_json_test.csv", index=False)
