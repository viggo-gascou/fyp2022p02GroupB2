import pandas as pd
import matplotlib.pyplot as plt
from classify import classify
from tqdm.contrib.concurrent import process_map


def wrapper(img_name):
    img = plt.imread(f"../data/example_image/{img_name}.jpg")
    seg = plt.imread(f"../data/example_segmentation/{img_name}_segmentation.png")
    return classify(img, seg)[0]


if __name__ == "__main__":
    df = pd.read_csv("../data/example_ground_truth.csv")[["image_id"]]
    labels = process_map(wrapper, list(df["image_id"]))
    df["label"] = labels
    df.to_csv("features/labels_example.csv")
