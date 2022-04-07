import pandas as pd
import matplotlib.pyplot as plt
from classify import classify
from tqdm.contrib.concurrent import process_map


def wrapper(img_name):
    img = plt.imread(f"../data_to_resize/test_image/{img_name}.jpg")
    seg = plt.imread(f"../data_to_resize/test_segmentation/{img_name}_segmentation.png")
    return classify(img, seg)


if __name__ == "__main__":
    df = pd.read_csv("../data/test_ground_truth.csv")[["image_id"]]
    labels, probabilities = list(zip(*process_map(wrapper, list(df["image_id"]))))
    df["label"] = labels
    df["probabilities"] = probabilities
    df.to_csv("features/classification_test.csv", index=False)
