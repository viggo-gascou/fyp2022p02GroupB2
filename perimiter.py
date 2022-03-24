import pandas as pd
import numpy as np
from PIL import Image
from os import listdir
from tqdm import tqdm
from math import pi


def measure(seg_file):
    perimiter = 0
    img = np.array(Image.open(seg_file))
    img[img != 0] = 1
    for x1, x2 in zip(range(img.shape[0] - 1), range(3, img.shape[0] - 1)):
        for y1, y2 in zip(range(img.shape[1] - 1), range(3, img.shape[1] - 1)):
            view = img[x1:x2, y1:y2]
            if view[1, 1] and np.sum(view) < 9:
                perimiter += 1
    area = np.sum(img)
    return (area, perimiter)


seg_path = "resized_data/example_segmentation_resized/"
seg_names = sorted(listdir(seg_path))
img_names = [seg[:-17] for seg in seg_names]
areas = []
perimiters = []
for seg in tqdm(seg_names):
    area, perim = measure(seg_path + seg)
    areas.append(area)
    perimiters.append(perim)
columns = {"img": img_names, "area": areas, "perimiter": perimiters}
df = pd.DataFrame(columns)
df["compactness"] = df["perimiter"] ** 2 / ((4 * pi) * df["area"])
df.to_csv("features/area_perim.csv", index=False)
