import pandas as pd
from PIL import Image
from os import listdir
from tqdm import tqdm
from math import pi


def measure(seg_file):
    area = 0
    perimiter = 0
    img = Image.open(seg_file)
    for x in range(img.size[0] - 1):
        for y in range(img.size[1] - 1):
            if img.getpixel((x, y)) != 0:
                area += 1
                surround = [
                    (x, y + 1),
                    (x, y - 1),
                    (x + 1, y),
                    (x - 1, y),
                    (x - 1, y - 1),
                    (x - 1, y + 1),
                    (x + 1, y - 1),
                    (x + 1, y + 1),
                ]
                surround = [img.getpixel(pix) for pix in surround]
                if 0 in surround:
                    perimiter += 1
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
df.to_csv("data/area_perim.csv", index=False)
