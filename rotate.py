from os import listdir, mkdir
from shutil import rmtree
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
from matplotlib import cm


def degrees_to_rotate(image):
    '''
    Input: Masked image
    Returns which degrees an image should be rotated, in order to be able to fold it as
    "symmestrically" as possible . Returns the degree.
    '''
    max_height = 0
    degrees = 0
    for i in range(5, 181, 5):
        height_mask = transform.rotate(image, i)
        pixels_in_col = np.sum(height_mask, axis=0)
        max_pixels_in_col = np.max(pixels_in_col)
        if max_pixels_in_col > max_height:
            max_height = max_pixels_in_col
            degrees = i

    img_rotated = transform.rotate(image, degrees)
    white_mask = np.where(img_rotated == 1)
    max_x, min_x = max(white_mask[0]), min(white_mask[0])
    max_y, min_y = max(white_mask[1]), min(white_mask[1])
    img_rotated = img_rotated[min_x:max_x + 1, min_y:max_y + 1]
    return img_rotated


try:
    rmtree("rotated_data")
except FileNotFoundError:
    pass
mkdir("rotated_data")
path = "resized_data/example_segmentation_resized"
img_names = [file for file in listdir(path) if file[0] != "."]
rotated_path = "rotated_data/example_segmentation_rotated/"
mkdir(rotated_path)
for img_name in tqdm(img_names):
    img = plt.imread(path + "/" + img_name)
    img = degrees_to_rotate(img)
    img_pil = Image.fromarray(np.uint8(cm.gist_earth(img) * 255))
    img_pil.save(rotated_path + img_name)
