import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from time import perf_counter
from shutil import rmtree
from os import makedirs, listdir
from PIL import Image
from skimage import transform


path = "resized_data/example_segmentation_resized/"
rot_path = "rotated_data/example_segmentation_rotated/"


def rotate(image):
    '''
    Input: Masked image
    Returns which degrees an image should be rotated, in order to be able to fold it as
    "symmestrically" as possible . Returns the degree.
    '''
    max_height = 0
    degrees = 0
    for i in range(5, 181, 5):
        # Rotate image in intervals of 5 degrees and save max height.
        height_mask = transform.rotate(image, i)
        pixels_in_col = np.sum(height_mask, axis=0)
        max_pixels_in_col = np.max(pixels_in_col)
        if max_pixels_in_col > max_height:
            max_height = max_pixels_in_col
            degrees = i

    # Add padding to image before rotating to ensure entire skin lesion stays in frame.
    shape = image.shape
    width_add, height_add = int(shape[0] * 0.25), int(shape[1] * 0.25)
    image = np.pad(image, (width_add, height_add), constant_values=(0, 0))

    # Rotate image
    img_rotated = transform.rotate(image, degrees)
    # Mask where skin lesion is, find corners to crop image close to lesion.
    white_mask = np.where(img_rotated == 1)
    max_x, min_x = max(white_mask[0]), min(white_mask[0])
    max_y, min_y = max(white_mask[1]), min(white_mask[1])
    # Crop image
    img_rotated = img_rotated[min_x:max_x + 1, min_y:max_y + 1]
    return img_rotated


def rotate_to_pil(img):
    img = rotate(img)
    img = Image.fromarray(np.uint8(img * 255))
    return img


if __name__ == '__main__':
    try:
        rmtree(rot_path)
    except FileNotFoundError:
        pass
    makedirs(rot_path, exist_ok=True)
    start = perf_counter()
    img_names = [file for file in listdir(path) if file[0] != "."]
    img_files = [path + img for img in img_names]
    img_arrs = [plt.imread(img) for img in img_files]

    p = Pool()
    rot_imgs = list(p.map(rotate_to_pil, img_arrs))

    for img, img_name in zip(rot_imgs, img_names):
        img.save(rot_path + img_name)

    end = perf_counter()
    print(end - start)
