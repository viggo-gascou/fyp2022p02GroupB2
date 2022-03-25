import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from shutil import rmtree
from os import makedirs, listdir
from PIL import Image
from skimage import transform
from skimage.filters import gaussian
from tqdm.contrib.concurrent import process_map


# This script uses a lot of memory, especially for larger pictures


def rotate(img):
    '''
    Takes segmentation mask array as input, determines the best rotation degree
    for the segmentation for it to be folded as "symetrically" as possible.
    Return the rotated array
    '''
    max_height = 0
    max_deg = 0

    # Add padding to image before rotating to ensure entire skin lesion stays in frame.
    shape = img.shape
    width_add, height_add = int(shape[0] * 0.25), int(shape[1] * 0.25)
    image = np.pad(img, (width_add, height_add), constant_values=(0, 0))

    for i in range(5, 181, 5):
        # Rotate image in intervals of 5 degrees and save max height.
        height_mask = transform.rotate(image, i)
        pixels_in_col = np.sum(height_mask, axis=0)
        max_pixels_in_col = np.max(pixels_in_col)
        if max_pixels_in_col > max_height:
            max_height = max_pixels_in_col
            max_deg = i

    # Rotate image
    img_rot = transform.rotate(image, max_deg)

    # Gaussian blur to smooth edges
    sigma = min(shape) * 0.004
    for i in range(3):
        img_rot = gaussian(img_rot, sigma=(sigma, sigma))

    img_rot[img_rot > 0.1] = 1
    img_rot[img_rot <= 0.1] = 0

    # Mask where skin lesion is, find corners to crop image close to lesion.
    white_mask = np.where(img_rot == 1)
    max_x, min_x = max(white_mask[0]), min(white_mask[0])
    max_y, min_y = max(white_mask[1]), min(white_mask[1])
    # Crop image
    img_rot = img_rot[min_x:max_x + 1, min_y:max_y + 1]
    return img_rot


def rotate_to_pil(img_name):
    "Takes an image name as argument and reads, rotates and saves the image"
    img = plt.imread(path + img_name)
    img = rotate(img)
    img = Image.fromarray(np.uint8(img * 255))
    img.save(rot_path + img_name)


# Has to be outside the if __name__ == '__main__' for processes to access

# path = "resized_data/example_segmentation_resized/"
# rot_path = "rotated_data/example_segmentation_rotated/"
path = "fullsize_segmentation/"
rot_path = "rotated_data/example_segmentation_fullsize_rotated/"

if __name__ == '__main__':
    # Remove the target directory if it exists and create it again to start
    # with an empty directory
    try:
        rmtree(rot_path)
    except FileNotFoundError:
        pass
    makedirs(rot_path, exist_ok=True)
    start = perf_counter()
    # Read the file names in the input path, excluding hidden files
    img_names = [file for file in listdir(path) if file[0] != "."]
    # Rotate and save all images in the directory, using a multiprocessing
    # pool map with progress bar
    process_map(rotate_to_pil, img_names)

    end = perf_counter()
    print(end - start)
