from os import listdir, mkdir
from os.path import isdir
from shutil import rmtree
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


"""Resizes all images in all subdirectories of the directory provided as the
   command line argument when running the script."""


def resize(img_name, path, resized_path):
    "Resizes image to 600 pixels in longest dimension"
    img = Image.open(path + "/" + img_name)
    img.thumbnail((600, 600), resample=False)
    img.save(resized_path + img_name)


def wrapper(tup):
    """Wrapper for process_map"""
    resize(*tup)


data_dir = "data_to_resize/"
if __name__ == "__main__":
    # Remove output directory if exists, to start from scratch
    try:
        rmtree("resized_data")
    except FileNotFoundError:
        pass
    # Make output directory
    mkdir("resized_data")
    # List subdirectories of the provided parent directory
    subdirs = listdir(data_dir)
    subdirs = [dir for dir in subdirs if isdir(data_dir + dir)]
    paths = list(map(lambda x: data_dir + x, subdirs))
    # Go through each subdirectory and its path
    for dir, path in zip(subdirs, paths):
        # List the image names in the subdirectory
        img_names = [file for file in listdir(path) if file[0] != "."]
        # Make output subdirectory
        resized_path = f"resized_data/{dir}_resized/"
        mkdir(resized_path)
        # Resize all images in the subdirectory and save them in the output
        # subdirectory. process_map provides a progress bar and multiprocesses.
        process_map(wrapper, list(zip(img_names, [path] * len(img_names), [resized_path] * len(img_names))), chunksize=5)
