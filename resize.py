from os import listdir, mkdir
from os.path import isdir
from shutil import rmtree
from PIL import Image
from sys import argv
from pathlib import Path
from tqdm import tqdm


"""Resizes all images in all subdirectories of the directory provided as the
   command line argument when running the script."""

# Checking if a directory was provided as an argument
if len(argv) < 2:
    print(
        "Please provide a parent directory to resize the subdirectories of as an argument:"
    )
    print("python resize.py [path]")
else:
    data_dir = argv[1] + "/"
    # Checking if provided path exists
    if not Path(data_dir).exists():
        print("Provided path does not exist")
    elif not isdir(data_dir):
        print("Provided path is not a directory")
    else:
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
            # subdirectory. tqdm() provides a progress bar.
            for img_name in tqdm(img_names):
                img = Image.open(path + "/" + img_name)
                img.thumbnail((600, 600), resample=False)
                img.save(resized_path + img_name)
